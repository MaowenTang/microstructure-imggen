# -*- coding: utf-8 -*-
"""
ldm_microstructure_pipeline.py
Latent Diffusion pipeline for microstructure patches (default: grayscale 1-channel, 512x512)

Commands:
  1) train_vae : Train VAE -> latent (z_channels, 64, 64) for 512 input
  2) recon_vae : Reconstruct sample images using trained VAE
  3) train_ldm : Train DDPM UNet in latent space
  4) sample    : Sample from latent DDPM and decode using VAE

DDP examples:
  torchrun --standalone --nnodes=1 --nproc_per_node=4 ldm_microstructure_pipeline.py train_vae ...
  torchrun --standalone --nnodes=1 --nproc_per_node=4 ldm_microstructure_pipeline.py train_ldm ...
"""

import os
import math
import time
import glob
import json
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusers import UNet2DModel, DDPMScheduler


# =========================================================
# Utilities
# =========================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def list_images(root: str):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(files))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_rank():
    return int(os.environ.get("RANK", "0"))


def ddp_world():
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_setup():
    if not is_ddp():
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(ddp_local_rank())
    return True


def ddp_cleanup():
    if is_ddp() and dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return (not is_ddp()) or ddp_rank() == 0


def barrier():
    if is_ddp() and dist.is_initialized():
        dist.barrier()


def log0(msg: str):
    if is_main():
        print(msg, flush=True)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_pil(p: str, in_channels: int):
    im = Image.open(p)
    if in_channels == 1:
        if im.mode != "L":
            im = im.convert("L")
    elif in_channels == 3:
        if im.mode != "RGB":
            im = im.convert("RGB")
    else:
        raise ValueError(f"in_channels must be 1 or 3, got {in_channels}")
    return im


def pil_center_crop_resize(im: Image.Image, resolution: int):
    w, h = im.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    im = im.crop((left, top, left + s, top + s))
    if s != resolution:
        im = im.resize((resolution, resolution), Image.BICUBIC)
    return im


def to_tensor_m11(im: Image.Image, in_channels: int):
    arr = np.array(im)

    if in_channels == 1:
        # H,W -> 1,H,W
        x = torch.from_numpy(arr).float().unsqueeze(0) / 127.5 - 1.0
    else:
        # H,W,3 -> 3,H,W
        x = torch.from_numpy(arr).float().permute(2, 0, 1).contiguous() / 127.5 - 1.0

    return x


def tensor_to_pil_m11(x: torch.Tensor):
    # x: [C,H,W] in [-1,1]
    x = x.detach().cpu().clamp(-1, 1)

    if x.size(0) == 1:
        arr = ((x[0] + 1.0) * 127.5).to(torch.uint8).numpy()
        return Image.fromarray(arr, mode="L")
    elif x.size(0) == 3:
        arr = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).numpy()
        return Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Unsupported channel count in tensor_to_pil_m11: {x.size(0)}")


def save_ckpt(model, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"model": model.state_dict()}
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_ckpt(model, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


# =========================================================
# Dataset
# =========================================================
class PatchFolder(Dataset):
    def __init__(self, root: str, resolution: int = 512, in_channels: int = 1, flip: bool = True):
        self.paths = list_images(root)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {root}")

        self.res = resolution
        self.in_channels = in_channels
        self.flip = flip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = load_pil(p, self.in_channels)
        img = pil_center_crop_resize(img, self.res)

        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        x = to_tensor_m11(img, self.in_channels)
        return x


# =========================================================
# VAE
# =========================================================
class SimpleVAE(nn.Module):
    def __init__(self, in_channels=1, z_channels=4, base_ch=64):
        super().__init__()
        ch = base_ch

        # Encoder: 512 -> 256 -> 128 -> 64
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, ch, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.SiLU(),
        )

        self.to_mu = nn.Conv2d(ch * 4, z_channels, 1)
        self.to_logvar = nn.Conv2d(ch * 4, z_channels, 1)

        # Decoder: 64 -> 128 -> 256 -> 512
        self.dec = nn.Sequential(
            nn.Conv2d(z_channels, ch * 4, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, 1, 1),
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar, z


def vae_kl(mu, logvar):
    return 0.5 * torch.mean(torch.exp(logvar) + mu * mu - 1.0 - logvar)


def make_recon_canvas(x, x_rec, out_png, resolution):
    """
    x, x_rec: [N,C,H,W] in [-1,1]
    top row: original
    bottom row: reconstruction
    """
    n = x.size(0)
    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))

    canvas = Image.new("L" if x.size(1) == 1 else "RGB", (cols * resolution, rows * resolution * 2))

    for i in range(n):
        r, c = divmod(i, cols)
        im0 = tensor_to_pil_m11(x[i])
        im1 = tensor_to_pil_m11(x_rec[i])

        canvas.paste(im0, (c * resolution, r * resolution * 2))
        canvas.paste(im1, (c * resolution, r * resolution * 2 + resolution))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)


@torch.no_grad()
def vae_recon_grid(vae, dl, device, out_png, resolution=512, n=9):
    vae.eval()

    xs = []
    for x in dl:
        xs.append(x)
        total = sum(t.size(0) for t in xs)
        if total >= n:
            break

    x = torch.cat(xs, dim=0)[:n].to(device)
    x_rec, mu, logvar, z = vae(x)

    make_recon_canvas(x, x_rec, out_png, resolution)
    vae.train()


def cmd_train_vae(args):
    ddp = ddp_setup()
    seed_all(args.seed + ddp_rank())

    data_root = os.path.expanduser(args.data_root)
    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    device = f"cuda:{ddp_local_rank()}" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device.startswith("cuda")

    ds = PatchFolder(
        data_root,
        resolution=args.resolution,
        in_channels=args.in_channels,
        flip=True
    )

    if ddp:
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed, drop_last=True)
    else:
        sampler = None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    vae = SimpleVAE(
        in_channels=args.in_channels,
        z_channels=args.z_channels,
        base_ch=args.base_ch
    ).to(device)

    if ddp:
        vae = torch.nn.parallel.DistributedDataParallel(
            vae,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False
        )

    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    ckpt_path = os.path.join(out_root, "vae_last.pt")

    if args.resume and os.path.exists(ckpt_path):
        log0(f"[VAE] resume from {ckpt_path}")
        ckpt = load_ckpt(vae.module if ddp else vae, ckpt_path, map_location="cpu")
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    log0(f"[VAE] data={len(ds)} batch={args.batch_size} ddp={ddp} world={ddp_world()} device={device}")
    log0(f"[VAE] in_channels={args.in_channels} resolution={args.resolution} z_channels={args.z_channels} base_ch={args.base_ch}")
    log0(f"[VAE] lr={args.lr} kl_weight={args.kl_weight} amp={use_amp}")

    t0 = time.time()
    step = start_step
    vae.train()

    while step < args.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // max(1, len(dl)))

        for x in dl:
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                x_rec, mu, logvar, z = vae(x)
                rec_loss = F.l1_loss(x_rec, x)
                kl = vae_kl(mu, logvar)
                loss = rec_loss + args.kl_weight * kl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if is_main() and step % args.log_every == 0:
                dt = (time.time() - t0) / 60.0
                log0(f"[VAE] step={step}/{args.max_steps} loss={loss.item():.6f} rec={rec_loss.item():.6f} kl={kl.item():.6f} time={dt:.1f}m")

            if is_main() and step > 0 and step % args.sample_every == 0:
                out_png = os.path.join(out_root, f"vae_recon_step{step:06d}.png")
                base_vae = vae.module if ddp else vae
                vae_recon_grid(base_vae, dl, device, out_png, resolution=args.resolution, n=args.n_vis)
                log0(f"[VAE] saved recon: {out_png}")

            if is_main() and step > 0 and step % args.save_every == 0:
                base_vae = vae.module if ddp else vae
                save_ckpt(base_vae, ckpt_path, extra={
                    "step": step,
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                })
                log0(f"[VAE] saved: {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break

    if is_main():
        base_vae = vae.module if ddp else vae
        save_ckpt(base_vae, ckpt_path, extra={
            "step": step,
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        })
        log0(f"[VAE] done. saved: {ckpt_path}")

    barrier()
    ddp_cleanup()


@torch.no_grad()
def cmd_recon_vae(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = os.path.expanduser(args.data_root)
    vae_ckpt = os.path.expanduser(args.vae_ckpt)
    out_png = os.path.expanduser(args.out_png)

    seed_all(args.seed)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    ds = PatchFolder(
        data_root,
        resolution=args.resolution,
        in_channels=args.in_channels,
        flip=False
    )

    dl = DataLoader(ds, batch_size=args.n, shuffle=False, num_workers=0)

    vae = SimpleVAE(
        in_channels=args.in_channels,
        z_channels=args.z_channels,
        base_ch=args.base_ch
    ).to(device)

    load_ckpt(vae, vae_ckpt, map_location="cpu")
    vae.eval()

    log0(f"[RECON] data={len(ds)} device={device}")
    log0(f"[RECON] ckpt={vae_ckpt}")

    x = next(iter(dl)).to(device)[:args.n]
    x_rec, mu, logvar, z = vae(x)

    make_recon_canvas(x, x_rec, out_png, args.resolution)
    log0(f"[RECON] saved: {out_png}")


# =========================================================
# Latent Diffusion
# =========================================================
@torch.no_grad()
def encode_latents(vae: SimpleVAE, x: torch.Tensor):
    mu, logvar = vae.encode(x)
    return mu


@torch.no_grad()
def decode_latents(vae: SimpleVAE, z: torch.Tensor):
    return vae.decode(z)


@torch.no_grad()
def sample_ldm(vae, unet, scheduler, device, out_png, out_channels, resolution=512, z_channels=4, n=9, infer_steps=250):
    vae.eval()
    unet.eval()

    latent_hw = resolution // 8
    z = torch.randn(n, z_channels, latent_hw, latent_hw, device=device)

    scheduler.set_timesteps(infer_steps, device=device)

    for t in scheduler.timesteps:
        eps = unet(z, t).sample
        z = scheduler.step(eps, t, z).prev_sample

    x = decode_latents(vae, z)

    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))

    mode = "L" if out_channels == 1 else "RGB"
    canvas = Image.new(mode, (cols * resolution, rows * resolution))

    for i in range(n):
        r, c = divmod(i, cols)
        im = tensor_to_pil_m11(x[i])
        canvas.paste(im, (c * resolution, r * resolution))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)

    unet.train()
    vae.train()


def cmd_train_ldm(args):
    ddp = ddp_setup()
    seed_all(args.seed + ddp_rank())

    data_root = os.path.expanduser(args.data_root)
    out_root = os.path.expanduser(args.out_root)
    vae_ckpt = os.path.expanduser(args.vae_ckpt)

    os.makedirs(out_root, exist_ok=True)

    device = f"cuda:{ddp_local_rank()}" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device.startswith("cuda")

    ds = PatchFolder(
        data_root,
        resolution=args.resolution,
        in_channels=args.in_channels,
        flip=True
    )

    if ddp:
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed, drop_last=True)
    else:
        sampler = None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    vae = SimpleVAE(
        in_channels=args.in_channels,
        z_channels=args.z_channels,
        base_ch=args.base_ch
    ).to(device)

    load_ckpt(vae, vae_ckpt, map_location="cpu")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    latent_hw = args.resolution // 8

    unet = UNet2DModel(
        sample_size=latent_hw,
        in_channels=args.z_channels,
        out_channels=args.z_channels,
        layers_per_block=2,
        block_out_channels=(256, 512, 768, 768),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    ).to(device)

    if ddp:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False
        )

    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ckpt_path = os.path.join(out_root, "ldm_last.pt")
    start_step = 0

    if args.resume and os.path.exists(ckpt_path):
        log0(f"[LDM] resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        (unet.module if ddp else unet).load_state_dict(ckpt["unet"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    accum = max(1, args.grad_accum)

    log0(f"[LDM] data={len(ds)} batch={args.batch_size} ddp={ddp} world={ddp_world()} device={device}")
    log0(f"[LDM] in_channels={args.in_channels} resolution={args.resolution} z_channels={args.z_channels}")
    log0(f"[LDM] lr={args.lr} grad_accum={accum} amp={use_amp} steps={args.max_steps}")
    log0(f"[LDM] using VAE ckpt: {vae_ckpt}")

    t0 = time.time()
    step = start_step
    unet.train()
    torch.backends.cudnn.benchmark = True

    while step < args.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // max(1, len(dl)))

        for x0 in dl:
            x0 = x0.to(device, non_blocking=True)

            with torch.no_grad():
                z0 = encode_latents(vae, x0)

            b = z0.size(0)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=device).long()
            noise = torch.randn_like(z0)
            zt = scheduler.add_noise(z0, noise, t)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = unet(zt, t).sample
                loss = F.mse_loss(pred, noise) / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if is_main() and step % args.log_every == 0:
                dt = (time.time() - t0) / 60.0
                lr = opt.param_groups[0]["lr"]
                log0(f"[LDM] step={step}/{args.max_steps} loss={loss.item() * accum:.6f} lr={lr:.2e} time={dt:.1f}m")

            if is_main() and step > 0 and step % args.sample_every == 0:
                out_png = os.path.join(out_root, f"samples_step{step:06d}.png")
                base_unet = unet.module if ddp else unet
                sample_ldm(
                    vae=vae,
                    unet=base_unet,
                    scheduler=scheduler,
                    device=device,
                    out_png=out_png,
                    out_channels=args.in_channels,
                    resolution=args.resolution,
                    z_channels=args.z_channels,
                    n=args.n_vis,
                    infer_steps=args.infer_steps,
                )
                log0(f"[LDM] saved sample: {out_png}")

            if is_main() and step > 0 and step % args.save_every == 0:
                base_unet = unet.module if ddp else unet
                torch.save({
                    "step": step,
                    "unet": base_unet.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                log0(f"[LDM] saved: {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break

    if is_main():
        base_unet = unet.module if ddp else unet
        torch.save({
            "step": step,
            "unet": base_unet.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        log0(f"[LDM] done. saved: {ckpt_path}")

    barrier()
    ddp_cleanup()


def cmd_sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    vae = SimpleVAE(
        in_channels=args.in_channels,
        z_channels=args.z_channels,
        base_ch=args.base_ch
    ).to(device)
    load_ckpt(vae, os.path.expanduser(args.vae_ckpt), map_location="cpu")
    vae.eval()

    latent_hw = args.resolution // 8

    unet = UNet2DModel(
        sample_size=latent_hw,
        in_channels=args.z_channels,
        out_channels=args.z_channels,
        layers_per_block=2,
        block_out_channels=(256, 512, 768, 768),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    ).to(device)

    ckpt = torch.load(os.path.expanduser(args.ldm_ckpt), map_location="cpu")
    unet.load_state_dict(ckpt["unet"], strict=True)
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    seed_all(args.seed)
    out_png = os.path.join(out_root, f"samples_seed{args.seed}_steps{args.infer_steps}.png")

    sample_ldm(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device,
        out_png=out_png,
        out_channels=args.in_channels,
        resolution=args.resolution,
        z_channels=args.z_channels,
        n=args.n,
        infer_steps=args.infer_steps,
    )

    print(f"[SAMPLE] saved: {out_png}")


# =========================================================
# Argument Parser
# =========================================================
def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------------------------
    # train_vae
    # -------------------------
    p_v = sub.add_parser("train_vae")
    p_v.add_argument("--data_root", required=True)
    p_v.add_argument("--out_root", required=True)
    p_v.add_argument("--resolution", type=int, default=512)
    p_v.add_argument("--in_channels", type=int, default=1)
    p_v.add_argument("--z_channels", type=int, default=4)
    p_v.add_argument("--base_ch", type=int, default=64)
    p_v.add_argument("--batch_size", type=int, default=4)
    p_v.add_argument("--lr", type=float, default=2e-4)
    p_v.add_argument("--kl_weight", type=float, default=1e-3)
    p_v.add_argument("--max_steps", type=int, default=30000)
    p_v.add_argument("--num_workers", type=int, default=4)
    p_v.add_argument("--amp", action="store_true")
    p_v.add_argument("--resume", action="store_true")
    p_v.add_argument("--save_every", type=int, default=2000)
    p_v.add_argument("--sample_every", type=int, default=1000)
    p_v.add_argument("--log_every", type=int, default=50)
    p_v.add_argument("--n_vis", type=int, default=9)
    p_v.add_argument("--seed", type=int, default=0)

    # -------------------------
    # recon_vae
    # -------------------------
    p_r = sub.add_parser("recon_vae")
    p_r.add_argument("--data_root", required=True)
    p_r.add_argument("--vae_ckpt", required=True)
    p_r.add_argument("--out_png", required=True)
    p_r.add_argument("--resolution", type=int, default=512)
    p_r.add_argument("--in_channels", type=int, default=1)
    p_r.add_argument("--z_channels", type=int, default=4)
    p_r.add_argument("--base_ch", type=int, default=64)
    p_r.add_argument("--n", type=int, default=9)
    p_r.add_argument("--seed", type=int, default=0)

    # -------------------------
    # train_ldm
    # -------------------------
    p_l = sub.add_parser("train_ldm")
    p_l.add_argument("--data_root", required=True)
    p_l.add_argument("--out_root", required=True)
    p_l.add_argument("--vae_ckpt", required=True)
    p_l.add_argument("--resolution", type=int, default=512)
    p_l.add_argument("--in_channels", type=int, default=1)
    p_l.add_argument("--z_channels", type=int, default=4)
    p_l.add_argument("--base_ch", type=int, default=64)
    p_l.add_argument("--batch_size", type=int, default=2)
    p_l.add_argument("--lr", type=float, default=1e-4)
    p_l.add_argument("--max_steps", type=int, default=40000)
    p_l.add_argument("--num_workers", type=int, default=4)
    p_l.add_argument("--grad_accum", type=int, default=2)
    p_l.add_argument("--amp", action="store_true")
    p_l.add_argument("--resume", action="store_true")
    p_l.add_argument("--save_every", type=int, default=2000)
    p_l.add_argument("--sample_every", type=int, default=1000)
    p_l.add_argument("--log_every", type=int, default=50)
    p_l.add_argument("--infer_steps", type=int, default=250)
    p_l.add_argument("--n_vis", type=int, default=9)
    p_l.add_argument("--num_train_timesteps", type=int, default=1000)
    p_l.add_argument("--seed", type=int, default=0)

    # -------------------------
    # sample
    # -------------------------
    p_s = sub.add_parser("sample")
    p_s.add_argument("--vae_ckpt", required=True)
    p_s.add_argument("--ldm_ckpt", required=True)
    p_s.add_argument("--out_root", required=True)
    p_s.add_argument("--resolution", type=int, default=512)
    p_s.add_argument("--in_channels", type=int, default=1)
    p_s.add_argument("--z_channels", type=int, default=4)
    p_s.add_argument("--base_ch", type=int, default=64)
    p_s.add_argument("--n", type=int, default=9)
    p_s.add_argument("--infer_steps", type=int, default=250)
    p_s.add_argument("--num_train_timesteps", type=int, default=1000)
    p_s.add_argument("--seed", type=int, default=0)

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "train_vae":
        cmd_train_vae(args)
    elif args.cmd == "recon_vae":
        cmd_recon_vae(args)
    elif args.cmd == "train_ldm":
        cmd_train_ldm(args)
    elif args.cmd == "sample":
        cmd_sample(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()