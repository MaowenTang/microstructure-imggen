#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
COORD_RE = re.compile(r"^(?P<source>.+)_p\d+_x(?P<x>\d+)_y(?P<y>\d+)$")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PatchDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted(
            str(path)
            for path in Path(root).iterdir()
            if path.is_file() and path.suffix.lower() in IMG_EXTS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor / 127.5 - 1.0


def parse_patch_metadata(paths):
    source_counts = Counter()
    experiment_counts = Counter()
    coordinates = defaultdict(list)

    for path in paths:
        match = COORD_RE.match(Path(path).stem)
        if not match:
            continue
        source = match.group("source")
        experiment = source.split(".", 1)[0]
        x = int(match.group("x"))
        y = int(match.group("y"))
        source_counts[source] += 1
        experiment_counts[experiment] += 1
        coordinates[source].append((x, y))

    overlap = {}
    patch_area = 512 * 512
    for source, points in coordinates.items():
        xy = np.asarray(points, dtype=np.int32)
        if len(xy) < 2:
            continue

        dx = np.abs(xy[:, None, 0] - xy[None, :, 0])
        dy = np.abs(xy[:, None, 1] - xy[None, :, 1])
        intersection = np.maximum(0, 512 - dx) * np.maximum(0, 512 - dy)
        union = 2 * patch_area - intersection
        iou = intersection / np.maximum(union, 1)
        np.fill_diagonal(iou, -1.0)
        nearest_iou = iou.max(axis=1)

        distance = np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)
        np.fill_diagonal(distance, np.inf)
        nearest_distance = distance.min(axis=1)

        overlap[source] = {
            "patches": len(points),
            "nearest_iou_mean": float(nearest_iou.mean()),
            "nearest_iou_median": float(np.median(nearest_iou)),
            "fraction_nearest_iou_gt_0_5": float((nearest_iou > 0.5).mean()),
            "nearest_top_left_distance_median_px": float(np.median(nearest_distance)),
        }

    return {
        "source_counts": dict(sorted(source_counts.items())),
        "experiment_counts": dict(sorted(experiment_counts.items())),
        "overlap_by_source": overlap,
    }


def sobel_magnitude(x):
    gray = x.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def reconstruction_metrics(reconstruction, target):
    mse = F.mse_loss(reconstruction, target)
    l1 = F.l1_loss(reconstruction, target)
    psnr = 10.0 * torch.log10(4.0 / torch.clamp(mse, min=1e-12))
    edge = F.l1_loss(sobel_magnitude(reconstruction), sobel_magnitude(target))

    rec_fft = torch.log1p(torch.abs(torch.fft.rfft2(reconstruction.float(), norm="ortho")))
    tgt_fft = torch.log1p(torch.abs(torch.fft.rfft2(target.float(), norm="ortho")))
    fft = F.l1_loss(rec_fft, tgt_fft)
    return {
        "mse": float(mse.item()),
        "l1": float(l1.item()),
        "psnr": float(psnr.item()),
        "edge_mae": float(edge.item()),
        "log_fft_mae": float(fft.item()),
    }


class MetricAverage:
    def __init__(self):
        self.total = Counter()
        self.weight = 0

    def add(self, metrics, weight):
        for key, value in metrics.items():
            self.total[key] += value * weight
        self.weight += weight

    def result(self):
        return {key: value / self.weight for key, value in self.total.items()}


class ChannelStats:
    def __init__(self, channels):
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.sum_sq = torch.zeros(channels, dtype=torch.float64)
        self.count = 0

    def add(self, tensor):
        values = tensor.detach().double().permute(1, 0, 2, 3).reshape(tensor.shape[1], -1).cpu()
        self.sum += values.sum(dim=1)
        self.sum_sq += values.square().sum(dim=1)
        self.count += values.shape[1]

    def result(self):
        mean = self.sum / self.count
        variance = torch.clamp(self.sum_sq / self.count - mean.square(), min=0.0)
        return {
            "mean": mean.tolist(),
            "std": torch.sqrt(variance).tolist(),
            "global_mean": float(mean.mean().item()),
            "global_std_mean": float(torch.sqrt(variance).mean().item()),
        }


def tensor_to_pil(x):
    array = ((x.detach().clamp(-1, 1) + 1.0) * 127.5)
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def save_recon_grid(rows, path, max_samples=6):
    sample_count = min(max_samples, rows[0].shape[0])
    canvas = Image.new("RGB", (sample_count * 512, len(rows) * 512))
    for row_index, tensors in enumerate(rows):
        for sample_index in range(sample_count):
            canvas.paste(tensor_to_pil(tensors[sample_index]), (sample_index * 512, row_index * 512))
    canvas.save(path)


@torch.no_grad()
def diagnose_option_a(module, checkpoint_path, loader, device, max_images, output_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = module.SimpleVAE(z_channels=4).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    deterministic = MetricAverage()
    stochastic = MetricAverage()
    mu_stats = ChannelStats(4)
    posterior_std_stats = ChannelStats(4)
    processed = 0
    grid_saved = False

    for x in loader:
        x = x.to(device, non_blocking=True)
        if processed + x.shape[0] > max_images:
            x = x[: max_images - processed]

        mu, logvar = model.encode(x)
        posterior_std = torch.exp(0.5 * logvar)
        x_det = model.decode(mu)
        x_stochastic = model.decode(mu + posterior_std * torch.randn_like(posterior_std))

        deterministic.add(reconstruction_metrics(x_det, x), x.shape[0])
        stochastic.add(reconstruction_metrics(x_stochastic, x), x.shape[0])
        mu_stats.add(mu)
        posterior_std_stats.add(posterior_std)

        if not grid_saved:
            save_recon_grid(
                [x, x_det, x_stochastic],
                os.path.join(output_dir, "optionA_original_deterministic_stochastic.png"),
            )
            grid_saved = True

        processed += x.shape[0]
        if processed >= max_images:
            break

    return {
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "images": processed,
        "deterministic_reconstruction": deterministic.result(),
        "stochastic_reconstruction": stochastic.result(),
        "mu_stats": mu_stats.result(),
        "posterior_std_stats": posterior_std_stats.result(),
    }


@torch.no_grad()
def diagnose_option_a_plus(module, checkpoint_path, loader, device, max_images, output_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})
    z_low = int(args.get("z_low_channels", 4))
    z_high = int(args.get("z_high_channels", 4))
    kernel = int(args.get("decomp_kernel", 9))
    sigma = float(args.get("decomp_sigma", 1.0))

    model = module.DualBranchVAEPlus(z_low_channels=z_low, z_high_channels=z_high).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    deterministic = MetricAverage()
    stochastic = MetricAverage()
    low_mu_stats = ChannelStats(z_low)
    high_mu_stats = ChannelStats(z_high)
    low_posterior_stats = ChannelStats(z_low)
    high_posterior_stats = ChannelStats(z_high)
    processed = 0
    grid_saved = False

    for x in loader:
        x = x.to(device, non_blocking=True)
        if processed + x.shape[0] > max_images:
            x = x[: max_images - processed]

        x_low, x_high = module.decompose_freq(x, kernel_size=kernel, sigma=sigma)
        mu_low, logvar_low = model.encode_low(x_low)
        mu_high, logvar_high = model.encode_high(x_high)
        std_low = torch.exp(0.5 * logvar_low)
        std_high = torch.exp(0.5 * logvar_high)

        x_det = model.decode_low(mu_low) + model.decode_high(mu_high)
        x_stochastic = (
            model.decode_low(mu_low + std_low * torch.randn_like(std_low))
            + model.decode_high(mu_high + std_high * torch.randn_like(std_high))
        )

        deterministic.add(reconstruction_metrics(x_det, x), x.shape[0])
        stochastic.add(reconstruction_metrics(x_stochastic, x), x.shape[0])
        low_mu_stats.add(mu_low)
        high_mu_stats.add(mu_high)
        low_posterior_stats.add(std_low)
        high_posterior_stats.add(std_high)

        if not grid_saved:
            save_recon_grid(
                [x, x_det, x_stochastic],
                os.path.join(output_dir, "optionAplus_original_deterministic_stochastic.png"),
            )
            grid_saved = True

        processed += x.shape[0]
        if processed >= max_images:
            break

    return {
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "images": processed,
        "decomposition": {"kernel": kernel, "sigma": sigma},
        "deterministic_reconstruction": deterministic.result(),
        "stochastic_reconstruction": stochastic.result(),
        "low_mu_stats": low_mu_stats.result(),
        "high_mu_stats": high_mu_stats.result(),
        "low_posterior_std_stats": low_posterior_stats.result(),
        "high_posterior_std_stats": high_posterior_stats.result(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--option_a_ckpt", required=True)
    parser.add_argument("--option_a_plus_ckpt", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("E0 latent diagnostics requires a CUDA GPU.")

    dataset = PatchDataset(args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    option_a = load_module(
        os.path.join(args.project_root, "src", "ldm_microstructure_optionA.py"),
        "option_a",
    )
    option_a_plus = load_module(
        os.path.join(args.project_root, "src", "ldm_microstructure_optionA_plus.py"),
        "option_a_plus",
    )

    report = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "dataset_images": len(dataset),
        "patch_metadata": parse_patch_metadata(dataset.paths),
        "option_a": diagnose_option_a(
            option_a,
            args.option_a_ckpt,
            loader,
            device,
            args.max_images,
            args.out_root,
        ),
        "option_a_plus": diagnose_option_a_plus(
            option_a_plus,
            args.option_a_plus_ckpt,
            loader,
            device,
            args.max_images,
            args.out_root,
        ),
    }

    output_path = os.path.join(args.out_root, "e0_report.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2), flush=True)
    print(f"E0_REPORT={output_path}", flush=True)
    print("E0_DIAGNOSTICS=PASS", flush=True)


if __name__ == "__main__":
    main()
