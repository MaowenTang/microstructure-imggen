#!/usr/bin/env python3
"""Sample the fixed SC26 branch morphology comparison set.

This sampler generates six branch-aligned sample sets with common seeds:

- E2 base unconditional (`ldm_best.pt`)
- E3 global conditional targeting process 9 (`conditional_best.pt`)
- E4 spatial conditional targeting a fixed held-out process-9 validation patch
  (`spatial_best.pt`)
- E8 Sobel process-9 expert (`e7_last.pt`)
- E12 gray process-9 expert (`e7_last.pt`)
- E13 RGB process-9 expert (`e7_last.pt`)

For each branch and each seed, the sampler writes four 512x512 PNGs plus a
manifest in both CSV and JSON forms. Branch defaults follow the documented
experiment paths and sampling settings, but every path and branch-specific
sampling hyperparameter is available as an explicit CLI flag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch


SCHEMA_VERSION = "sc26-branch-morphology-sampling-v1"
BRANCH_ORDER = (
    "e2",
    "e3",
    "e4",
    "e8",
    "e12",
    "e13",
)
MANIFEST_FIELDS = (
    "branch",
    "seed",
    "sample_index",
    "count_per_seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_kind",
    "checkpoint_state_key",
    "cae_checkpoint_path",
    "condition_reference",
    "output_path",
    "output_sha256",
    "width",
    "height",
    "settings_json",
)


def import_project_modules():
    global e1
    global e2
    global e3
    global e4
    global e7

    import microstructure_e1_cae as e1
    import microstructure_e2_ldm as e2
    import microstructure_e3_conditional_ldm as e3
    import microstructure_e4_spatial_condition as e4
    import microstructure_e7_patch_sobel_ldm as e7


@dataclass
class PreparedBranch:
    branch: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_kind: str
    checkpoint_state_key: str
    cae_checkpoint_path: Path
    count_per_seed: int
    settings: dict[str, Any]
    condition_reference: dict[str, Any]
    sample_seed: Callable[[int], torch.Tensor]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_existing(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required file not found: {resolved}")
    return resolved


def resolve_directory(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Required directory not found: {resolved}")
    return resolved


def parse_csv_list(value: str, cast: Callable[[str], Any]) -> tuple[Any, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return tuple(cast(item) for item in items)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = ((image.detach().clamp(-1, 1) + 1.0) * 127.5)
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def load_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.BICUBIC)
        array = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
        return array.contiguous() / 127.5 - 1.0


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    return checkpoint


def save_branch_seed_images(
    output_root: Path,
    branch: str,
    seed: int,
    images: torch.Tensor,
) -> list[dict[str, Any]]:
    if images.ndim != 4 or images.shape[1:] != (3, 512, 512):
        raise ValueError(f"Unexpected image tensor shape for {branch}: {tuple(images.shape)}")
    if images.shape[0] < 1:
        raise ValueError(f"No images returned for {branch} seed={seed}.")
    rows = []
    seed_root = output_root / branch / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    for sample_index, image in enumerate(images):
        output_path = seed_root / f"sample_{sample_index:02d}.png"
        temporary = output_path.with_name(output_path.name + ".tmp")
        tensor_to_pil(image).save(temporary, format="PNG")
        os.replace(temporary, output_path)
        rows.append(
            {
                "sample_index": sample_index,
                "output_path": output_path,
                "output_sha256": sha256_file(output_path),
            }
        )
    return rows


def branch_condition_path(output_root: Path, branch: str, suffix: str) -> Path:
    return output_root / "conditions" / f"{branch}_{suffix}"


def build_root_relative_path(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def default_path_args(root: Path) -> dict[str, Path]:
    return {
        "data_root": build_root_relative_path(root, "data/500um_p512_n500"),
        "e2_cae_checkpoint": build_root_relative_path(root, "runs/e1_cae_pilot_z8/cae_last.pt"),
        "e2_checkpoint": build_root_relative_path(root, "runs/e2_ldm_pilot_z8/ldm_best.pt"),
        "e3_cae_checkpoint": build_root_relative_path(root, "runs/e1_cae_pilot_z8/cae_last.pt"),
        "e3_checkpoint": build_root_relative_path(root, "runs/e3_conditional_pilot/conditional_best.pt"),
        "e4_cae_checkpoint": build_root_relative_path(root, "runs/e1_cae_pilot_z8/cae_last.pt"),
        "e4_checkpoint": build_root_relative_path(root, "runs/e4_spatial_pilot/spatial_best.pt"),
        "e8_cae_checkpoint": build_root_relative_path(root, "runs/e1_cae_pilot_z8/cae_last.pt"),
        "e8_checkpoint": build_root_relative_path(root, "runs/e8_p9_sobel70_ldm/e7_last.pt"),
        "e12_cae_checkpoint": build_root_relative_path(root, "runs/e12_gray_cae_z8/cae_last.pt"),
        "e12_checkpoint": build_root_relative_path(root, "runs/e12_gray_p9_sobel50_w2_ldm/e7_last.pt"),
        "e13_cae_checkpoint": build_root_relative_path(root, "runs/e1_cae_pilot_z8/cae_last.pt"),
        "e13_checkpoint": build_root_relative_path(root, "runs/e13_rgb_p9_sobel50_w2_ldm/e7_last.pt"),
    }


def process_features_for(process: str, device: torch.device) -> torch.Tensor:
    return e3.process_features([f"{process}.placeholder.png"], device)


def prepare_e2(args: argparse.Namespace, device: torch.device, output_root: Path) -> PreparedBranch:
    cae_path = resolve_existing(args.e2_cae_checkpoint)
    checkpoint_path = resolve_existing(args.e2_checkpoint)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    cae, latent_channels = e2.load_cae(str(cae_path), device)
    model = e2.build_unet(latent_channels).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    latent_mean = checkpoint["latent_mean"].float().to(device)
    latent_std = checkpoint["latent_std"].float().to(device)
    checkpoint_sha = sha256_file(checkpoint_path)
    settings = {
        "branch": "e2",
        "count": args.count_per_seed,
        "infer_steps": args.e2_infer_steps,
        "seed_schedule": list(args.seeds),
        "source_branch": "E2 standardized latent diffusion",
    }
    condition_reference = {"type": "unconditional", "detail": "No conditioning input."}

    def sample_seed(seed: int) -> torch.Tensor:
        return e2.generate(
            cae,
            model,
            e2.build_noise_scheduler(),
            latent_mean,
            latent_std,
            device,
            args.count_per_seed,
            args.e2_infer_steps,
            seed,
        )

    return PreparedBranch(
        branch="e2",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_kind="ldm_best",
        checkpoint_state_key="ema",
        cae_checkpoint_path=cae_path,
        count_per_seed=args.count_per_seed,
        settings=settings,
        condition_reference=condition_reference,
        sample_seed=sample_seed,
    )


def prepare_e3(args: argparse.Namespace, device: torch.device, output_root: Path) -> PreparedBranch:
    cae_path = resolve_existing(args.e3_cae_checkpoint)
    checkpoint_path = resolve_existing(args.e3_checkpoint)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    cae, latent_channels = e2.load_cae(str(cae_path), device)
    model = e3.ConditionedUNet(
        latent_channels,
        condition_dim=len(e3.DESCRIPTOR_NAMES) + len(e3.PROCESS_FEATURE_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    latent_mean = checkpoint["latent_mean"].float().to(device)
    latent_std = checkpoint["latent_std"].float().to(device)
    condition_mean = checkpoint["condition_mean"].float().to(device)
    condition_std = checkpoint["condition_std"].float().to(device)
    descriptor_centroids = checkpoint["descriptor_centroids"]
    descriptor_target = descriptor_centroids[args.e3_process].float().to(device)
    process_target = process_features_for(args.e3_process, device)[0]
    raw_condition = torch.cat((descriptor_target, process_target), dim=0)
    normalized_condition = e3.normalize_condition(
        raw_condition[None],
        condition_mean,
        condition_std,
    ).repeat(args.count_per_seed, 1)
    condition_path = branch_condition_path(output_root, "e3", "process9_condition.json")
    condition_reference = {
        "type": "process_descriptor_condition",
        "process": args.e3_process,
        "condition_json": str(condition_path),
    }
    write_json(
        condition_path,
        {
            "branch": "e3",
            "process": args.e3_process,
            "descriptor_names": list(e3.DESCRIPTOR_NAMES),
            "process_feature_names": list(e3.PROCESS_FEATURE_NAMES),
            "descriptor_centroid": descriptor_target.detach().cpu().tolist(),
            "process_features": process_target.detach().cpu().tolist(),
            "normalized_condition": normalized_condition[0].detach().cpu().tolist(),
            "checkpoint_path": str(checkpoint_path),
        },
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    settings = {
        "branch": "e3",
        "count": args.count_per_seed,
        "infer_steps": args.e3_infer_steps,
        "guidance_scale": args.e3_guidance_scale,
        "process": args.e3_process,
        "seed_schedule": list(args.seeds),
        "source_branch": "E3 global conditional baseline",
    }

    def sample_seed(seed: int) -> torch.Tensor:
        return e3.generate_conditioned(
            cae,
            model,
            e2.build_noise_scheduler(),
            latent_mean,
            latent_std,
            normalized_condition,
            device,
            args.e3_infer_steps,
            args.e3_guidance_scale,
            seed,
        )

    return PreparedBranch(
        branch="e3",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_kind="conditional_best",
        checkpoint_state_key="ema",
        cae_checkpoint_path=cae_path,
        count_per_seed=args.count_per_seed,
        settings=settings,
        condition_reference=condition_reference,
        sample_seed=sample_seed,
    )


def prepare_e4(args: argparse.Namespace, device: torch.device, output_root: Path) -> PreparedBranch:
    data_root = resolve_directory(args.data_root)
    cae_path = resolve_existing(args.e4_cae_checkpoint)
    checkpoint_path = resolve_existing(args.e4_checkpoint)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    cae, latent_channels = e2.load_cae(str(cae_path), device)
    model = e4.SpatialConditionedUNet(
        latent_channels,
        len(e4.STRUCTURE_NAMES),
        len(e4.PROCESS_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    latent_mean = checkpoint["latent_mean"].float().to(device)
    latent_std = checkpoint["latent_std"].float().to(device)
    statistics = checkpoint["statistics"]
    structure_mean = statistics["structure_mean"].float().to(device)
    structure_std = statistics["structure_std"].float().to(device)
    process_mean = statistics["process_mean"].float().to(device)
    process_std = statistics["process_std"].float().to(device)

    paths = e1.list_images(str(data_root))
    _, validation_paths, _, _ = e1.build_spatial_split(
        paths,
        train_max=args.e4_train_max_x,
        val_min=args.e4_val_min_x,
    )
    oracle_paths = e4.select_oracle_paths(validation_paths)
    target_path = Path(next(path for path in oracle_paths if e4.process_id(path) == args.e4_process))
    target_tensor = load_rgb_tensor(target_path).unsqueeze(0).to(device)
    raw_structure = e4.structure_map(target_tensor)
    normalized_structure = e4.normalize_channels(
        raw_structure,
        structure_mean,
        structure_std,
    ).repeat(args.count_per_seed, 1, 1, 1)
    process_rows = e4.normalize_rows(
        e4.process_features([str(target_path)], device),
        process_mean,
        process_std,
    ).repeat(args.count_per_seed, 1)
    original_path = branch_condition_path(output_root, "e4", "p9_validation.png")
    ridge_path = branch_condition_path(output_root, "e4", "p9_ridge_map.png")
    original_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(target_tensor[0].cpu()).save(original_path)
    e4.map_to_pil(raw_structure[0, 1]).save(ridge_path)
    condition_path = branch_condition_path(output_root, "e4", "p9_condition.json")
    condition_reference = {
        "type": "held_out_spatial_condition",
        "process": args.e4_process,
        "reuse_mode": "single held-out P9 validation image and ridge map repeated 4 draws per seed",
        "validation_image": str(original_path),
        "ridge_map": str(ridge_path),
        "source_patch": str(target_path),
    }
    write_json(
        condition_path,
        {
            "branch": "e4",
            "process": args.e4_process,
            "source_patch": str(target_path),
            "structure_names": list(e4.STRUCTURE_NAMES),
            "process_names": list(e4.PROCESS_NAMES),
            "train_max_x": args.e4_train_max_x,
            "val_min_x": args.e4_val_min_x,
            "normalized_process": process_rows[0].detach().cpu().tolist(),
            "structure_image": str(original_path),
            "ridge_map_image": str(ridge_path),
        },
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    settings = {
        "branch": "e4",
        "count": args.count_per_seed,
        "infer_steps": args.e4_infer_steps,
        "guidance_scale": args.e4_guidance_scale,
        "process": args.e4_process,
        "train_max_x": args.e4_train_max_x,
        "val_min_x": args.e4_val_min_x,
        "seed_schedule": list(args.seeds),
        "source_branch": "E4 spatial ripple-map conditioning",
    }

    def sample_seed(seed: int) -> torch.Tensor:
        return e4.generate(
            cae,
            model,
            e2.build_noise_scheduler(),
            latent_mean,
            latent_std,
            normalized_structure,
            process_rows,
            device,
            args.e4_infer_steps,
            args.e4_guidance_scale,
            seed,
        )

    return PreparedBranch(
        branch="e4",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_kind="spatial_best",
        checkpoint_state_key="ema",
        cae_checkpoint_path=cae_path,
        count_per_seed=args.count_per_seed,
        settings=settings,
        condition_reference=condition_reference,
        sample_seed=sample_seed,
    )


def prepare_expert_branch(
    branch: str,
    cae_checkpoint: str | Path,
    checkpoint_path_value: str | Path,
    checkpoint_kind: str,
    process: str,
    infer_steps: int,
    guidance_scale: float,
    count_per_seed: int,
    device: torch.device,
) -> PreparedBranch:
    cae_path = resolve_existing(cae_checkpoint)
    checkpoint_path = resolve_existing(checkpoint_path_value)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    cae, latent_channels = e2.load_cae(str(cae_path), device)
    model = e3.ConditionedUNet(
        latent_channels,
        condition_dim=len(e3.PROCESS_FEATURE_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    latent_mean = checkpoint["latent_mean"].float().to(device)
    latent_std = checkpoint["latent_std"].float().to(device)
    process_mean = checkpoint["process_mean"].float().to(device)
    process_std = checkpoint["process_std"].float().to(device)
    checkpoint_sha = sha256_file(checkpoint_path)
    condition_reference = {
        "type": "process_only_expert",
        "process": process,
        "mode": "repeat same process condition four times per seed",
    }
    settings = {
        "branch": branch,
        "count": count_per_seed,
        "infer_steps": infer_steps,
        "guidance_scale": guidance_scale,
        "processes": [process],
        "samples_per_process": count_per_seed,
        "source_branch": branch.upper(),
    }

    def sample_seed(seed: int) -> torch.Tensor:
        images, labels = e7.generate_selected_process_grid(
            cae,
            model,
            e2.build_noise_scheduler(),
            latent_mean,
            latent_std,
            process_mean,
            process_std,
            device,
            count_per_seed,
            infer_steps,
            guidance_scale,
            seed,
            (process,),
        )
        expected = [process] * count_per_seed
        if list(labels) != expected:
            raise ValueError(f"Unexpected labels for {branch}: {labels}")
        return images

    return PreparedBranch(
        branch=branch,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_kind=checkpoint_kind,
        checkpoint_state_key="ema",
        cae_checkpoint_path=cae_path,
        count_per_seed=count_per_seed,
        settings=settings,
        condition_reference=condition_reference,
        sample_seed=sample_seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the fixed SC26 six-branch morphology sample set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default="/scratch/user/u.mt227311/microstructure-imggen",
        help="Default scratch root used to derive branch checkpoints and data paths.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Dedicated output directory for images, references, and manifests.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Patch dataset root. Defaults to <root>/data/500um_p512_n500.",
    )
    parser.add_argument(
        "--branches",
        default="e2,e3,e4,e8,e12,e13",
        help="Comma-separated subset of branches to sample.",
    )
    parser.add_argument(
        "--seeds",
        default="9090,9091,9092,9093,9094,9095,9096,9097",
        help="Comma-separated fixed seeds shared by all branches.",
    )
    parser.add_argument(
        "--count-per-seed",
        type=int,
        default=4,
        help="Number of images to emit per branch per seed.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device for sampling.",
    )

    parser.add_argument("--e2-cae-checkpoint", default=None)
    parser.add_argument("--e2-checkpoint", default=None)
    parser.add_argument("--e2-infer-steps", type=int, default=250)

    parser.add_argument("--e3-cae-checkpoint", default=None)
    parser.add_argument("--e3-checkpoint", default=None)
    parser.add_argument("--e3-process", default="9")
    parser.add_argument("--e3-infer-steps", type=int, default=250)
    parser.add_argument("--e3-guidance-scale", type=float, default=2.0)

    parser.add_argument("--e4-cae-checkpoint", default=None)
    parser.add_argument("--e4-checkpoint", default=None)
    parser.add_argument("--e4-process", default="9")
    parser.add_argument("--e4-infer-steps", type=int, default=250)
    parser.add_argument("--e4-guidance-scale", type=float, default=2.0)
    parser.add_argument("--e4-train-max-x", type=float, default=0.60)
    parser.add_argument("--e4-val-min-x", type=float, default=0.80)

    parser.add_argument("--e8-cae-checkpoint", default=None)
    parser.add_argument("--e8-checkpoint", default=None)
    parser.add_argument("--e8-process", default="9")
    parser.add_argument("--e8-infer-steps", type=int, default=250)
    parser.add_argument("--e8-guidance-scale", type=float, default=1.0)

    parser.add_argument("--e12-cae-checkpoint", default=None)
    parser.add_argument("--e12-checkpoint", default=None)
    parser.add_argument("--e12-process", default="9")
    parser.add_argument("--e12-infer-steps", type=int, default=250)
    parser.add_argument("--e12-guidance-scale", type=float, default=1.0)

    parser.add_argument("--e13-cae-checkpoint", default=None)
    parser.add_argument("--e13-checkpoint", default=None)
    parser.add_argument("--e13-process", default="9")
    parser.add_argument("--e13-infer-steps", type=int, default=250)
    parser.add_argument("--e13-guidance-scale", type=float, default=1.0)
    return parser


def apply_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = Path(args.root).expanduser()
    defaults = default_path_args(root)
    for name, default_value in defaults.items():
        if getattr(args, name, None) in (None, ""):
            setattr(args, name, str(default_value))
    if args.data_root in (None, ""):
        args.data_root = str(defaults["data_root"])
    return args


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.seeds = parse_csv_list(args.seeds, int)
    selected = parse_csv_list(args.branches, str)
    unknown = [branch for branch in selected if branch not in BRANCH_ORDER]
    if unknown:
        raise ValueError(f"Unknown branches: {unknown}")
    args.branches = tuple(selected)
    if args.count_per_seed != 4:
        raise ValueError("This protocol expects exactly 4 images per seed for comparability.")
    if not args.device.startswith("cuda"):
        raise ValueError("This script is intended for CUDA sampling; use a CUDA device.")
    return args


def prepare_branches(
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> list[PreparedBranch]:
    prepared = []
    if "e2" in args.branches:
        prepared.append(prepare_e2(args, device, output_root))
    if "e3" in args.branches:
        prepared.append(prepare_e3(args, device, output_root))
    if "e4" in args.branches:
        prepared.append(prepare_e4(args, device, output_root))
    if "e8" in args.branches:
        prepared.append(
            prepare_expert_branch(
                branch="e8",
                cae_checkpoint=args.e8_cae_checkpoint,
                checkpoint_path_value=args.e8_checkpoint,
                checkpoint_kind="e7_last",
                process=args.e8_process,
                infer_steps=args.e8_infer_steps,
                guidance_scale=args.e8_guidance_scale,
                count_per_seed=args.count_per_seed,
                device=device,
            )
        )
    if "e12" in args.branches:
        prepared.append(
            prepare_expert_branch(
                branch="e12",
                cae_checkpoint=args.e12_cae_checkpoint,
                checkpoint_path_value=args.e12_checkpoint,
                checkpoint_kind="e7_last",
                process=args.e12_process,
                infer_steps=args.e12_infer_steps,
                guidance_scale=args.e12_guidance_scale,
                count_per_seed=args.count_per_seed,
                device=device,
            )
        )
    if "e13" in args.branches:
        prepared.append(
            prepare_expert_branch(
                branch="e13",
                cae_checkpoint=args.e13_cae_checkpoint,
                checkpoint_path_value=args.e13_checkpoint,
                checkpoint_kind="e7_last",
                process=args.e13_process,
                infer_steps=args.e13_infer_steps,
                guidance_scale=args.e13_guidance_scale,
                count_per_seed=args.count_per_seed,
                device=device,
            )
        )
    return prepared


def main() -> None:
    args = validate_args(apply_default_paths(build_parser().parse_args()))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SC26 branch morphology sampling.")
    import_project_modules()

    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prepared = prepare_branches(args, device, output_root)

    manifest_rows: list[dict[str, Any]] = []
    branch_summaries = []
    for branch in prepared:
        branch_root = output_root / branch.branch
        branch_root.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            images = branch.sample_seed(seed).cpu()
            saved_rows = save_branch_seed_images(output_root, branch.branch, seed, images)
            for saved in saved_rows:
                manifest_rows.append(
                    {
                        "branch": branch.branch,
                        "seed": seed,
                        "sample_index": saved["sample_index"],
                        "count_per_seed": branch.count_per_seed,
                        "checkpoint_path": str(branch.checkpoint_path),
                        "checkpoint_sha256": branch.checkpoint_sha256,
                        "checkpoint_kind": branch.checkpoint_kind,
                        "checkpoint_state_key": branch.checkpoint_state_key,
                        "cae_checkpoint_path": str(branch.cae_checkpoint_path),
                        "condition_reference": json.dumps(
                            branch.condition_reference,
                            sort_keys=True,
                        ),
                        "output_path": str(saved["output_path"]),
                        "output_sha256": saved["output_sha256"],
                        "width": 512,
                        "height": 512,
                        "settings_json": json.dumps(branch.settings, sort_keys=True),
                    }
                )
        branch_summaries.append(
            {
                "branch": branch.branch,
                "checkpoint_path": str(branch.checkpoint_path),
                "checkpoint_sha256": branch.checkpoint_sha256,
                "checkpoint_kind": branch.checkpoint_kind,
                "cae_checkpoint_path": str(branch.cae_checkpoint_path),
                "condition_reference": branch.condition_reference,
                "settings": branch.settings,
                "n_files": len(args.seeds) * branch.count_per_seed,
            }
        )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "device": args.device,
        "seeds": list(args.seeds),
        "count_per_seed": args.count_per_seed,
        "branches": branch_summaries,
        "files": manifest_rows,
    }
    write_csv(output_root / "manifest.csv", manifest_rows)
    write_json(output_root / "manifest.json", metadata)
    print(f"Wrote {len(manifest_rows)} files across {len(prepared)} branches to {output_root}", flush=True)


if __name__ == "__main__":
    main()
