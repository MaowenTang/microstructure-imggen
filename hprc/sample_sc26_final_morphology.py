#!/usr/bin/env python3
"""Generate the fixed SC26 morphology sample set from a checkpoint inventory.

The input CSV contains one logical experiment job per row.  Required columns are
``condition``, ``wave_id``, ``job_name``, and ``checkpoint_path``.  An optional
``checkpoint_id`` may be supplied; otherwise it is derived from the wave and job
names.  Repeated references to the same checkpoint are sampled only once but are
retained as separate logical jobs in the output manifest.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

SCHEMA_VERSION = "sc26-final-morphology-sampling-v1"
PROCESS_ID = "9"
SEEDS = tuple(range(9090, 9098))
SAMPLES_PER_SEED = 4
INFER_STEPS = 250
GUIDANCE_SCALE = 1.0
DDIM_ETA = 0.0
EXPECTED_STEP = 3000
EXPECTED_IMAGE_SIZE = (512, 512)
MODEL_STATE_KEY = "ema"
MANIFEST_FIELDS = (
    "condition",
    "wave_id",
    "job_name",
    "checkpoint_id",
    "canonical_checkpoint_id",
    "checkpoint_path",
    "checkpoint_realpath",
    "checkpoint_sha256",
    "checkpoint_step",
    "model_state_key",
    "process_id",
    "seed",
    "sample_index",
    "tile_path",
    "tile_sha256",
    "width",
    "height",
    "status",
)
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def import_project_components():
    """Delay GPU-stack imports so ``--help`` also works on login machines."""
    global ConditionedUNet
    global PROCESS_FEATURE_NAMES
    global build_noise_scheduler
    global generate_selected_process_grid
    global load_cae
    global tensor_to_pil

    from microstructure_e2_ldm import build_noise_scheduler, load_cae
    from microstructure_e3_conditional_ldm import (
        ConditionedUNet,
        PROCESS_FEATURE_NAMES,
        tensor_to_pil,
    )
    from microstructure_e7_patch_sobel_ldm import generate_selected_process_grid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_manifest(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def safe_checkpoint_id(value):
    if not SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"Unsafe checkpoint_id={value!r}; use only letters, digits, '.', '_', and '-'."
        )
    return value


def derived_checkpoint_id(wave_id, job_name):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{wave_id}__{job_name}").strip("._-")
    if not value:
        raise ValueError(f"Cannot derive checkpoint_id from wave_id={wave_id!r}, job_name={job_name!r}")
    return value


def read_checkpoint_csv(path):
    path = path.expanduser().resolve(strict=True)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"condition", "wave_id", "job_name", "checkpoint_path"}
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Checkpoint CSV is missing required columns: {missing}")
        rows = []
        for line_number, source in enumerate(reader, start=2):
            values = {key: (value or "").strip() for key, value in source.items()}
            empty = [key for key in required if not values[key]]
            if empty:
                raise ValueError(f"Checkpoint CSV line {line_number} has empty fields: {sorted(empty)}")
            checkpoint_path = Path(values["checkpoint_path"]).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = path.parent / checkpoint_path
            checkpoint_realpath = checkpoint_path.resolve(strict=True)
            if not checkpoint_realpath.is_file():
                raise ValueError(f"Checkpoint is not a file: {checkpoint_realpath}")
            checkpoint_id = values.get("checkpoint_id") or derived_checkpoint_id(
                values["wave_id"], values["job_name"]
            )
            rows.append(
                {
                    "condition": values["condition"],
                    "wave_id": values["wave_id"],
                    "job_name": values["job_name"],
                    "checkpoint_id": safe_checkpoint_id(checkpoint_id),
                    "checkpoint_path": values["checkpoint_path"],
                    "checkpoint_realpath": str(checkpoint_realpath),
                }
            )
    if not rows:
        raise ValueError("Checkpoint CSV has no data rows.")
    seen_ids = set()
    seen_jobs = set()
    for row in rows:
        if row["checkpoint_id"] in seen_ids:
            raise ValueError(f"Duplicate checkpoint_id: {row['checkpoint_id']}")
        seen_ids.add(row["checkpoint_id"])
        logical_job = (row["wave_id"], row["job_name"])
        if logical_job in seen_jobs:
            raise ValueError(f"Duplicate logical job: wave_id={logical_job[0]}, job_name={logical_job[1]}")
        seen_jobs.add(logical_job)
    return path, rows


def ensure_output_isolated(output_root, logical_rows):
    output_root = output_root.expanduser().resolve()
    for row in logical_rows:
        job_directory = Path(row["checkpoint_realpath"]).parent
        if output_root == job_directory or job_directory in output_root.parents:
            raise ValueError(
                f"Output root {output_root} is inside original job directory {job_directory}."
            )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def cached_sha256(path, old_records):
    stat = path.stat()
    cached = old_records.get(str(path))
    if (
        cached
        and cached.get("size_bytes") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and cached.get("sha256")
    ):
        return cached["sha256"], stat, True
    return sha256_file(path), stat, False


def load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_checkpoint(checkpoint, path, latent_channels):
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    required = {
        "step",
        MODEL_STATE_KEY,
        "latent_mean",
        "latent_std",
        "process_mean",
        "process_std",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint {path} is missing keys: {missing}")
    step = int(checkpoint["step"])
    if step != EXPECTED_STEP:
        raise ValueError(f"Checkpoint {path} has step={step}; expected {EXPECTED_STEP}.")
    state = checkpoint[MODEL_STATE_KEY]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint {path} has an empty or invalid EMA state.")

    tensors = {}
    for key in ("latent_mean", "latent_std", "process_mean", "process_std"):
        value = checkpoint[key]
        if not torch.is_tensor(value):
            raise ValueError(f"Checkpoint {path} key {key} is not a tensor.")
        value = value.detach().float().cpu()
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint {path} key {key} contains non-finite values.")
        tensors[key] = value
    if tensors["latent_mean"].shape != tensors["latent_std"].shape:
        raise ValueError(f"Checkpoint {path} latent mean/std shapes differ.")
    if tensors["latent_mean"].numel() != latent_channels:
        raise ValueError(
            f"Checkpoint {path} has {tensors['latent_mean'].numel()} latent channels; "
            f"CAE has {latent_channels}."
        )
    if not torch.all(tensors["latent_std"] > 0):
        raise ValueError(f"Checkpoint {path} latent_std must be positive.")
    if tensors["process_mean"].shape != tensors["process_std"].shape:
        raise ValueError(f"Checkpoint {path} process mean/std shapes differ.")
    if tensors["process_mean"].numel() != len(PROCESS_FEATURE_NAMES):
        raise ValueError(
            f"Checkpoint {path} has {tensors['process_mean'].numel()} process features; "
            f"expected {len(PROCESS_FEATURE_NAMES)}."
        )
    if not torch.all(tensors["process_std"] > 0):
        raise ValueError(f"Checkpoint {path} process_std must be positive.")
    feature_names = checkpoint.get("process_feature_names")
    if feature_names is not None and tuple(feature_names) != tuple(PROCESS_FEATURE_NAMES):
        raise ValueError(f"Checkpoint {path} uses unexpected process feature names.")
    parameters = checkpoint.get("process_parameters")
    if parameters is not None and PROCESS_ID not in parameters:
        raise ValueError(f"Checkpoint {path} has no parameters for process {PROCESS_ID}.")
    return step, tensors


def valid_png(path):
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == EXPECTED_IMAGE_SIZE and image.mode == "RGB"
    except (OSError, ValueError):
        return False


def tile_relative_path(canonical_id, seed, sample_index):
    return Path("tiles") / canonical_id / f"seed_{seed}" / f"sample_{sample_index:02d}.png"


def seed_complete(output_root, canonical_id, seed):
    return all(
        valid_png(output_root / tile_relative_path(canonical_id, seed, sample_index))
        for sample_index in range(SAMPLES_PER_SEED)
    )


def save_seed_tiles(images, output_root, canonical_id, seed):
    if tuple(images.shape) != (SAMPLES_PER_SEED, 3, 512, 512):
        raise ValueError(f"Unexpected generated tensor shape: {tuple(images.shape)}")
    for sample_index, tensor in enumerate(images):
        path = output_root / tile_relative_path(canonical_id, seed, sample_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        tensor_to_pil(tensor).save(temporary, format="PNG")
        os.replace(temporary, path)
        if not valid_png(path):
            raise RuntimeError(f"Generated tile failed PNG validation: {path}")


def build_manifest_rows(output_root, logical_rows):
    rows = []
    tile_hashes = {}
    for logical in logical_rows:
        canonical_id = logical["canonical_checkpoint_id"]
        for seed in SEEDS:
            for sample_index in range(SAMPLES_PER_SEED):
                relative = tile_relative_path(canonical_id, seed, sample_index)
                absolute = output_root / relative
                if not valid_png(absolute):
                    continue
                if relative.as_posix() not in tile_hashes:
                    tile_hashes[relative.as_posix()] = sha256_file(absolute)
                rows.append(
                    {
                        "condition": logical["condition"],
                        "wave_id": logical["wave_id"],
                        "job_name": logical["job_name"],
                        "checkpoint_id": logical["checkpoint_id"],
                        "canonical_checkpoint_id": canonical_id,
                        "checkpoint_path": logical["checkpoint_path"],
                        "checkpoint_realpath": logical["checkpoint_realpath"],
                        "checkpoint_sha256": logical["checkpoint_sha256"],
                        "checkpoint_step": EXPECTED_STEP,
                        "model_state_key": MODEL_STATE_KEY,
                        "process_id": PROCESS_ID,
                        "seed": seed,
                        "sample_index": sample_index,
                        "tile_path": relative.as_posix(),
                        "tile_sha256": tile_hashes[relative.as_posix()],
                        "width": EXPECTED_IMAGE_SIZE[0],
                        "height": EXPECTED_IMAGE_SIZE[1],
                        "status": "completed",
                    }
                )
    return rows


def protocol_metadata():
    return {
        "process_id": PROCESS_ID,
        "seeds": list(SEEDS),
        "samples_per_seed": SAMPLES_PER_SEED,
        "samples_per_logical_job": len(SEEDS) * SAMPLES_PER_SEED,
        "sampler": "DDIM",
        "inference_steps": INFER_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "eta": DDIM_ETA,
        "model_state_key": MODEL_STATE_KEY,
        "expected_checkpoint_step": EXPECTED_STEP,
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "image_mode": "RGB",
    }


def write_progress(metadata_path, manifest_path, output_root, logical_rows, metadata, status):
    manifest_rows = build_manifest_rows(output_root, logical_rows)
    atomic_manifest(manifest_path, manifest_rows)
    expected = len(logical_rows) * len(SEEDS) * SAMPLES_PER_SEED
    metadata["status"] = status
    metadata["updated_at_utc"] = utc_now()
    metadata["logical_tile_rows_completed"] = len(manifest_rows)
    metadata["logical_tile_rows_expected"] = expected
    metadata["manifest"] = manifest_path.name
    atomic_json(metadata_path, metadata)
    return len(manifest_rows), expected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the fixed 32-image SC26 morphology sample set per logical job."
    )
    parser.add_argument(
        "--checkpoint-csv",
        type=Path,
        required=True,
        help="CSV with condition,wave_id,job_name,checkpoint_path[,checkpoint_id].",
    )
    parser.add_argument("--cae-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dedicated evaluation directory outside all original job directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SC26 final morphology sampling requires CUDA.")
    import_project_components()

    checkpoint_csv, logical_rows = read_checkpoint_csv(args.checkpoint_csv)
    output_root = ensure_output_isolated(args.output_root, logical_rows)
    cae_path = args.cae_checkpoint.expanduser().resolve(strict=True)
    if not cae_path.is_file():
        raise ValueError(f"CAE checkpoint is not a file: {cae_path}")
    metadata_path = output_root / "sampling_metadata.json"
    manifest_path = output_root / "sampling_manifest.csv"

    existing = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Existing metadata has incompatible schema: {metadata_path}")
        if existing.get("protocol") != protocol_metadata():
            raise ValueError(f"Existing output uses a different sampling protocol: {output_root}")

    old_fingerprints = {
        item["checkpoint_realpath"]: item
        for item in existing.get("logical_checkpoints", [])
        if item.get("checkpoint_realpath")
    }
    fingerprint_cache = {}
    for logical in logical_rows:
        realpath = Path(logical["checkpoint_realpath"])
        if str(realpath) not in fingerprint_cache:
            digest, stat, reused = cached_sha256(realpath, old_fingerprints)
            fingerprint_cache[str(realpath)] = (digest, stat, reused)
        digest, stat, reused = fingerprint_cache[str(realpath)]
        logical.update(
            {
                "checkpoint_sha256": digest,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "fingerprint_reused": reused,
                "canonical_checkpoint_id": f"sha256_{digest[:16]}",
            }
        )

    old_jobs = {
        (item.get("wave_id"), item.get("job_name"), item.get("checkpoint_realpath"))
        for item in existing.get("logical_checkpoints", [])
    }
    new_jobs = {
        (item["wave_id"], item["job_name"], item["checkpoint_realpath"])
        for item in logical_rows
    }
    if old_jobs and old_jobs != new_jobs:
        raise ValueError(
            "Existing output root was created for a different logical checkpoint set; "
            "use a new --output-root."
        )

    cae_digest = sha256_file(cae_path)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "created_at_utc": existing.get("created_at_utc", utc_now()),
        "updated_at_utc": utc_now(),
        "checkpoint_csv": str(checkpoint_csv),
        "checkpoint_csv_sha256": sha256_file(checkpoint_csv),
        "output_root": str(output_root),
        "cae_checkpoint": str(cae_path),
        "cae_checkpoint_sha256": cae_digest,
        "protocol": protocol_metadata(),
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "diffusers": package_version("diffusers"),
            "cuda_device_index": 0,
            "cuda_device_name": properties.name,
            "cuda_device_total_memory_bytes": properties.total_memory,
        },
        "logical_checkpoints": logical_rows,
        "unique_checkpoint_count": len({row["checkpoint_sha256"] for row in logical_rows}),
    }
    write_progress(
        metadata_path,
        manifest_path,
        output_root,
        logical_rows,
        metadata,
        "in_progress",
    )

    print(f"Loading CAE once: {cae_path}", flush=True)
    cae, latent_channels = load_cae(str(cae_path), device)
    cae.eval()
    scheduler = build_noise_scheduler()

    canonical = {}
    for row in logical_rows:
        canonical.setdefault(row["checkpoint_sha256"], row)

    generated_seed_groups = 0
    reused_seed_groups = 0
    try:
        for index, row in enumerate(canonical.values(), start=1):
            checkpoint_path = Path(row["checkpoint_realpath"])
            canonical_id = row["canonical_checkpoint_id"]
            print(
                f"[{index}/{len(canonical)}] Loading canonical EMA checkpoint once: "
                f"{checkpoint_path}",
                flush=True,
            )
            checkpoint = load_torch_checkpoint(checkpoint_path)
            step, tensors = validate_checkpoint(checkpoint, checkpoint_path, latent_channels)

            incomplete_seeds = [
                seed for seed in SEEDS if not seed_complete(output_root, canonical_id, seed)
            ]
            if incomplete_seeds:
                model = ConditionedUNet(
                    latent_channels,
                    condition_dim=len(PROCESS_FEATURE_NAMES),
                ).to(device)
                model.load_state_dict(checkpoint[MODEL_STATE_KEY], strict=True)
                model.eval()
                for seed in SEEDS:
                    if seed not in incomplete_seeds:
                        reused_seed_groups += 1
                        continue
                    print(
                        f"  sampling process={PROCESS_ID} seed={seed} "
                        f"n={SAMPLES_PER_SEED} DDIM={INFER_STEPS} CFG={GUIDANCE_SCALE}",
                        flush=True,
                    )
                    images, labels = generate_selected_process_grid(
                        cae,
                        model,
                        scheduler,
                        tensors["latent_mean"].to(device),
                        tensors["latent_std"].to(device),
                        tensors["process_mean"].to(device),
                        tensors["process_std"].to(device),
                        device,
                        SAMPLES_PER_SEED,
                        INFER_STEPS,
                        GUIDANCE_SCALE,
                        seed,
                        (PROCESS_ID,),
                    )
                    if labels != [PROCESS_ID] * SAMPLES_PER_SEED:
                        raise RuntimeError(f"Unexpected generated labels: {labels}")
                    save_seed_tiles(images, output_root, canonical_id, seed)
                    generated_seed_groups += 1
                    write_progress(
                        metadata_path,
                        manifest_path,
                        output_root,
                        logical_rows,
                        metadata,
                        "in_progress",
                    )
                del model
            else:
                reused_seed_groups += len(SEEDS)
            del checkpoint
            torch.cuda.empty_cache()

        completed, expected = write_progress(
            metadata_path,
            manifest_path,
            output_root,
            logical_rows,
            metadata,
            "complete",
        )
        if completed != expected:
            raise RuntimeError(f"Completion check failed: {completed}/{expected} logical tile rows.")
        metadata["generated_seed_groups_this_run"] = generated_seed_groups
        metadata["reused_seed_groups_this_run"] = reused_seed_groups
        write_progress(
            metadata_path,
            manifest_path,
            output_root,
            logical_rows,
            metadata,
            "complete",
        )
        print(
            f"Complete: {completed} logical manifest rows, "
            f"{len(canonical) * len(SEEDS) * SAMPLES_PER_SEED} unique tile slots checked.",
            flush=True,
        )
    except Exception:
        metadata["generated_seed_groups_this_run"] = generated_seed_groups
        metadata["reused_seed_groups_this_run"] = reused_seed_groups
        write_progress(
            metadata_path,
            manifest_path,
            output_root,
            logical_rows,
            metadata,
            "failed_or_interrupted",
        )
        raise


if __name__ == "__main__":
    main()
