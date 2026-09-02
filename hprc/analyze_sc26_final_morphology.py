#!/usr/bin/env python3
"""Analyze the final-checkpoint morphology of the SC26 GH200 proxy jobs.

The script deliberately keeps the experimental hierarchy intact:

1. summarize 32 generated tiles within each logical job;
2. average job medians within each execution wave; and
3. report the mean and sample standard deviation of four wave statistics.

Generated tiles are therefore observations within a job, not independent
experimental repetitions.  The real images come from the process-9 subset of
the Sobel-curated training manifest and are a target reference, not a holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, __version__ as PIL_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from manufacturing_letters_evidence import (  # noqa: E402
    image_array,
    metrics_for_array,
    nearest,
    thumbnail_feature,
)


BASE_METRICS = [
    "intensity_mean",
    "intensity_std",
    "intensity_iqr",
    "dark_fraction",
    "bright_fraction",
    "extreme_fraction",
    "block_mean_std",
    "block_mean_range",
    "rgb_tint_std",
    "edge_mean",
    "edge_p90",
    "orientation_coherence",
    "spectral_spacing_px",
    "spectral_peak_confidence",
    "spectral_target_band_fraction",
    "radial_ripple_score",
    "radial_spacing_px",
    "radial_alignment",
    "orientation_error_deg",
    "q01",
    "q05",
    "q50",
    "q95",
    "q99",
]

GENERATED_METRICS = BASE_METRICS + [
    "nearest_reference_distance",
    "nearest_reference_ratio_to_real_loo_median",
]

REFERENCE_RELATIVE_METRIC_SOURCES = [
    "spectral_spacing_px",
    "edge_mean",
    "radial_spacing_px",
    "radial_alignment",
    "orientation_error_deg",
]

REFERENCE_RELATIVE_METRICS = [
    f"{metric}_relative_error_to_reference_median"
    for metric in REFERENCE_RELATIVE_METRIC_SOURCES
]

JOB_DERIVED_METRICS = REFERENCE_RELATIVE_METRICS + [
    "radial_spacing_detection_fraction"
]

AGGREGATED_METRICS = GENERATED_METRICS + JOB_DERIVED_METRICS

OK_SAMPLE_STATUSES = {
    "",
    "ok",
    "complete",
    "completed",
    "generated",
    "reused",
    "cached",
    "success",
}

CANONICAL_SAMPLING_ID_BY_TRAIN_SEED = {
    1: "single_rep1",
    2: "single_rep2",
    101: "c2_a",
    102: "c2_b",
    201: "c3_a",
    202: "c3_b",
    203: "c3_c",
}


def fail(message: str) -> None:
    raise ValueError(message)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def first_value(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = clean(row.get(name))
        if value:
            return value
    return ""


def int_value(value: Any, label: str) -> int:
    text = clean(value)
    if not text:
        fail(f"Missing integer field: {label}")
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {label}: {text!r}") from exc
    if not number.is_integer():
        fail(f"Expected an integer for {label}, got {text!r}")
    return int(number)


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path, preferred_json_keys: Sequence[str]) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    fail(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
        return rows
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = None
        for key in preferred_json_keys:
            candidate = value.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            fail(
                f"Could not find a row list in {path}; tried keys "
                f"{', '.join(preferred_json_keys)}"
            )
    else:
        fail(f"Unsupported table structure in {path}")
    if not all(isinstance(row, dict) for row in rows):
        fail(f"All rows in {path} must be JSON objects")
    return [dict(row) for row in rows]


def parse_job_count(row: Mapping[str, Any]) -> int:
    explicit = first_value(
        row, ("jobs", "job_count", "jobs_in_wave", "concurrency")
    )
    if explicit:
        return int_value(explicit, "job count")
    label = first_value(row, ("condition", "mode")).lower().replace("_", "-")
    if label in {"single", "warm-single", "1", "1-job", "job-1"}:
        return 1
    for count in (2, 3):
        if label in {
            str(count),
            f"c{count}",
            f"concurrent{count}",
            f"concurrent-{count}",
            f"{count}-job",
            f"job-{count}",
        }:
            return count
    fail(f"Cannot infer job count from condition/mode {label!r}")
    return -1


def condition_name(job_count: int) -> str:
    if job_count == 1:
        return "warm-single"
    return f"concurrent-{job_count}"


def derived_checkpoint_id(wave_id: str, job_name: str) -> str:
    """Match the paired sampler's optional checkpoint-ID derivation."""
    value = re.sub(
        r"[^A-Za-z0-9._-]+", "_", f"{wave_id}__{job_name}"
    ).strip("._-")
    if not value:
        fail(f"Cannot derive checkpoint_id from {wave_id!r}, {job_name!r}")
    return value


def logical_job_key(row: Mapping[str, Any]) -> tuple[str, str]:
    wave_id = first_value(row, ("wave_id", "wave"))
    job_name = first_value(row, ("job_name", "job", "logical_job"))
    if not wave_id or not job_name:
        fail(f"Every logical job needs wave_id and job_name; row={dict(row)!r}")
    return wave_id, job_name


def canonicalize_jobs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in rows:
        warm_policy = clean(source.get("warm_policy")).lower()
        if warm_policy in {"warmup_excluded", "warm-up-excluded", "excluded"}:
            continue
        measured = clean(source.get("measured")).lower()
        if measured in {"0", "false", "no"}:
            continue
        status = clean(source.get("status")).lower()
        if status and status not in {"complete", "completed", "success", "ok"}:
            fail(f"Logical job is not complete: {dict(source)!r}")
        wave_id, job_name = logical_job_key(source)
        key = (wave_id, job_name)
        if key in seen:
            fail(f"Duplicate logical job mapping for wave/job {key!r}")
        seen.add(key)
        job_count = parse_job_count(source)
        checkpoint_path = first_value(
            source,
            ("checkpoint_path", "checkpoint_realpath", "last_checkpoint", "checkpoint"),
        )
        if not checkpoint_path:
            fail(f"Missing checkpoint path for logical job {key!r}")
        checkpoint_realpath = first_value(source, ("checkpoint_realpath",))
        checkpoint_sha256 = first_value(
            source, ("checkpoint_sha256", "checkpoint_hash", "sha256")
        ).lower()
        checkpoint_id = first_value(source, ("checkpoint_id",)) or derived_checkpoint_id(
            wave_id, job_name
        )
        training_seed_text = first_value(
            source, ("training_seed", "train_seed", "seed")
        )
        training_seed = (
            int_value(training_seed_text, "training seed")
            if training_seed_text
            else None
        )
        canonical_sampling_id = (
            CANONICAL_SAMPLING_ID_BY_TRAIN_SEED.get(training_seed, "")
            if training_seed is not None
            else ""
        )
        jobs.append(
            {
                "condition": condition_name(job_count),
                "job_count": job_count,
                "wave_id": wave_id,
                "job_name": job_name,
                "checkpoint_id": checkpoint_id,
                "canonical_checkpoint_id": first_value(
                    source, ("canonical_checkpoint_id",)
                ),
                "checkpoint_path": checkpoint_path,
                "checkpoint_realpath": checkpoint_realpath,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_step": first_value(source, ("checkpoint_step", "final_step")),
                "training_seed": training_seed if training_seed is not None else "",
                "canonical_sampling_id": canonical_sampling_id,
            }
        )
    jobs.sort(key=lambda row: (row["job_count"], row["wave_id"], row["job_name"]))
    return jobs


def normalized_path_token(value: Any) -> str:
    text = clean(value)
    return os.path.normpath(text) if text else ""


def build_job_lookups(
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[Any, list[dict[str, Any]]]]:
    lookups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "key": defaultdict(list),
        "id": defaultdict(list),
        "sha": defaultdict(list),
        "path": defaultdict(list),
        "name": defaultdict(list),
        "canonical": defaultdict(list),
        "canonical_checkpoint": defaultdict(list),
    }
    for job in jobs:
        key = (job["wave_id"], job["job_name"])
        lookups["key"][key].append(dict(job))
        lookups["id"][job["checkpoint_id"]].append(dict(job))
        if job["checkpoint_sha256"]:
            lookups["sha"][job["checkpoint_sha256"]].append(dict(job))
        for path_field in ("checkpoint_path", "checkpoint_realpath"):
            token = normalized_path_token(job[path_field])
            if token:
                lookups["path"][token].append(dict(job))
        lookups["name"][job["job_name"]].append(dict(job))
        if job["canonical_sampling_id"]:
            lookups["canonical"][job["canonical_sampling_id"]].append(dict(job))
        if job["canonical_checkpoint_id"]:
            lookups["canonical_checkpoint"][job["canonical_checkpoint_id"]].append(
                dict(job)
            )
    return lookups


def jobs_for_sample(
    row: Mapping[str, Any], lookups: Mapping[str, Mapping[Any, list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    wave_id = first_value(row, ("wave_id", "wave"))
    job_name = first_value(row, ("job_name", "job", "logical_job"))
    if wave_id and job_name:
        matches = list(lookups["key"].get((wave_id, job_name), []))
        if matches:
            return matches

    checkpoint_id = first_value(row, ("checkpoint_id",))
    canonical_checkpoint_id = first_value(row, ("canonical_checkpoint_id",))
    for token in (
        checkpoint_id,
        first_value(row, ("canonical_sampling_id",)),
        job_name,
    ):
        if token and token in lookups["canonical"]:
            return list(lookups["canonical"][token])
    if (
        canonical_checkpoint_id
        and canonical_checkpoint_id in lookups["canonical_checkpoint"]
    ):
        return list(lookups["canonical_checkpoint"][canonical_checkpoint_id])
    if checkpoint_id and checkpoint_id in lookups["id"]:
        return list(lookups["id"][checkpoint_id])
    checkpoint_sha = first_value(
        row, ("checkpoint_sha256", "checkpoint_hash", "sha256")
    ).lower()
    if checkpoint_sha and checkpoint_sha in lookups["sha"]:
        return list(lookups["sha"][checkpoint_sha])
    for field in ("checkpoint_realpath", "checkpoint_path", "checkpoint"):
        token = normalized_path_token(row.get(field))
        if token and token in lookups["path"]:
            return list(lookups["path"][token])
    if job_name and job_name in lookups["name"]:
        matches = list(lookups["name"][job_name])
        if len(matches) == 1:
            return matches
        fail(
            f"Ambiguous job_name-only sample mapping for {job_name!r}: "
            f"{len(matches)} logical jobs"
        )
    fail(f"Cannot map sample row to a logical job: {dict(row)!r}")
    return []


def resolve_tile_path(value: Any, manifest_path: Path) -> Path:
    path = Path(clean(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def canonicalize_samples(
    rows: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    expected_process: str,
    expected_checkpoint_step: int,
) -> list[dict[str, Any]]:
    lookups = build_job_lookups(jobs)
    samples: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int, int, str]] = set()
    for source in rows:
        status = clean(source.get("status")).lower()
        if status not in OK_SAMPLE_STATUSES:
            fail(f"Sample row has unsuccessful status {status!r}: {dict(source)!r}")
        process_id = first_value(source, ("process_id", "process", "target_process"))
        if process_id and process_id != expected_process:
            fail(f"Expected process {expected_process}, found {process_id!r}")
        checkpoint_step = first_value(source, ("checkpoint_step", "final_step", "step"))
        if checkpoint_step and int_value(checkpoint_step, "checkpoint_step") != expected_checkpoint_step:
            fail(
                f"Expected checkpoint step {expected_checkpoint_step}, found {checkpoint_step!r}"
            )
        model_state_key = first_value(source, ("model_state_key", "state_key"))
        if model_state_key and "ema" not in model_state_key.lower():
            fail(f"Expected EMA sampling state, found {model_state_key!r}")
        seed = int_value(first_value(source, ("seed", "sample_seed")), "sample seed")
        sample_index = int_value(
            first_value(source, ("sample_index", "tile_index", "index")),
            "sample index",
        )
        tile_value = first_value(source, ("tile_path", "sample_path", "path", "source_png"))
        if not tile_value:
            fail(f"Sample row has no tile path: {dict(source)!r}")
        tile_path = resolve_tile_path(tile_value, manifest_path)
        tile_sha256 = first_value(source, ("tile_sha256", "sample_sha256")).lower()

        for job in jobs_for_sample(source, lookups):
            source_checkpoint_sha = first_value(
                source, ("checkpoint_sha256", "checkpoint_hash")
            ).lower()
            if (
                source_checkpoint_sha
                and job["checkpoint_sha256"]
                and source_checkpoint_sha != job["checkpoint_sha256"]
            ):
                fail(
                    f"Checkpoint hash mismatch for {(job['wave_id'], job['job_name'])!r}"
                )
            identity = (
                job["wave_id"],
                job["job_name"],
                seed,
                sample_index,
                str(tile_path),
            )
            if identity in identities:
                fail(f"Duplicate logical sample row: {identity!r}")
            identities.add(identity)
            samples.append(
                {
                    **dict(job),
                    "canonical_checkpoint_id": first_value(
                        source, ("canonical_checkpoint_id",)
                    )
                    or job["canonical_checkpoint_id"],
                    "checkpoint_realpath": first_value(
                        source, ("checkpoint_realpath",)
                    )
                    or job["checkpoint_realpath"],
                    "checkpoint_sha256": source_checkpoint_sha
                    or job["checkpoint_sha256"],
                    "process_id": expected_process,
                    "seed": seed,
                    "sample_index": sample_index,
                    "tile_path": str(tile_path),
                    "tile_sha256": tile_sha256,
                }
            )
    samples.sort(
        key=lambda row: (
            row["job_count"],
            row["wave_id"],
            row["job_name"],
            row["seed"],
            row["sample_index"],
            row["tile_path"],
        )
    )
    return samples


def hydrate_jobs_from_samples(
    jobs: list[dict[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> None:
    """Carry sampler-computed hashes back into the logical job mapping."""
    samples_by_job: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_job[(sample["wave_id"], sample["job_name"])].append(sample)
    for job in jobs:
        key = (job["wave_id"], job["job_name"])
        job_samples = samples_by_job.get(key, [])
        for field in (
            "checkpoint_sha256",
            "checkpoint_realpath",
            "canonical_checkpoint_id",
        ):
            values = {clean(sample.get(field)) for sample in job_samples}
            values.discard("")
            if len(values) > 1:
                fail(f"{key!r}: inconsistent {field} values in sample manifest")
            if values:
                sampled_value = next(iter(values))
                mapped_value = clean(job.get(field))
                if mapped_value and mapped_value != sampled_value:
                    fail(f"{key!r}: job-map/sample-manifest mismatch for {field}")
                job[field] = sampled_value


def parse_expected_seeds(value: str) -> list[int]:
    seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not seeds:
        fail("At least one expected seed is required")
    if len(seeds) != len(set(seeds)):
        fail("Expected seeds must be unique")
    return seeds


def validate_design(
    jobs: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    expected_jobs: int,
    expected_conditions: int,
    expected_waves_per_condition: int,
    expected_tiles_per_job: int,
    expected_seeds: Sequence[int],
    samples_per_seed: int,
) -> None:
    if len(jobs) != expected_jobs:
        fail(f"Expected {expected_jobs} logical jobs, found {len(jobs)}")
    jobs_with_training_seed = [job for job in jobs if clean(job.get("training_seed"))]
    if jobs_with_training_seed and len(jobs_with_training_seed) != len(jobs):
        fail("Training-seed canonical mapping is partially populated")
    if expected_jobs == 24 and len(jobs_with_training_seed) != 24:
        fail("The formal 24-job design requires train_seed for every logical job")
    if jobs_with_training_seed:
        missing_alias = [
            (job["wave_id"], job["job_name"], job["training_seed"])
            for job in jobs
            if not job["canonical_sampling_id"]
        ]
        if missing_alias:
            fail(f"Training seeds have no canonical sampling ID: {missing_alias!r}")
        if expected_jobs == 24:
            observed_aliases = {job["canonical_sampling_id"] for job in jobs}
            expected_aliases = set(CANONICAL_SAMPLING_ID_BY_TRAIN_SEED.values())
            if observed_aliases != expected_aliases:
                fail(
                    "Formal canonical sampling IDs differ from the seven expected IDs: "
                    f"{sorted(observed_aliases)!r}"
                )
            expected_alias_counts = {
                "single_rep1": 2,
                "single_rep2": 2,
                "c2_a": 4,
                "c2_b": 4,
                "c3_a": 4,
                "c3_b": 4,
                "c3_c": 4,
            }
            observed_alias_counts = {
                alias: sum(job["canonical_sampling_id"] == alias for job in jobs)
                for alias in expected_aliases
            }
            if observed_alias_counts != expected_alias_counts:
                fail(
                    "Formal canonical sampling ID multiplicities are wrong: "
                    f"{observed_alias_counts!r}"
                )
            for alias in expected_aliases:
                alias_jobs = [
                    job for job in jobs if job["canonical_sampling_id"] == alias
                ]
                if len({job["wave_id"] for job in alias_jobs}) != len(alias_jobs):
                    fail(f"Canonical sampling ID {alias!r} maps twice within a wave")
    conditions = {job["condition"] for job in jobs}
    if len(conditions) != expected_conditions:
        fail(f"Expected {expected_conditions} conditions, found {sorted(conditions)!r}")

    jobs_by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        jobs_by_condition[job["condition"]].append(job)
    for condition, condition_jobs in jobs_by_condition.items():
        waves = {job["wave_id"] for job in condition_jobs}
        if len(waves) != expected_waves_per_condition:
            fail(
                f"{condition}: expected {expected_waves_per_condition} waves, "
                f"found {len(waves)}"
            )
        for wave_id in waves:
            wave_jobs = [job for job in condition_jobs if job["wave_id"] == wave_id]
            expected_wave_jobs = int(wave_jobs[0]["job_count"])
            if len(wave_jobs) != expected_wave_jobs:
                fail(
                    f"{wave_id}: expected {expected_wave_jobs} jobs from condition, "
                    f"found {len(wave_jobs)}"
                )

    samples_by_job: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_job[(sample["wave_id"], sample["job_name"])].append(sample)
    expected_seed_set = set(expected_seeds)
    for job in jobs:
        key = (job["wave_id"], job["job_name"])
        job_samples = samples_by_job.get(key, [])
        if len(job_samples) != expected_tiles_per_job:
            fail(
                f"{key!r}: expected {expected_tiles_per_job} tiles, found {len(job_samples)}"
            )
        unique_paths = {sample["tile_path"] for sample in job_samples}
        if len(unique_paths) != expected_tiles_per_job:
            fail(
                f"{key!r}: expected {expected_tiles_per_job} distinct tile paths, "
                f"found {len(unique_paths)}"
            )
        observed_seeds = {int(sample["seed"]) for sample in job_samples}
        if observed_seeds != expected_seed_set:
            fail(
                f"{key!r}: expected seeds {sorted(expected_seed_set)}, "
                f"found {sorted(observed_seeds)}"
            )
        for seed in expected_seeds:
            seed_samples = [
                sample for sample in job_samples if int(sample["seed"]) == seed
            ]
            count = len(seed_samples)
            if count != samples_per_seed:
                fail(
                    f"{key!r}, seed {seed}: expected {samples_per_seed} samples, found {count}"
                )
            observed_indices = {int(sample["sample_index"]) for sample in seed_samples}
            expected_indices = set(range(samples_per_seed))
            if observed_indices != expected_indices:
                fail(
                    f"{key!r}, seed {seed}: expected sample indices "
                    f"{sorted(expected_indices)}, found {sorted(observed_indices)}"
                )


def resolve_reference_path(
    record: Mapping[str, Any],
    manifest_path: Path,
    manifest_data_root: str,
    reference_root: Path | None,
) -> Path:
    raw = first_value(record, ("path", "image_path", "source_png"))
    if not raw:
        fail(f"Reference record has no path: {dict(record)!r}")
    original = Path(raw).expanduser()
    candidates: list[Path] = []
    if reference_root is not None:
        if manifest_data_root:
            try:
                relative = original.relative_to(Path(manifest_data_root))
                candidates.append(reference_root / relative)
            except ValueError:
                pass
        candidates.extend((reference_root / original.name, reference_root / raw))
    if original.is_absolute():
        candidates.append(original)
    else:
        if manifest_data_root:
            candidates.append(Path(manifest_data_root) / original)
        candidates.append(manifest_path.parent / original)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else original.resolve()


def load_reference_records(
    manifest_path: Path,
    expected_process: str,
    expected_count: int,
    reference_root: Path | None,
) -> list[dict[str, Any]]:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("records"), list):
        fail(f"Curated manifest must contain a records list: {manifest_path}")
    data_root = clean(manifest.get("data_root"))
    selected: list[dict[str, Any]] = []
    for record in manifest["records"]:
        if not isinstance(record, dict):
            fail("Curated manifest records must be JSON objects")
        process = first_value(record, ("process", "process_id"))
        if not process:
            source = first_value(record, ("source",))
            process = source.split(".", 1)[0] if source else ""
        if process != expected_process:
            continue
        resolved = resolve_reference_path(
            record, manifest_path, data_root, reference_root
        )
        selected.append({**dict(record), "resolved_path": str(resolved)})
    if len(selected) != expected_count:
        fail(
            f"Expected {expected_count} curated process-{expected_process} references, "
            f"found {len(selected)}"
        )
    paths = [row["resolved_path"] for row in selected]
    if len(paths) != len(set(paths)):
        fail("Curated process reference contains duplicate paths")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        example = ", ".join(missing[:3])
        fail(f"Missing {len(missing)} curated reference images; examples: {example}")
    return selected


def with_orientation_error(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {key: float(value) for key, value in metrics.items()}
    alignment = result.get("radial_alignment", float("nan"))
    if math.isfinite(alignment):
        result["orientation_error_deg"] = math.degrees(
            math.acos(min(1.0, max(0.0, alignment)))
        )
    else:
        result["orientation_error_deg"] = float("nan")
    return result


def analyze_reference(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    for index, record in enumerate(records):
        path = Path(record["resolved_path"])
        with Image.open(path) as source:
            # This is the exact real-domain transform used by the gray E12 proxy.
            rgb = image_array(source.convert("L").convert("RGB"))
        metrics = with_orientation_error(metrics_for_array(rgb))
        rows.append(
            {
                "reference_index": index,
                "reference_path": str(path),
                "source": first_value(record, ("source",)),
                "process": first_value(record, ("process", "process_id")) or "9",
                "sobel_selection_score": first_value(record, ("score",)),
                **metrics,
            }
        )
        features.append(thumbnail_feature(rgb))
    return rows, np.vstack(features).astype(np.float32)


def analyze_generated(
    samples: Sequence[Mapping[str, Any]], expected_size: int
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    cache: dict[tuple[str, str], tuple[dict[str, float], np.ndarray, str]] = {}
    for sample in samples:
        path = Path(sample["tile_path"])
        if not path.is_file():
            fail(f"Generated tile does not exist: {path}")
        declared_hash = clean(sample.get("tile_sha256")).lower()
        cache_key = (str(path), declared_hash)
        if cache_key not in cache:
            actual_hash = sha256_file(path)
            if declared_hash and actual_hash != declared_hash:
                fail(f"Tile SHA-256 mismatch: {path}")
            with Image.open(path) as source:
                if source.size != (expected_size, expected_size):
                    fail(
                        f"Expected an extracted {expected_size}x{expected_size} tile, "
                        f"found {source.size} at {path}"
                    )
                rgb = image_array(source.convert("RGB"), size=expected_size)
            metrics = with_orientation_error(metrics_for_array(rgb))
            cache[cache_key] = (metrics, thumbnail_feature(rgb), actual_hash)
        metrics, feature, actual_hash = cache[cache_key]
        rows.append(
            {
                "condition": sample["condition"],
                "job_count": sample["job_count"],
                "wave_id": sample["wave_id"],
                "job_name": sample["job_name"],
                "checkpoint_id": sample["checkpoint_id"],
                "canonical_checkpoint_id": sample["canonical_checkpoint_id"],
                "canonical_sampling_id": sample["canonical_sampling_id"],
                "checkpoint_path": sample["checkpoint_path"],
                "checkpoint_sha256": sample["checkpoint_sha256"],
                "process_id": sample["process_id"],
                "seed": sample["seed"],
                "sample_index": sample["sample_index"],
                "tile_path": str(path),
                "tile_sha256": actual_hash,
                **metrics,
            }
        )
        features.append(feature)
    return (
        rows,
        np.vstack(features).astype(np.float32),
        {
            "logical_tile_rows": len(rows),
            "unique_analyzed_tiles": len(cache),
        },
    )


def nearest_excluding_self(
    features: np.ndarray, batch: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] < 2:
        fail("Leave-one-out nearest-neighbor calibration needs at least two references")
    best_dist: list[float] = []
    best_index: list[int] = []
    for start in range(0, features.shape[0], batch):
        query = features[start : start + batch]
        distances = np.sqrt(
            ((query[:, None, :] - features[None, :, :]) ** 2).mean(axis=2)
        )
        local = np.arange(distances.shape[0])
        global_indices = start + local
        distances[local, global_indices] = np.inf
        indices = np.argmin(distances, axis=1)
        best_dist.extend(distances[local, indices].tolist())
        best_index.extend(indices.tolist())
    return np.asarray(best_dist), np.asarray(best_index)


def add_nearest_neighbors(
    reference_rows: list[dict[str, Any]],
    reference_features: np.ndarray,
    tile_rows: list[dict[str, Any]],
    generated_features: np.ndarray,
    nn_batch: int,
) -> tuple[list[dict[str, Any]], float]:
    mean = reference_features.mean(axis=0, keepdims=True)
    std = reference_features.std(axis=0, keepdims=True).clip(min=1e-5)
    reference_z = (reference_features - mean) / std
    generated_z = (generated_features - mean) / std
    real_dist, real_index = nearest_excluding_self(reference_z, batch=nn_batch)
    calibration = float(np.median(real_dist))
    if not math.isfinite(calibration) or calibration <= 0:
        fail(f"Invalid real leave-one-out NN median: {calibration}")
    generated_dist, generated_index = nearest(
        generated_z, reference_z, batch=nn_batch
    )

    nearest_rows: list[dict[str, Any]] = []
    for row, distance, index in zip(reference_rows, real_dist, real_index):
        neighbor = reference_rows[int(index)]
        row["loo_nearest_reference_index"] = int(index)
        row["loo_nearest_reference_path"] = neighbor["reference_path"]
        row["loo_nearest_distance"] = float(distance)
        row["loo_distance_ratio_to_real_loo_median"] = float(distance / calibration)
        nearest_rows.append(
            {
                "query_kind": "real_reference_leave_one_out",
                "condition": "curated-reference",
                "wave_id": "",
                "job_name": "",
                "query_id": row["reference_index"],
                "query_path": row["reference_path"],
                "nearest_reference_index": int(index),
                "nearest_reference_path": neighbor["reference_path"],
                "distance": float(distance),
                "distance_ratio_to_real_loo_median": float(distance / calibration),
            }
        )
    for row, distance, index in zip(tile_rows, generated_dist, generated_index):
        neighbor = reference_rows[int(index)]
        row["nearest_reference_index"] = int(index)
        row["nearest_reference_path"] = neighbor["reference_path"]
        row["nearest_reference_distance"] = float(distance)
        row["nearest_reference_ratio_to_real_loo_median"] = float(
            distance / calibration
        )
        nearest_rows.append(
            {
                "query_kind": "generated_tile",
                "condition": row["condition"],
                "wave_id": row["wave_id"],
                "job_name": row["job_name"],
                "query_id": f"seed{row['seed']}_sample{row['sample_index']}",
                "query_path": row["tile_path"],
                "nearest_reference_index": int(index),
                "nearest_reference_path": neighbor["reference_path"],
                "distance": float(distance),
                "distance_ratio_to_real_loo_median": float(distance / calibration),
            }
        )
    return nearest_rows, calibration


def finite_summary(values: Iterable[Any]) -> dict[str, float | int]:
    array = np.asarray([float_or_nan(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "n": 0,
            "mean": float("nan"),
            "sample_sd": float("nan"),
            "median": float("nan"),
            "q05": float("nan"),
            "q95": float("nan"),
        }
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size >= 2 else float("nan"),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def summarize_jobs(
    tile_rows: Sequence[Mapping[str, Any]],
    reference_medians: Mapping[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in tile_rows:
        grouped[(row["wave_id"], row["job_name"])].append(row)
    output = []
    for (_, _), rows in grouped.items():
        first = rows[0]
        summary: dict[str, Any] = {
            "condition": first["condition"],
            "job_count": first["job_count"],
            "wave_id": first["wave_id"],
            "job_name": first["job_name"],
            "checkpoint_id": first["checkpoint_id"],
            "canonical_checkpoint_id": first["canonical_checkpoint_id"],
            "canonical_sampling_id": first["canonical_sampling_id"],
            "checkpoint_path": first["checkpoint_path"],
            "checkpoint_sha256": first["checkpoint_sha256"],
            "n_tiles": len(rows),
            "n_unique_tile_paths": len({row["tile_path"] for row in rows}),
        }
        for metric in GENERATED_METRICS:
            stats = finite_summary(row.get(metric) for row in rows)
            summary[f"{metric}_n"] = stats["n"]
            summary[f"{metric}_median"] = stats["median"]
            summary[f"{metric}_q05"] = stats["q05"]
            summary[f"{metric}_q95"] = stats["q95"]
        finite_radial_spacing = int(summary["radial_spacing_px_n"])
        summary["radial_spacing_detection_fraction"] = (
            finite_radial_spacing / len(rows) if rows else float("nan")
        )
        for metric in REFERENCE_RELATIVE_METRIC_SOURCES:
            job_median = float_or_nan(summary[f"{metric}_median"])
            reference_median = float_or_nan(reference_medians.get(metric))
            relative_name = f"{metric}_relative_error_to_reference_median"
            if (
                math.isfinite(job_median)
                and math.isfinite(reference_median)
                and abs(reference_median) > 1e-12
            ):
                summary[relative_name] = abs(job_median - reference_median) / abs(
                    reference_median
                )
            else:
                summary[relative_name] = float("nan")
        output.append(summary)
    output.sort(key=lambda row: (row["job_count"], row["wave_id"], row["job_name"]))
    return output


def summarize_waves(job_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in job_rows:
        grouped[(row["condition"], row["wave_id"])].append(row)
    output = []
    for (_, _), rows in grouped.items():
        first = rows[0]
        summary: dict[str, Any] = {
            "condition": first["condition"],
            "job_count": first["job_count"],
            "wave_id": first["wave_id"],
            "n_jobs": len(rows),
            "n_logical_tiles": sum(int(row["n_tiles"]) for row in rows),
            "n_unique_checkpoints": len(
                {
                    row["checkpoint_sha256"] or row["checkpoint_path"]
                    for row in rows
                }
            ),
        }
        for metric in AGGREGATED_METRICS:
            job_field = (
                f"{metric}_median" if metric in GENERATED_METRICS else metric
            )
            stats = finite_summary(row.get(job_field) for row in rows)
            summary[f"{metric}_n_finite_job_values"] = stats["n"]
            summary[f"{metric}_mean_across_jobs"] = stats["mean"]
        output.append(summary)
    output.sort(key=lambda row: (row["job_count"], row["wave_id"]))
    return output


def summarize_conditions(
    wave_rows: Sequence[Mapping[str, Any]],
    job_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    waves_by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    jobs_by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in wave_rows:
        waves_by_condition[row["condition"]].append(row)
    for row in job_rows:
        jobs_by_condition[row["condition"]].append(row)
    output = []
    for condition, rows in waves_by_condition.items():
        jobs = jobs_by_condition[condition]
        first = rows[0]
        summary: dict[str, Any] = {
            "condition": condition,
            "job_count": first["job_count"],
            "n_waves": len(rows),
            "n_logical_jobs": len(jobs),
            "n_logical_tiles": sum(int(row["n_tiles"]) for row in jobs),
            "n_unique_checkpoints": len(
                {
                    row["checkpoint_sha256"] or row["checkpoint_path"]
                    for row in jobs
                }
            ),
        }
        for metric in AGGREGATED_METRICS:
            stats = finite_summary(
                row.get(f"{metric}_mean_across_jobs") for row in rows
            )
            summary[f"{metric}_n_finite_waves"] = stats["n"]
            summary[f"{metric}_wave_mean"] = stats["mean"]
            summary[f"{metric}_wave_sample_sd"] = stats["sample_sd"]
        output.append(summary)
    output.sort(key=lambda row: row["job_count"])
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    return fields


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def format_pm(mean: Any, sd: Any, digits: int = 3) -> str:
    mean_value = float_or_nan(mean)
    sd_value = float_or_nan(sd)
    if not math.isfinite(mean_value):
        return "NA"
    if not math.isfinite(sd_value):
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} ± {sd_value:.{digits}f}"


def markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    path: Path,
    condition_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    calibration: float,
    tile_counts: Mapping[str, int],
) -> None:
    compact_rows = []
    for row in condition_rows:
        compact_rows.append(
            [
                row["condition"],
                row["n_waves"],
                row["n_logical_jobs"],
                row["n_unique_checkpoints"],
                format_pm(row["edge_mean_wave_mean"], row["edge_mean_wave_sample_sd"]),
                format_pm(
                    row["orientation_coherence_wave_mean"],
                    row["orientation_coherence_wave_sample_sd"],
                ),
                format_pm(
                    row["spectral_spacing_px_wave_mean"],
                    row["spectral_spacing_px_wave_sample_sd"],
                    1,
                ),
                format_pm(
                    row["radial_ripple_score_wave_mean"],
                    row["radial_ripple_score_wave_sample_sd"],
                ),
                format_pm(
                    row["radial_spacing_detection_fraction_wave_mean"],
                    row["radial_spacing_detection_fraction_wave_sample_sd"],
                ),
                format_pm(
                    row["nearest_reference_ratio_to_real_loo_median_wave_mean"],
                    row[
                        "nearest_reference_ratio_to_real_loo_median_wave_sample_sd"
                    ],
                ),
            ]
        )
    relative_rows = []
    for row in condition_rows:
        relative_rows.append(
            [
                row["condition"],
                format_pm(
                    row[
                        "spectral_spacing_px_relative_error_to_reference_median_wave_mean"
                    ],
                    row[
                        "spectral_spacing_px_relative_error_to_reference_median_wave_sample_sd"
                    ],
                ),
                format_pm(
                    row["edge_mean_relative_error_to_reference_median_wave_mean"],
                    row[
                        "edge_mean_relative_error_to_reference_median_wave_sample_sd"
                    ],
                ),
                format_pm(
                    row[
                        "radial_spacing_px_relative_error_to_reference_median_wave_mean"
                    ],
                    row[
                        "radial_spacing_px_relative_error_to_reference_median_wave_sample_sd"
                    ],
                ),
                format_pm(
                    row[
                        "radial_alignment_relative_error_to_reference_median_wave_mean"
                    ],
                    row[
                        "radial_alignment_relative_error_to_reference_median_wave_sample_sd"
                    ],
                ),
                format_pm(
                    row[
                        "orientation_error_deg_relative_error_to_reference_median_wave_mean"
                    ],
                    row[
                        "orientation_error_deg_relative_error_to_reference_median_wave_sample_sd"
                    ],
                ),
            ]
        )
    reference_edge = finite_summary(row["edge_mean"] for row in reference_rows)["median"]
    reference_orientation = finite_summary(
        row["orientation_coherence"] for row in reference_rows
    )["median"]
    reference_spacing = finite_summary(
        row["spectral_spacing_px"] for row in reference_rows
    )["median"]
    reference_ripple = finite_summary(
        row["radial_ripple_score"] for row in reference_rows
    )["median"]

    report = [
        "# SC26 final-checkpoint morphology summary",
        "",
        "Each cell is the mean ± sample SD across execution-wave statistics. "
        "Within a wave, the statistic is the mean of job-level medians; each job "
        "median uses its generated tiles.",
        "",
        markdown_table(
            compact_rows,
            [
                "Condition",
                "Waves",
                "Logical jobs",
                "Unique checkpoints",
                "Sobel edge mean",
                "Orientation coherence",
                "FFT spacing (px)",
                "Ripple score",
                "Radial-spacing finite fraction",
                "NN ratio",
            ],
        ),
        "",
        "Relative errors below are absolute job-median deviations divided by the "
        "corresponding curated-reference median, then aggregated through the same "
        "job → wave → condition hierarchy.",
        "",
        markdown_table(
            relative_rows,
            [
                "Condition",
                "FFT spacing rel. error",
                "Sobel edge rel. error",
                "Radial spacing rel. error",
                "Radial alignment rel. error",
                "Effective radial-angle rel. error",
            ],
        ),
        "",
        "## Curated process-9 reference",
        "",
        f"The reference contains {len(reference_rows)} Sobel-curated process-9 training "
        "patches. Images were converted with `L → RGB` before analysis. Reference "
        f"medians were edge={reference_edge:.3f}, orientation={reference_orientation:.3f}, "
        f"FFT spacing={reference_spacing:.1f} px, and ripple={reference_ripple:.3f}.",
        "",
        f"The real leave-one-out nearest-neighbor median was {calibration:.6f}. "
        "`NN ratio` is a generated tile's nearest-reference distance divided by this "
        "calibration median.",
        "",
        "## Interpretation limits",
        "",
        f"- The {len(reference_rows)} real patches are the curated training target, "
        "not an independent holdout.",
        "- Generated tiles are not independent repetitions. Inference follows the "
        "job → wave → condition hierarchy shown above.",
        "- `orientation_error_deg = degrees(arccos(radial_alignment))` is an "
        "effective radial angular deviation, not an estimated global direction angle.",
        "- Radial-spacing quantiles use finite detections only; the detection fraction "
        "is reported separately so missing detections are not silently discarded.",
        "- Reused checkpoint hashes and reused sample files remain separate logical jobs "
        "only where the execution protocol contains those jobs. The unique-checkpoint "
        "column makes this reuse visible.",
        "- These are 3,000-iteration expert proxies. They do not establish full-training "
        "model quality or generalization.",
        "",
        "## Output counts",
        "",
        f"- Logical generated tile rows: {tile_counts['logical_tile_rows']}",
        f"- Unique generated tile files analyzed: {tile_counts['unique_analyzed_tiles']}",
        "- Machine-readable files: `reference_metrics.csv`, `tile_metrics.csv`, "
        "`nearest_neighbors.csv`, `job_summary.csv`, `wave_summary.csv`, "
        "`condition_summary.csv`, and `metadata.json`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CPU FFT/Sobel/orientation/ripple and nearest-reference metrics "
            "for the SC26 final GH200 checkpoints without treating tiles as repeats."
        )
    )
    parser.add_argument("--job-map", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--curated-manifest", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Optional local replacement for the data_root stored in the curated manifest.",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="Markdown path; defaults to OUT_ROOT/morphology_summary.md.",
    )
    parser.add_argument("--expected-process", default="9")
    parser.add_argument("--expected-checkpoint-step", type=int, default=3000)
    parser.add_argument("--expected-reference-count", type=int, default=750)
    parser.add_argument("--expected-jobs", type=int, default=24)
    parser.add_argument("--expected-conditions", type=int, default=3)
    parser.add_argument("--expected-waves-per-condition", type=int, default=4)
    parser.add_argument("--expected-tiles-per-job", type=int, default=32)
    parser.add_argument(
        "--expected-seeds",
        default="9090,9091,9092,9093,9094,9095,9096,9097",
    )
    parser.add_argument("--samples-per-seed", type=int, default=4)
    parser.add_argument("--expected-tile-size", type=int, default=512)
    parser.add_argument(
        "--nn-batch",
        type=int,
        default=4,
        help="Small batches bound memory for the handcrafted nearest-neighbor feature.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_seeds = parse_expected_seeds(args.expected_seeds)
    for input_path in (args.job_map, args.sample_manifest, args.curated_manifest):
        if not input_path.is_file():
            fail(f"Input file does not exist: {input_path}")
    if args.samples_per_seed * len(expected_seeds) != args.expected_tiles_per_job:
        fail(
            "samples-per-seed × number of expected seeds must equal "
            "expected-tiles-per-job"
        )
    if args.nn_batch < 1:
        fail("nn-batch must be positive")

    job_source = read_table(args.job_map, ("jobs", "records", "mappings"))
    jobs = canonicalize_jobs(job_source)
    sample_source = read_table(
        args.sample_manifest, ("samples", "tiles", "records")
    )
    samples = canonicalize_samples(
        sample_source,
        jobs,
        args.sample_manifest,
        args.expected_process,
        args.expected_checkpoint_step,
    )
    hydrate_jobs_from_samples(jobs, samples)
    validate_design(
        jobs,
        samples,
        expected_jobs=args.expected_jobs,
        expected_conditions=args.expected_conditions,
        expected_waves_per_condition=args.expected_waves_per_condition,
        expected_tiles_per_job=args.expected_tiles_per_job,
        expected_seeds=expected_seeds,
        samples_per_seed=args.samples_per_seed,
    )

    reference_records = load_reference_records(
        args.curated_manifest,
        args.expected_process,
        args.expected_reference_count,
        args.reference_root,
    )
    print(
        f"reference: {len(reference_records)} curated process-{args.expected_process} patches",
        flush=True,
    )
    reference_rows, reference_features = analyze_reference(reference_records)
    print(f"generated: {len(samples)} logical tile rows", flush=True)
    tile_rows, generated_features, tile_counts = analyze_generated(
        samples, args.expected_tile_size
    )
    nearest_rows, calibration = add_nearest_neighbors(
        reference_rows,
        reference_features,
        tile_rows,
        generated_features,
        args.nn_batch,
    )

    reference_summary = {
        metric: finite_summary(row.get(metric) for row in reference_rows)
        for metric in BASE_METRICS
    }
    reference_medians = {
        metric: float_or_nan(summary["median"])
        for metric, summary in reference_summary.items()
    }
    job_rows = summarize_jobs(tile_rows, reference_medians)
    wave_rows = summarize_waves(job_rows)
    condition_rows = summarize_conditions(wave_rows, job_rows)
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.report.resolve()
        if args.report is not None
        else out_root / "morphology_summary.md"
    )

    write_csv(
        out_root / "reference_metrics.csv",
        reference_rows,
        union_fieldnames(reference_rows),
    )
    write_csv(out_root / "tile_metrics.csv", tile_rows, union_fieldnames(tile_rows))
    write_csv(
        out_root / "nearest_neighbors.csv",
        nearest_rows,
        union_fieldnames(nearest_rows),
    )
    write_csv(out_root / "job_summary.csv", job_rows, union_fieldnames(job_rows))
    write_csv(out_root / "wave_summary.csv", wave_rows, union_fieldnames(wave_rows))
    write_csv(
        out_root / "condition_summary.csv",
        condition_rows,
        union_fieldnames(condition_rows),
    )

    checkpoint_groups: dict[str, list[str]] = defaultdict(list)
    canonical_sampling_groups: dict[str, list[str]] = defaultdict(list)
    for job in jobs:
        token = job["checkpoint_sha256"] or job["checkpoint_path"]
        logical_id = f"{job['wave_id']}::{job['job_name']}"
        checkpoint_groups[token].append(logical_id)
        if job["canonical_sampling_id"]:
            canonical_sampling_groups[job["canonical_sampling_id"]].append(logical_id)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "job_map": str(args.job_map.resolve()),
            "job_map_sha256": sha256_file(args.job_map),
            "sample_manifest": str(args.sample_manifest.resolve()),
            "sample_manifest_sha256": sha256_file(args.sample_manifest),
            "curated_manifest": str(args.curated_manifest.resolve()),
            "curated_manifest_sha256": sha256_file(args.curated_manifest),
            "reference_root_override": (
                str(args.reference_root.resolve()) if args.reference_root else None
            ),
        },
        "protocol": {
            "checkpoint": f"last checkpoint at iteration {args.expected_checkpoint_step}",
            "model_state": "EMA",
            "process": args.expected_process,
            "sample_seeds": expected_seeds,
            "samples_per_seed": args.samples_per_seed,
            "tiles_per_logical_job": args.expected_tiles_per_job,
            "real_transform": "PIL L -> RGB, matching gray E12 training domain",
            "reference_role": (
                "Sobel-curated process-9 training target; not an independent holdout"
            ),
            "nearest_neighbor_feature": (
                "32x32 grayscale + 32x32 Sobel magnitude + intensity/color summaries; "
                "features standardized by curated-reference mean and SD"
            ),
            "nearest_neighbor_distance": (
                "root mean squared Euclidean distance in standardized feature space"
            ),
            "nearest_neighbor_calibration": (
                "median real-reference leave-one-out nearest-neighbor distance"
            ),
            "orientation_error_deg": (
                "degrees(arccos(clamp(radial_alignment, 0, 1))); an effective radial "
                "angular deviation, not a global direction angle"
            ),
            "reference_relative_error": (
                "absolute(job median - curated-reference median) / "
                "absolute(curated-reference median)"
            ),
            "radial_spacing_missingness": (
                "q05/median/q95 use finite radial_spacing_px values only; the job-level "
                "finite detection fraction is propagated to wave and condition summaries"
            ),
            "statistical_hierarchy": (
                "tile median [q05, q95] within job; mean of job medians within wave; "
                "condition mean +/- sample SD across wave statistics"
            ),
            "statistical_unit_warning": (
                "tiles and jobs that reuse a checkpoint are not independent model repeats"
            ),
        },
        "counts": {
            "logical_jobs": len(jobs),
            "sample_manifest_rows_before_logical_expansion": len(sample_source),
            "waves": len(wave_rows),
            "conditions": len(condition_rows),
            "curated_reference_images": len(reference_rows),
            **tile_counts,
            "unique_checkpoint_tokens": len(checkpoint_groups),
            "canonical_sampling_ids": len(canonical_sampling_groups),
        },
        "checkpoint_reuse": dict(sorted(checkpoint_groups.items())),
        "canonical_sampling_mapping": dict(sorted(canonical_sampling_groups.items())),
        "real_leave_one_out_nn_median": calibration,
        "reference_metric_summary": reference_summary,
        "metrics": BASE_METRICS,
        "metric_implementation": str(
            (SRC_ROOT / "manufacturing_letters_evidence.py").resolve()
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL_VERSION,
            "platform": platform.platform(),
        },
        "validation": {
            "strict_expected_counts_passed": True,
            "canonical_train_seed_mapping_complete_and_unambiguous": (
                len(canonical_sampling_groups)
                == len(CANONICAL_SAMPLING_ID_BY_TRAIN_SEED)
                if args.expected_jobs == 24
                else None
            ),
            "generated_tile_dimensions_checked": True,
            "declared_tile_sha256_checked_when_present": True,
        },
    }
    with (out_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_report(report_path, condition_rows, reference_rows, calibration, tile_counts)

    print(f"wrote {out_root}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
