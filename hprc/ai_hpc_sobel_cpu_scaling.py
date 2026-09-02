#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in IMG_EXTS:
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def patch_meta(path):
    name = Path(path).name
    source = name.split("_p", 1)[0]
    process = source.split(".", 1)[0]
    return source, process


def sobel_score(path):
    image = Image.open(path).convert("L").resize((512, 512), Image.Resampling.BICUBIC)
    x = np.asarray(image, dtype=np.float32) / 255.0
    padded = np.pad(x, ((1, 1), (1, 1)), mode="edge")
    tl = padded[:-2, :-2]
    tc = padded[:-2, 1:-1]
    tr = padded[:-2, 2:]
    ml = padded[1:-1, :-2]
    mr = padded[1:-1, 2:]
    bl = padded[2:, :-2]
    bc = padded[2:, 1:-1]
    br = padded[2:, 2:]
    gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br
    gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br
    mag = np.sqrt(gx * gx + gy * gy + 1e-12).reshape(-1)
    k = max(1, int(0.10 * mag.size))
    topk = np.partition(mag, mag.size - k)[-k:]
    source, process = patch_meta(path)
    return {
        "path": path,
        "source": source,
        "process": process,
        "score": float(topk.mean()),
    }


def score_all(paths, workers):
    if workers <= 1:
        return [sobel_score(path) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(sobel_score, paths, chunksize=8))


def select_top(records, top_ratio):
    by_source = {}
    for record in records:
        by_source.setdefault(record["source"], []).append(record)
    kept = []
    summary = {}
    for source, source_records in sorted(by_source.items()):
        source_records = sorted(source_records, key=lambda item: item["score"], reverse=True)
        keep_n = max(1, int(round(len(source_records) * top_ratio)))
        source_kept = source_records[:keep_n]
        kept.extend(source_kept)
        summary[source] = {
            "total": len(source_records),
            "kept": len(source_kept),
            "threshold": source_kept[-1]["score"],
        }
    kept = sorted(kept, key=lambda item: item["path"])
    return kept, summary


def stable_hash(records):
    h = hashlib.sha256()
    for record in records:
        rel = record["path"]
        h.update(rel.encode("utf-8"))
        h.update(b"\t")
        h.update(f"{record['score']:.10f}".encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--top_ratio", type=float, default=0.50)
    parser.add_argument("--workers", default="1,4,8,16,32,64")
    args = parser.parse_args()

    paths = list_images(args.data_root)
    if not paths:
        raise RuntimeError(f"No images under {args.data_root}")
    worker_values = [int(item) for item in args.workers.split(",") if item.strip()]
    rows = []
    reference_hash = None
    reference_selected_hash = None
    for workers in worker_values:
        t0 = time.perf_counter()
        records = score_all(paths, workers)
        score_seconds = time.perf_counter() - t0
        kept, summary = select_top(records, args.top_ratio)
        record_hash = stable_hash(sorted(records, key=lambda item: item["path"]))
        selected_hash = stable_hash(kept)
        if reference_hash is None:
            reference_hash = record_hash
            reference_selected_hash = selected_hash
        rows.append({
            "workers": workers,
            "images": len(paths),
            "selected": len(kept),
            "score_seconds": f"{score_seconds:.4f}",
            "images_per_second": f"{len(paths) / score_seconds:.4f}",
            "record_hash": record_hash,
            "selected_hash": selected_hash,
            "record_hash_matches_w1": record_hash == reference_hash,
            "selected_hash_matches_w1": selected_hash == reference_selected_hash,
            "summary_json": json.dumps(summary, sort_keys=True),
        })
        print(
            f"[SOBEL_CPU] workers={workers} images={len(paths)} "
            f"seconds={score_seconds:.2f} img_s={len(paths)/score_seconds:.2f} "
            f"selected={len(kept)} hash_match={selected_hash == reference_selected_hash}",
            flush=True,
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SOBEL_CPU] saved {out_csv}", flush=True)


if __name__ == "__main__":
    main()
