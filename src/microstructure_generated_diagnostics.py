#!/usr/bin/env python3

import argparse
import json
import math
import os

import torch
from torch.utils.data import DataLoader

from microstructure_e1_cae import build_spatial_split, list_images
from microstructure_e2_diagnostics import (
    PathDataset,
    descriptor_summary,
    encode_collection,
    nearest_distances,
    pairwise_diversity,
)
from microstructure_e2_ldm import load_cae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--cae_checkpoint", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Generated diagnostics require CUDA.")
    device = torch.device("cuda")
    all_paths = list_images(args.data_root)
    train_paths, validation_paths, _, _ = build_spatial_split(all_paths)
    generated_paths = list_images(args.generated_root)
    if len(generated_paths) < 2:
        raise RuntimeError("At least two generated samples are required.")

    def loader(paths):
        return DataLoader(
            PathDataset(paths),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    cae, _ = load_cae(args.cae_checkpoint, device)
    train_desc, train_features, _ = encode_collection(cae, loader(train_paths), device)
    val_desc, val_features, _ = encode_collection(cae, loader(validation_paths), device)
    gen_desc, gen_features, _ = encode_collection(cae, loader(generated_paths), device)

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-5)
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    gen_features = (gen_features - mean) / std
    val_nearest = nearest_distances(val_features, train_features) / math.sqrt(
        train_features.shape[1]
    )
    gen_nearest = nearest_distances(gen_features, train_features) / math.sqrt(
        train_features.shape[1]
    )

    descriptor_mean = train_desc.mean(dim=0)
    descriptor_std = train_desc.std(dim=0).clamp_min(1e-6)
    generated_z = torch.abs((gen_desc - descriptor_mean) / descriptor_std)
    result = {
        "counts": {
            "train": len(train_paths),
            "validation": len(validation_paths),
            "generated": len(generated_paths),
        },
        "cae_feature": {
            "validation_nearest_train_median": float(val_nearest.median().item()),
            "generated_nearest_train_median": float(gen_nearest.median().item()),
            "nearest_ratio_generated_over_validation": float(
                gen_nearest.median().item() / val_nearest.median().item()
            ),
            "validation_pairwise_diversity": pairwise_diversity(val_features),
            "generated_pairwise_diversity": pairwise_diversity(gen_features),
        },
        "generated_descriptors": descriptor_summary(gen_desc),
        "generated_descriptor_outlier": {
            "mean_absolute_z": float(generated_z.mean().item()),
            "fraction_absolute_z_gt_2": float(
                (generated_z > 2).float().mean().item()
            ),
            "fraction_absolute_z_gt_3": float(
                (generated_z > 3).float().mean().item()
            ),
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
