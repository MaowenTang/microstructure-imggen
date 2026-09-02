#!/usr/bin/env python3

import argparse
import json
import math
import os

from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

from microstructure_e2_ldm import build_noise_scheduler, load_cae, seed_all
from microstructure_e3_conditional_ldm import PROCESS_PARAMETERS
from microstructure_e4_spatial_condition import (
    SpatialConditionedUNet,
    adherence_metrics,
    generate,
    normalize_channels,
    normalize_rows,
    process_features,
    structure_map,
    tensor_to_pil,
)
from microstructure_process_geometry import map_process


PROCESS_DEFAULTS = {
    # Process 4 has broad pool boundaries rather than the regular fine ripples
    # seen in the other three conditions.
    "4": {"radius": 360.0, "spacing": 120.0, "width": 28.0, "contrast": 0.32},
    "5": {"radius": 230.0, "spacing": 32.0, "width": 7.0, "contrast": 0.45},
    "7": {"radius": 290.0, "spacing": 43.0, "width": 10.0, "contrast": 0.40},
    "9": {"radius": 190.0, "spacing": 39.0, "width": 8.0, "contrast": 0.60},
}
PROCESS_BASE_INTENSITY = {"4": 0.52, "5": 0.45, "7": 0.86, "9": 0.67}


def render_ripples(parameters, device):
    size = 512
    coordinates = torch.arange(size, device=device, dtype=torch.float32)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    angle = math.radians(parameters.get("rotation_degrees", 0.0))
    center_x = parameters.get("center_x", 256.0)
    center_y = parameters.get("center_y", 256.0 - parameters["radius"])
    dx = x - center_x
    dy = y - center_y
    rotated_x = math.cos(angle) * dx + math.sin(angle) * dy
    rotated_y = -math.sin(angle) * dx + math.cos(angle) * dy
    aspect = parameters.get("aspect", 1.0)
    elliptical_radius = torch.sqrt(
        (rotated_x / aspect).square() + rotated_y.square() + 1e-8
    )

    spacing = parameters["spacing"]
    radius = parameters["radius"]
    width = parameters["width"]
    phase = torch.remainder(
        elliptical_radius - radius + 0.5 * spacing,
        spacing,
    ) - 0.5 * spacing
    bands = torch.exp(-0.5 * (phase / width).square())
    radial_window = torch.sigmoid(
        (elliptical_radius - (radius - 3.5 * spacing)) / (0.25 * spacing)
    )
    radial_window *= torch.sigmoid(
        ((radius + 3.5 * spacing) - elliptical_radius) / (0.25 * spacing)
    )
    sector = torch.sigmoid((rotated_y + 0.15 * radius) / 12.0)
    bands = bands * radial_window * sector

    base = parameters["base_intensity"]
    illumination = (
        0.04 * (x / size - 0.5)
        + 0.025 * (y / size - 0.5)
    )
    gray = (base + illumination - parameters["contrast"] * bands).clamp(0, 1)
    image = gray[None].repeat(3, 1, 1) * 2.0 - 1.0
    metadata = {
        **parameters,
        "center_x": center_x,
        "center_y": center_y,
    }
    return image, metadata


def ridge_f1(target, generated, tolerance):
    target_threshold = torch.quantile(target.flatten(1), 0.82, dim=1)
    generated_threshold = torch.quantile(generated.flatten(1), 0.82, dim=1)
    target_binary = target >= target_threshold[:, None, None]
    generated_binary = generated >= generated_threshold[:, None, None]
    kernel = 2 * tolerance + 1
    target_dilated = F.max_pool2d(
        target_binary.float()[:, None],
        kernel,
        stride=1,
        padding=tolerance,
    )[:, 0] > 0
    generated_dilated = F.max_pool2d(
        generated_binary.float()[:, None],
        kernel,
        stride=1,
        padding=tolerance,
    )[:, 0] > 0
    precision = (
        (generated_binary & target_dilated).flatten(1).sum(dim=1)
        / generated_binary.flatten(1).sum(dim=1).clamp_min(1)
    )
    recall = (
        (target_binary & generated_dilated).flatten(1).sum(dim=1)
        / target_binary.flatten(1).sum(dim=1).clamp_min(1)
    )
    return 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def recover_geometry(ridge, parameters, bin_width=4.0):
    height, width = ridge.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=ridge.device) * (512.0 / height),
        torch.arange(width, device=ridge.device) * (512.0 / width),
        indexing="ij",
    )
    angle = math.radians(parameters.get("rotation_degrees", 0.0))
    dx = x - parameters["center_x"]
    dy = y - parameters["center_y"]
    rotated_x = math.cos(angle) * dx + math.sin(angle) * dy
    rotated_y = -math.sin(angle) * dx + math.cos(angle) * dy
    elliptical_radius = torch.sqrt(
        (rotated_x / parameters.get("aspect", 1.0)).square()
        + rotated_y.square()
        + 1e-8
    )
    lower = max(0.0, parameters["radius"] - 4 * parameters["spacing"])
    upper = parameters["radius"] + 4 * parameters["spacing"]
    indices = torch.floor((elliptical_radius - lower) / bin_width).long()
    valid = (indices >= 0) & (indices < int((upper - lower) / bin_width) + 1)
    count = int((upper - lower) / bin_width) + 1
    sums = torch.zeros(count, device=ridge.device)
    totals = torch.zeros(count, device=ridge.device)
    sums.scatter_add_(0, indices[valid], ridge[valid])
    totals.scatter_add_(0, indices[valid], torch.ones_like(ridge[valid]))
    profile = sums / totals.clamp_min(1)
    profile = F.avg_pool1d(profile[None, None], 3, stride=1, padding=1)[0, 0]
    threshold = torch.quantile(profile, 0.60)
    candidates = torch.nonzero(
        (profile[1:-1] > profile[:-2])
        & (profile[1:-1] >= profile[2:])
        & (profile[1:-1] > threshold),
        as_tuple=False,
    ).flatten() + 1
    ordered = candidates[torch.argsort(profile[candidates], descending=True)]
    minimum_separation = max(1, int(0.45 * parameters["spacing"] / bin_width))
    selected = []
    for candidate in ordered.tolist():
        if all(abs(candidate - existing) >= minimum_separation for existing in selected):
            selected.append(candidate)
    recovered = sorted(lower + candidate * bin_width for candidate in selected)
    target_radius = parameters["radius"]
    if recovered:
        radius_error = min(abs(value - target_radius) for value in recovered)
    else:
        radius_error = float("nan")
    differences = [
        recovered[index + 1] - recovered[index]
        for index in range(len(recovered) - 1)
        if 0.5 * parameters["spacing"]
        <= recovered[index + 1] - recovered[index]
        <= 1.5 * parameters["spacing"]
    ]
    recovered_spacing = (
        float(torch.tensor(differences).median().item())
        if differences
        else float("nan")
    )
    spacing_error = (
        abs(recovered_spacing - parameters["spacing"])
        if math.isfinite(recovered_spacing)
        else float("nan")
    )
    return {
        "recovered_peak_radii": recovered,
        "central_radius_error_px": radius_error,
        "recovered_spacing_px": recovered_spacing,
        "spacing_error_px": spacing_error,
    }


def save_pair_grid(conditions, generated, labels, path):
    columns = len(labels)
    header = 30
    canvas = Image.new("RGB", (columns * 512, 2 * 512 + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        draw.text((index * 512 + 5, 7), label, fill="black")
        canvas.paste(tensor_to_pil(conditions[index]), (index * 512, header))
        canvas.paste(tensor_to_pil(generated[index]), (index * 512, header + 512))
    canvas.save(path)


def run_group(
    name,
    specifications,
    cae,
    model,
    scheduler,
    checkpoint,
    device,
    args,
):
    condition_images = []
    metadata = []
    process_rows = []
    labels = []
    for label, process, parameters in specifications:
        image, values = render_ripples(parameters, device)
        condition_images.append(image)
        metadata.append(values)
        if "process_values" in parameters:
            process_values = parameters["process_values"]
            power = process_values["laser_power"]
            speed = process_values["scan_speed"]
            process_rows.append(
                (power, speed, process_values["time"], power / speed)
            )
        else:
            power, speed, duration = PROCESS_PARAMETERS[process]
            process_rows.append((power, speed, duration, power / speed))
        labels.append(label)
    condition_images = torch.stack(condition_images)
    raw_structure = structure_map(condition_images)
    statistics = checkpoint["statistics"]
    structure = normalize_channels(
        raw_structure,
        statistics["structure_mean"].to(device),
        statistics["structure_std"].to(device),
    )
    process = normalize_rows(
        torch.tensor(process_rows, dtype=torch.float32, device=device),
        statistics["process_mean"].to(device),
        statistics["process_std"].to(device),
    )
    generated = generate(
        cae,
        model,
        scheduler,
        checkpoint["latent_mean"].to(device),
        checkpoint["latent_std"].to(device),
        structure,
        process,
        device,
        args.infer_steps,
        args.guidance_scale,
        args.seed,
    )
    generated_structure = structure_map(generated.to(device)).cpu()
    target_structure = raw_structure.cpu()
    adherence = adherence_metrics(target_structure, generated)
    f1 = {
        f"ridge_f1_tolerance_{tolerance}_latent_px": ridge_f1(
            target_structure[:, 1],
            generated_structure[:, 1],
            tolerance,
        ).tolist()
        for tolerance in (1, 2, 4)
    }
    geometry = [
        recover_geometry(generated_structure[index, 1].to(device), values)
        for index, values in enumerate(metadata)
    ]
    result = {
        "name": name,
        "labels": labels,
        "parameters": metadata,
        "adherence": adherence,
        **f1,
        "recovered_geometry": geometry,
    }
    save_pair_grid(
        condition_images.cpu(),
        generated,
        labels,
        os.path.join(args.out_root, f"{name}.png"),
    )
    sample_root = os.path.join(args.out_root, f"{name}_samples")
    os.makedirs(sample_root, exist_ok=True)
    for index, image in enumerate(generated):
        tensor_to_pil(image).save(os.path.join(sample_root, f"sample_{index:03d}.png"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cae_checkpoint", required=True)
    parser.add_argument("--spatial_checkpoint", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--infer_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=5050)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Procedural sampling requires CUDA.")
    seed_all(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out_root, exist_ok=True)
    checkpoint = torch.load(args.spatial_checkpoint, map_location="cpu")
    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    model = SpatialConditionedUNet(latent_channels, 3, 4).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    scheduler = build_noise_scheduler()

    default_specs = []
    for process in ("4", "5", "7", "9"):
        parameters = {
            **PROCESS_DEFAULTS[process],
            "base_intensity": PROCESS_BASE_INTENSITY[process],
            "aspect": 1.05,
        }
        default_specs.append((f"P{process}", process, parameters))

    spacing_specs = []
    for spacing in (12.0, 20.0, 32.0, 48.0):
        parameters = {
            "radius": 220.0,
            "spacing": spacing,
            "width": max(3.0, 0.22 * spacing),
            "contrast": 0.26,
            "base_intensity": PROCESS_BASE_INTENSITY["9"],
            "aspect": 1.05,
        }
        spacing_specs.append((f"spacing={spacing:.0f}px", "9", parameters))

    radius_specs = []
    for radius in (120.0, 180.0, 260.0, 360.0):
        parameters = {
            "radius": radius,
            "spacing": 24.0,
            "width": 5.0,
            "contrast": 0.26,
            "base_intensity": PROCESS_BASE_INTENSITY["9"],
            "aspect": 1.05,
        }
        radius_specs.append((f"radius={radius:.0f}px", "9", parameters))

    stochastic_specs = []
    for index in range(8):
        parameters = {
            **PROCESS_DEFAULTS["9"],
            "base_intensity": PROCESS_BASE_INTENSITY["9"],
            "aspect": 1.05,
        }
        stochastic_specs.append((f"same geometry #{index + 1}", "9", parameters))

    interpolation_specs = []
    interpolation_queries = (
        (525.0, 3.225, 25.0),
        (412.5, 5.125, 30.0),
        (337.5, 6.50, 25.0),
    )
    for power, speed, duration in interpolation_queries:
        mapped = map_process(power, speed, duration)
        geometry = mapped["geometry"]
        presence = geometry["regular_ripple_probability"]
        parameters = {
            "radius": geometry["radius_px"],
            "spacing": geometry["spacing_px"],
            "width": geometry["width_px"],
            "contrast": geometry["contrast"] * (0.2 + 0.8 * presence),
            "base_intensity": geometry["base_intensity"],
            "aspect": 1.05,
            "process_values": {
                "laser_power": power,
                "scan_speed": speed,
                "time": duration,
            },
        }
        interpolation_specs.append(
            (
                f"{power:.0f}W, {speed:.2f}mm/s",
                mapped["nearest_anchor"],
                parameters,
            )
        )

    results = [
        run_group(
            "process_defaults",
            default_specs,
            cae,
            model,
            scheduler,
            checkpoint,
            device,
            args,
        ),
        run_group(
            "spacing_sweep",
            spacing_specs,
            cae,
            model,
            scheduler,
            checkpoint,
            device,
            args,
        ),
        run_group(
            "radius_sweep",
            radius_specs,
            cae,
            model,
            scheduler,
            checkpoint,
            device,
            args,
        ),
        run_group(
            "stochasticity",
            stochastic_specs,
            cae,
            model,
            scheduler,
            checkpoint,
            device,
            args,
        ),
        run_group(
            "process_interpolation",
            interpolation_specs,
            cae,
            model,
            scheduler,
            checkpoint,
            device,
            args,
        ),
    ]
    with open(os.path.join(args.out_root, "procedural_metrics.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
