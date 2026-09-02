#!/usr/bin/env python3

import argparse
import json
import math


ANCHORS = {
    "4": {
        "process": {"laser_power": 600.0, "scan_speed": 1.80, "time": 30.0},
        "geometry": {
            "regular_ripple_probability": 0.0,
            "radius_px": 360.0,
            "spacing_px": 120.0,
            "width_px": 28.0,
            "contrast": 0.32,
            "base_intensity": 0.52,
        },
    },
    "5": {
        "process": {"laser_power": 450.0, "scan_speed": 4.65, "time": 20.0},
        "geometry": {
            "regular_ripple_probability": 1.0,
            "radius_px": 230.0,
            "spacing_px": 32.0,
            "width_px": 7.0,
            "contrast": 0.45,
            "base_intensity": 0.45,
        },
    },
    "7": {
        "process": {"laser_power": 375.0, "scan_speed": 5.60, "time": 40.0},
        "geometry": {
            "regular_ripple_probability": 1.0,
            "radius_px": 290.0,
            "spacing_px": 43.0,
            "width_px": 10.0,
            "contrast": 0.40,
            "base_intensity": 0.86,
        },
    },
    "9": {
        "process": {"laser_power": 300.0, "scan_speed": 7.40, "time": 10.0},
        "geometry": {
            "regular_ripple_probability": 1.0,
            "radius_px": 190.0,
            "spacing_px": 39.0,
            "width_px": 8.0,
            "contrast": 0.60,
            "base_intensity": 0.67,
        },
    },
}
PROCESS_KEYS = ("laser_power", "scan_speed", "time", "linear_energy")
GEOMETRY_KEYS = (
    "regular_ripple_probability",
    "radius_px",
    "spacing_px",
    "width_px",
    "contrast",
    "base_intensity",
)


def process_vector(process):
    power = process["laser_power"]
    speed = process["scan_speed"]
    return (power, speed, process["time"], power / speed)


def column_statistics(rows):
    columns = list(zip(*rows))
    means = [sum(column) / len(column) for column in columns]
    stds = []
    for column, mean in zip(columns, means):
        variance = sum((value - mean) ** 2 for value in column) / max(
            1, len(column) - 1
        )
        stds.append(max(math.sqrt(variance), 1e-8))
    return means, stds


def map_process(laser_power, scan_speed, time, temperature=1.0):
    query = process_vector(
        {
            "laser_power": laser_power,
            "scan_speed": scan_speed,
            "time": time,
        }
    )
    labels = sorted(ANCHORS, key=int)
    anchor_process = [process_vector(ANCHORS[label]["process"]) for label in labels]
    means, stds = column_statistics(anchor_process)

    def standardized(row):
        return tuple((value - mean) / std for value, mean, std in zip(row, means, stds))

    query_standard = standardized(query)
    anchor_standard = [standardized(row) for row in anchor_process]
    distances = [
        math.sqrt(
            sum(
                (query_value - anchor_value) ** 2
                for query_value, anchor_value in zip(query_standard, anchor)
            )
        )
        for anchor in anchor_standard
    ]
    nearest = min(range(len(labels)), key=distances.__getitem__)
    if distances[nearest] < 1e-10:
        weights = [1.0 if index == nearest else 0.0 for index in range(len(labels))]
    else:
        logits = [-(distance**2) / max(temperature, 1e-6) for distance in distances]
        maximum = max(logits)
        unnormalized = [math.exp(value - maximum) for value in logits]
        total = sum(unnormalized)
        weights = [value / total for value in unnormalized]

    geometry = {}
    for key in GEOMETRY_KEYS:
        geometry[key] = sum(
            weight * ANCHORS[label]["geometry"][key]
            for label, weight in zip(labels, weights)
        )
    geometry["regular_ripple_probability"] = min(
        1.0, max(0.0, geometry["regular_ripple_probability"])
    )
    micrometers_per_pixel = 500.0 / 512.0
    geometry["radius_um"] = geometry["radius_px"] * micrometers_per_pixel
    geometry["spacing_um"] = geometry["spacing_px"] * micrometers_per_pixel

    nearest_distance = distances[nearest]
    return {
        "query": {
            "laser_power": laser_power,
            "scan_speed": scan_speed,
            "time": time,
            "linear_energy": laser_power / scan_speed,
        },
        "geometry": geometry,
        "anchor_weights": {
            label: weight for label, weight in zip(labels, weights)
        },
        "nearest_anchor": labels[nearest],
        "normalized_distance_to_nearest_anchor": nearest_distance,
        "extrapolation_warning": nearest_distance > 1.5,
        "interpretation": (
            "RBF interpolation over four observed process settings. "
            "It is a controllable prior, not validated causal generalization."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laser_power", type=float, required=True)
    parser.add_argument("--scan_speed", type=float, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out_json")
    args = parser.parse_args()
    result = map_process(
        args.laser_power,
        args.scan_speed,
        args.time,
        temperature=args.temperature,
    )
    text = json.dumps(result, indent=2)
    if args.out_json:
        with open(args.out_json, "w") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
