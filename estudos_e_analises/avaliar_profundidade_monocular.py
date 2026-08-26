#!/usr/bin/env python3
"""Avalia um modelo monocular usando frames de referencia gravados pelo pipeline.

O ajuste de escala e feito somente no conjunto de calibracao. O conjunto de
validacao permanece separado para evitar calibracao e avaliacao nos mesmos dados.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spatial_mapping.depth_model import MetricDepthOnnx


RANGE_BINS_M = ((0.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0))


def _sample_frames(run_dir, count):
    frames = sorted((run_dir / "frames").glob("frame_*.npz"))
    if not frames:
        raise FileNotFoundError(f"nenhum frame encontrado em {run_dir / 'frames'}")
    indices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    return [frames[index] for index in np.unique(indices)]


def _load_pairs(
    model, run_dir, count, min_reference_depth_m, max_reference_depth_m
):
    pairs = []
    latencies_ms = []
    for frame_path in _sample_frames(run_dir, count):
        with np.load(frame_path) as frame:
            rgb = frame["rgb_bgr"]
            reference = frame["reference_depth_m"].astype(np.float32)
        started = time.perf_counter()
        prediction = model.predict(rgb)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        valid = (
            np.isfinite(reference)
            & np.isfinite(prediction)
            & (reference >= min_reference_depth_m)
            & (reference <= max_reference_depth_m)
            & (prediction > 0.0)
        )
        pairs.append(
            {
                "frame": frame_path.name,
                "reference": reference[valid].astype(np.float64),
                "prediction": prediction[valid].astype(np.float64),
            }
        )
    return pairs, latencies_ms


def _fit_scale(pairs):
    numerator = sum(np.dot(item["prediction"], item["reference"]) for item in pairs)
    denominator = sum(np.dot(item["prediction"], item["prediction"]) for item in pairs)
    if denominator <= 0.0:
        raise ValueError("nao ha predicoes validas para calibrar a escala")
    return float(numerator / denominator)


def _metrics(pairs, scale):
    reference = np.concatenate([item["reference"] for item in pairs])
    prediction = np.concatenate([item["prediction"] for item in pairs]) * scale
    error = prediction - reference
    result = {
        "pixels": int(reference.size),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "abs_rel": float(np.mean(np.abs(error) / reference)),
        "bias_m": float(np.mean(error)),
        "prediction_median_m": float(np.median(prediction)),
        "reference_median_m": float(np.median(reference)),
    }
    bins = {}
    for lower, upper in RANGE_BINS_M:
        mask = (reference >= lower) & (reference < upper)
        if not np.any(mask):
            continue
        bin_error = prediction[mask] - reference[mask]
        bins[f"{lower:g}-{upper:g}m"] = {
            "pixels": int(np.count_nonzero(mask)),
            "mae_m": float(np.mean(np.abs(bin_error))),
            "bias_m": float(np.mean(bin_error)),
            "abs_rel": float(np.mean(np.abs(bin_error) / reference[mask])),
            "positive_error_p95_m": float(
                np.percentile(np.maximum(bin_error, 0.0), 95)
            ),
        }
    result["range_bins"] = bins
    per_frame_scales = []
    for item in pairs:
        pred = item["prediction"]
        ref = item["reference"]
        per_frame_scales.append(float(np.dot(pred, ref) / np.dot(pred, pred)))
    result["per_frame_scale"] = {
        "min": float(np.min(per_frame_scales)),
        "median": float(np.median(per_frame_scales)),
        "max": float(np.max(per_frame_scales)),
        "std": float(np.std(per_frame_scales)),
    }
    return result


def _latency_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT
        / "models/depth_anything_v2_metric_vkitti_vits_686x518_fp32.onnx",
    )
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--min-reference-depth-m", type=float, default=0.5)
    parser.add_argument("--max-reference-depth-m", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-qualified", action="store_true")
    args = parser.parse_args()

    model = MetricDepthOnnx(
        args.model,
        input_width=686,
        input_height=518,
        backend="onnxruntime",
        max_depth_m=80.0,
        input_aspect_tolerance=0.03,
    )
    calibration, calibration_latency = _load_pairs(
        model, args.calibration_run, args.frames,
        args.min_reference_depth_m, args.max_reference_depth_m
    )
    validation, validation_latency = _load_pairs(
        model, args.validation_run, args.frames,
        args.min_reference_depth_m, args.max_reference_depth_m
    )
    fitted_scale = _fit_scale(calibration)
    validation_raw = _metrics(validation, 1.0)
    validation_latency_summary = _latency_summary(validation_latency)
    near_metrics = validation_raw["range_bins"].get("0.5-5m", {})
    middle_metrics = validation_raw["range_bins"].get("5-10m", {})
    limits = {
        "mae_m_max": 2.0,
        "abs_rel_max": 0.30,
        "near_mae_m_max": 1.0,
        "near_positive_bias_m_max": 0.5,
        "near_positive_error_p95_m_max": 1.25,
        "middle_mae_m_max": 1.5,
        "latency_p95_ms_max": 400.0,
    }
    checks = {
        "mae": validation_raw["mae_m"] <= limits["mae_m_max"],
        "abs_rel": validation_raw["abs_rel"] <= limits["abs_rel_max"],
        "near_mae": near_metrics.get("mae_m", float("inf"))
        <= limits["near_mae_m_max"],
        "near_positive_bias": near_metrics.get("bias_m", float("inf"))
        <= limits["near_positive_bias_m_max"],
        "near_positive_error_p95": near_metrics.get(
            "positive_error_p95_m", float("inf")
        )
        <= limits["near_positive_error_p95_m_max"],
        "middle_mae": middle_metrics.get("mae_m", float("inf"))
        <= limits["middle_mae_m_max"],
        "latency": validation_latency_summary["p95_ms"]
        <= limits["latency_p95_ms_max"],
    }
    report = {
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "calibration_run": str(args.calibration_run.resolve()),
        "validation_run": str(args.validation_run.resolve()),
        "sampled_frames_per_run": args.frames,
        "min_reference_depth_m": args.min_reference_depth_m,
        "max_reference_depth_m": args.max_reference_depth_m,
        "fitted_scale_calibration_only": fitted_scale,
        "calibration_raw": _metrics(calibration, 1.0),
        "calibration_scaled": _metrics(calibration, fitted_scale),
        "validation_raw": validation_raw,
        "validation_scaled": _metrics(validation, fitted_scale),
        "latency": {
            "calibration": _latency_summary(calibration_latency),
            "validation": validation_latency_summary,
        },
        "qualification": {
            "passed": all(checks.values()),
            "limits": limits,
            "checks": checks,
        },
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
