"""Gera figuras reprodutiveis da avaliacao final da profundidade v19."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from avaliar_mapeamento_monocular_offline import load_config
from spatial_mapping.depth_model import MetricDepthOnnx
from spatial_mapping.geometry import CameraIntrinsics
from spatial_mapping.navigation import SpatialNavigator

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "datasets/spatial_mapping/run_20260831_082836_332335"
CACHE = ROOT / ("logs/spatial_offline_sweep/manual_v19_final_validation_01/"
                "prediction_cache/b56cbb291b8fade37a0b5f298fdd4bbe5e45de05f92628a1956828b0fc902dc4/"
                "reference_reserved_03")
OUTPUTS = [ROOT / "Parte_Escrita/ModeloTCC_Artigo_CC_Latex/figuras",
           ROOT / "Parte_Escrita/ModeloTCC_Monografia_CC_Latex/figuras"]


def save_depth_comparison(rgb, pred, gt, frame_number, output_name):
    """Salva a comparacao RGB, v19, Gazebo e erro com estilo unico."""

    valid = (pred > 0) & (gt > 0) & np.isfinite(pred) & np.isfinite(gt)
    error = np.where(valid, np.abs(pred - gt), np.nan)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1), constrained_layout=True)
    axes[0].imshow(rgb); axes[0].set_title("Imagem RGB")
    vmax = float(np.nanpercentile(gt[gt > 0], 98))
    im1 = axes[1].imshow(pred, cmap="turbo", vmin=0, vmax=vmax); axes[1].set_title("Profundidade monocular v19")
    axes[2].imshow(gt, cmap="turbo", vmin=0, vmax=vmax); axes[2].set_title("Profundidade do Gazebo")
    im2 = axes[3].imshow(error, cmap="magma", vmin=0, vmax=float(np.nanpercentile(error, 98))); axes[3].set_title("Erro absoluto por pixel")
    for ax in axes: ax.axis("off")
    fig.colorbar(im1, ax=axes[1:3], label="Profundidade (m)", shrink=.72)
    fig.colorbar(im2, ax=axes[3], label="Erro absoluto (m)", shrink=.72)
    fig.suptitle(f"Comparacao no quadro {int(frame_number):06d}", fontsize=12)
    primary = OUTPUTS[0] / output_name
    primary.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(primary, dpi=180); plt.close(fig)
    for directory in OUTPUTS[1:]:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary, directory / primary.name)
    return primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-path", type=Path)
    parser.add_argument("--frame-number", type=int)
    parser.add_argument(
        "--output-name",
        default="resultados_comparacao_profundidade_v19.png",
    )
    args = parser.parse_args()

    if args.frame_path is not None:
        frame_path = args.frame_path.expanduser().resolve()
        if args.frame_number is None:
            args.frame_number = int(frame_path.stem.replace("frame_", ""))
        with np.load(frame_path) as frame:
            bgr = frame["rgb_bgr"]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt = frame["reference_depth_m"].astype(float)
        model = MetricDepthOnnx(
            ROOT / "models/depth_anything_v2_metric_baylands_vits_v19_final_686x518_fp32.onnx",
            input_width=686,
            input_height=518,
            backend="onnxruntime",
            max_depth_m=80.0,
            input_aspect_tolerance=0.03,
        )
        pred = model.predict(bgr).astype(float)
        pred[pred > 0] = np.maximum(pred[pred > 0] - 1.2, 0.0)
        output = save_depth_comparison(
            rgb, pred, gt, args.frame_number, args.output_name
        )
        print(json.dumps({"frame": args.frame_number, "output": str(output)}, indent=2))
        return

    manifest = json.loads((RUN / "manifest.json").read_text())
    common = dict(free_observations_required=1, free_viewpoint_sectors_required=1,
                  occupied_observations_required=1, occupied_support_radius_voxels=0,
                  pending_clear_free_observations_required=3, occupied_evidence_window_frames=6)
    cfg_est = load_config(manifest, 5, 3.0, 0.12, 6.0,
                          obstacle_vertical_band_m=0.1, occupied_uncertainty_m=1.5,
                          vertical_clearance_m=0.1, **common)
    cfg_ref = load_config(manifest, 5, 0.0, obstacle_vertical_band_m=0.2,
                          occupied_uncertainty_m=0.0, vertical_clearance_m=0.2, **common)
    estimated, reference = SpatialNavigator(cfg_est), SpatialNavigator(cfg_ref)
    candidates = []
    events = [json.loads(line) for line in (RUN / "events.jsonl").read_text().splitlines() if line.strip()]
    for event in events:
        if event.get("event") != "frame" or not event.get("file"): continue
        frame_path = RUN / event["file"]
        cache_path = CACHE / f"{frame_path.stem}.npy"
        if not cache_path.exists(): continue
        with np.load(frame_path) as frame:
            rgb = cv2.cvtColor(frame["rgb_bgr"], cv2.COLOR_BGR2RGB)
            gt = frame["reference_depth_m"].astype(float)
            transform = frame["camera_to_ned"].astype(float)
            intrinsics = CameraIntrinsics(*frame["intrinsics"].astype(float))
        pred = np.load(cache_path).astype(float)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0])).astype(float)
        pred[pred > 0] = np.maximum(pred[pred > 0] - 1.2, 0.0)
        valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 0) & (gt > 0)
        score = float(np.mean(np.abs(pred[valid] - gt[valid]))) if valid.any() else -1
        candidates.append((score, frame_path.stem, rgb, pred, gt))
        estimated.integrate_depth(pred, intrinsics, transform)
        reference.integrate_depth(gt, intrinsics, transform)
    _, frame_id, rgb, pred, gt = max(candidates, key=lambda item: item[0])
    primary = save_depth_comparison(
        rgb,
        pred,
        gt,
        int(frame_id.replace("frame_", "")),
        args.output_name,
    )
    ref_occ, est_occ, est_free = reference.grid.occupied_voxels(), estimated.grid.occupied_voxels(), estimated.grid.free_voxels()
    false_free, ref_only = ref_occ & est_free, ref_occ - (ref_occ & est_free)
    def xy(voxels):
        return np.asarray([reference.grid.voxel_to_world(v)[:2] for v in voxels]) if voxels else np.empty((0, 2))
    a, b, c = xy(ref_only), xy(est_occ), xy(false_free)
    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    if len(a): ax.scatter(a[:, 1], a[:, 0], s=5, c="#6b7280", alpha=.35, label="Ocupado na referencia")
    if len(b): ax.scatter(b[:, 1], b[:, 0], s=5, c="#2563eb", alpha=.25, label="Ocupado na v19")
    if len(c): ax.scatter(c[:, 1], c[:, 0], s=15, c="#dc2626", alpha=.85, label="Falso espaco livre")
    rate = len(false_free) / len(ref_occ) if ref_occ else float("nan")
    ax.set(title=f"Falso espaco livre no mapa v19 ({rate:.2%})", xlabel="Leste (m)", ylabel="Norte (m)")
    ax.axis("equal"); ax.grid(alpha=.2); ax.legend(loc="best")
    primary2 = OUTPUTS[0] / "resultados_falso_espaco_livre_v19.png"
    fig.savefig(primary2, dpi=180); plt.close(fig)
    for directory in OUTPUTS[1:]:
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary2, directory / primary2.name)
    print(json.dumps({"frame": frame_id, "frames": len(candidates), "reference_occupied": len(ref_occ), "false_free": len(false_free), "false_free_rate": rate}, indent=2))

if __name__ == "__main__": main()
