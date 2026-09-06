"""Snapshots completos do mapa antes do A*: independentes de imagens reduzidas.

Nao substituem frames para avaliar profundidade. Preservam inclusive evidencia
temporal pendente, para reproduzir o planejamento sem inventar espaco livre.
"""

from dataclasses import asdict
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from .navigation import SpatialNavigationConfig, SpatialNavigator
from .occupancy import OccupancyGrid3D, OccupancyGridConfig


SCHEMA = "spatial_planning_snapshot_v1"


def capture_planning_snapshot(navigator):
    """Copia o estado; o chamador deve manter o lock do mapa durante a copia."""
    grid = navigator.grid
    items = list(grid._log_odds.items())
    return {
        "schema": np.asarray(SCHEMA),
        "navigation_config": np.asarray(json.dumps(asdict(navigator.config))),
        "occupancy_config": np.asarray(json.dumps(asdict(grid.config))),
        "frames_integrated": np.asarray(navigator.frames_integrated),
        "integration_frame": np.asarray(grid._integration_frame),
        "voxels": np.asarray([v for v, _ in items], dtype=np.int64).reshape(-1, 3),
        "log_odds": np.asarray([value for _, value in items], dtype=np.float64),
        "confirmed_occupied": np.asarray(
            list(grid._confirmed_occupied), dtype=np.int64
        ).reshape(-1, 3),
        "evidence_voxels": np.asarray(
            list(grid._occupied_evidence_frames), dtype=np.int64
        ).reshape(-1, 3),
        "evidence_frames": np.asarray([
            (*voxel, frame)
            for voxel, frames in grid._occupied_evidence_frames.items()
            for frame in frames
        ], dtype=np.int64).reshape(-1, 4),
        "pending_free": np.asarray([
            (*voxel, count)
            for voxel, count in grid._pending_free_observations.items()
        ], dtype=np.int64).reshape(-1, 4),
    }


def restore_planning_snapshot(data):
    """Restaura o estado completo, sem reinterpretar thresholds ou pendencias."""
    if str(data["schema"].item()) != SCHEMA:
        raise ValueError("schema de snapshot de planejamento desconhecido")
    nav_config = SpatialNavigationConfig(**json.loads(data["navigation_config"].item()))
    grid_config = OccupancyGridConfig(**json.loads(data["occupancy_config"].item()))
    if nav_config.voxel_resolution_m != grid_config.resolution_m:
        raise ValueError("resolucao inconsistente no snapshot")
    arrays = {}
    for key, width in (
        ("voxels", 3), ("confirmed_occupied", 3), ("evidence_voxels", 3),
        ("evidence_frames", 4), ("pending_free", 4),
    ):
        value = np.asarray(data[key])
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"formato invalido: {key}")
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"coordenadas nao inteiras: {key}")
        arrays[key] = value
    log_odds = np.asarray(data["log_odds"], dtype=np.float64)
    if log_odds.shape != (len(arrays["voxels"]),) or not np.isfinite(log_odds).all():
        raise ValueError("log_odds invalido no snapshot")
    navigator = SpatialNavigator(nav_config)
    grid = OccupancyGrid3D(grid_config)
    grid._log_odds = dict(zip(map(tuple, arrays["voxels"]), map(float, log_odds)))
    grid._confirmed_occupied = set(map(tuple, arrays["confirmed_occupied"]))
    grid._occupied_evidence_frames = {
        tuple(v): set() for v in arrays["evidence_voxels"]
    }
    for x, y, z, frame in arrays["evidence_frames"]:
        grid._occupied_evidence_frames[(x, y, z)].add(int(frame))
    grid._pending_free_observations = {
        (x, y, z): int(count) for x, y, z, count in arrays["pending_free"]
    }
    grid._integration_frame = int(data["integration_frame"].item())
    navigator.frames_integrated = int(data["frames_integrated"].item())
    navigator.grid = grid
    return navigator


def write_planning_snapshot(run_dir, snapshot):
    """Persiste fora do lock do mapa; nunca sobrescreve snapshots anteriores."""
    relative = Path("planning_maps") / f"map_{uuid4().hex}.npz"
    path = Path(run_dir) / relative
    path.parent.mkdir(exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **snapshot)
    return relative.as_posix()


def load_planning_snapshot(run_dir, relative_path):
    """Carrega um snapshot pertencente a run e rejeita escape de diretorio."""
    root = Path(run_dir).resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("snapshot deve pertencer ao diretorio da run")
    with np.load(path, allow_pickle=False) as data:
        return restore_planning_snapshot(data)
