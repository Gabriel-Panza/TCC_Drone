"""Persistencia dos frames, mapas e planos da validacao espacial."""

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import threading

import numpy as np


class SpatialRunRecorder:
    """Grava um diretorio autocontido para cada execucao do novo pipeline."""

    def __init__(self, base_dir, metadata=None, save_frames=True):
        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(base_dir).expanduser() / timestamp
        self.frames_dir = self.run_dir / "frames"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        if save_frames:
            self.frames_dir.mkdir()
        self.save_frames = save_frames
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.metadata = dict(metadata or {})
        self.frames_saved = 0
        self.plans_saved = 0
        self._lock = threading.Lock()
        self._write_manifest(complete=False)

    def record_frame(
        self,
        *,
        timestamp_s,
        rgb_bgr,
        estimated_depth_m,
        reference_depth_m,
        camera_to_ned,
        intrinsics,
        source,
        map_stats,
        depth_metrics=None,
    ):
        """Registra metadados e, opcionalmente, arrays sincronizados do frame."""

        with self._lock:
            self.frames_saved += 1
            frame_id = self.frames_saved
            frame_file = None
            if self.save_frames:
                frame_file = f"frames/frame_{frame_id:06d}.npz"
                rgb = np.asarray(rgb_bgr, dtype=np.uint8)
                estimated = self._optional_depth(estimated_depth_m)
                reference = self._optional_depth(reference_depth_m)
                np.savez_compressed(
                    self.run_dir / frame_file,
                    rgb_bgr=rgb,
                    estimated_depth_m=estimated,
                    reference_depth_m=reference,
                    camera_to_ned=np.asarray(camera_to_ned, dtype=np.float64),
                    intrinsics=np.asarray(
                        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy],
                        dtype=np.float64,
                    ),
                )
            self._append_event(
                {
                    "event": "frame",
                    "frame_id": frame_id,
                    "timestamp_s": float(timestamp_s),
                    "source": source,
                    "file": frame_file,
                    "map": map_stats,
                    "depth_metrics": depth_metrics,
                }
            )
            if self.frames_saved % 20 == 0:
                self._write_manifest(complete=False)

    def record_plan(self, timestamp_s, plan, map_kind):
        with self._lock:
            self.plans_saved += 1
            payload = asdict(plan) if is_dataclass(plan) else dict(plan)
            self._append_event(
                {
                    "event": "plan",
                    "plan_id": self.plans_saved,
                    "timestamp_s": float(timestamp_s),
                    "map_kind": map_kind,
                    "plan": payload,
                }
            )

    def record_state(self, timestamp_s, state, **details):
        """Registra marcos da missao usados para separar as fases da trajetoria."""

        with self._lock:
            self._append_event(
                {
                    "event": "mission_state",
                    "timestamp_s": float(timestamp_s),
                    "state": str(state),
                    **details,
                }
            )

    def save_map(self, name, navigator):
        np.savez_compressed(self.run_dir / f"{name}.npz", **navigator.export_map())

    def close(self, navigators=None, summary=None):
        for name, navigator in (navigators or {}).items():
            self.save_map(name, navigator)
        if summary:
            self._append_event({"event": "summary", **summary})
        self._write_manifest(complete=True)

    def _write_manifest(self, complete):
        manifest = {
            "schema_version": "spatial_mapping_v1",
            "complete": bool(complete),
            "frames_saved": self.frames_saved,
            "plans_saved": self.plans_saved,
            "metadata": self.metadata,
            "files": {
                "events": self.events_path.name,
                "estimated_map": "estimated_map.npz",
                "reference_map": "reference_map.npz",
            },
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=self._json_default),
            encoding="utf-8",
        )

    def _append_event(self, payload):
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, ensure_ascii=False, default=self._json_default) + "\n"
            )

    @staticmethod
    def _optional_depth(depth):
        if depth is None:
            return np.empty((0, 0), dtype=np.float16)
        return np.asarray(depth, dtype=np.float16)

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"tipo nao serializavel: {type(value).__name__}")
