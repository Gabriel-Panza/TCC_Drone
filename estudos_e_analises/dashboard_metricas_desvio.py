"""Dashboard interativo para inspecao das metricas de voo, depth e modelos."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"
NOTEBOOK_PATH = ANALYSIS_DIR / "estudo_das_metricas.ipynb"
MLP_FIXED_RESULTS_PATH = ANALYSIS_DIR / "comparacao_mlp_splits_fixos.csv"
EVENT_FIXED_RESULTS_PATH = ANALYSIS_DIR / "comparacao_eventos_splits_fixos.csv"
ANGULAR_COMPARISON_PATH = ANALYSIS_DIR / "comparacao_representacao_angular_splits_fixos.csv"
LOSS_CURVES_PATH = ANALYSIS_DIR / "curvas_loss_mlp.csv"
GRADIENT_IMPORTANCE_PATH = ANALYSIS_DIR / "importancia_gradiente_erro_validacao.csv"
GROUPED_GRADIENT_IMPORTANCE_PATH = ANALYSIS_DIR / "importancia_gradiente_erro_validacao_agrupada.csv"
LARGEST_GRADIENT_ERRORS_PATH = ANALYSIS_DIR / "maiores_erros_gradiente_validacao.csv"


def load_validated_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Carrega um CSV somente quando ele existe e contem o esquema esperado."""

    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if required_columns.issubset(df.columns) else pd.DataFrame()


def has_fixed_comparison_results() -> bool:
    return MLP_FIXED_RESULTS_PATH.exists() and EVENT_FIXED_RESULTS_PATH.exists()


def find_depth_ground_truth_dir() -> Path:
    """Localiza o dataset de depth nos layouts suportados pelo projeto."""

    candidates = [
        PROJECT_ROOT / "datasets" / "depth_ground_truth",
        PROJECT_ROOT.parent / "datasets" / "depth_ground_truth",
        ANALYSIS_DIR / "datasets" / "depth_ground_truth",
        Path.cwd() / "datasets" / "depth_ground_truth",
        Path.cwd().parent / "datasets" / "depth_ground_truth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "datasets" / "depth_ground_truth"


DEPTH_GT_DIR = find_depth_ground_truth_dir()

# Parametros espelhados de drone_controller.py. Eles representam a logica de
# navegacao/reatividade usada na missao atual, sem importar o modulo ROS.
WAYPOINTS_RELATIVOS = np.array(
    [
        [-25.0, 25.0, -1.75],
        [-50.0, 70.0, -1.75],
        [-25.0, 25.0, -1.75],
        [0.0, 0.0, -1.75],
    ],
    dtype=float,
)
VELOCIDADE_MAXIMA = 12.0
RAIO_DE_ACEITACAO = 5.0
RAIO_DESATIVA_EVASAO_FINAL = 10.0
OBSTACLE_RISK_THRESHOLD = 0.07

REFERENCE_RUN_ORDER = (
    "run_20260521_174507",
    "run_20260521_174545",
    "run_20260521_175244",
    "run_20260624_211302",
    "run_20260624_224839",
    "run_20260710_192442",
    "run_20260710_193015",
    "run_20260715_152840",
    "run_20260715_152924",
    "run_20260715_161701",
    "run_20260715_161906",
    "run_20260715_162101",
    "run_20260715_163428",
    "run_20260715_163825",
    "run_20260715_174528",
    "run_20260715_183633",
    "run_20260715_215803",
    "run_20260715_220110",
    "run_20260715_220346",
    "run_20260715_220437",
    "run_20260716_221426",
    "run_20260716_221504",
    "run_20260716_221545",
    "run_20260716_231519",
    "run_20260716_231600",
)
BATCH_SPECS = (
    (9, "Primeiras 9 runs"),
    (17, "Primeiras 17 runs"),
    (25, "Primeiras 25 runs"),
    (33, "Primeiras 33 runs"),
    (40, "Todas (40 runs)"),
)
BATCH_LABELS = dict(BATCH_SPECS)
# Mantem as referencias embutidas antigas importaveis quando os CSVs nao existem.
BATCH_LABELS.update({7: "Primeiras 7 runs", 16: "Primeiras 16 runs"})
BATCH_LABEL_ORDER = [label for _size, label in BATCH_SPECS]
RUN_ORDER_INDEX = {run_id: index for index, run_id in enumerate(REFERENCE_RUN_ORDER)}
BLOCK_LABELS = tuple(
    f"Runs {1 if index == 0 else BATCH_SPECS[index - 1][0] + 1}-{size}"
    for index, (size, _label) in enumerate(BATCH_SPECS)
)

REFERENCE_BATCH_STATS = [
    {
        "leva": BATCH_LABELS[7], "runs": 7, "intervalos": 118,
        "dt_s_mean": 0.092881, "depth_age_s_mean": 0.041831,
        "flow_valid_points_mean": 80.813559, "flow_track_retention_pct_mean": 96.534164,
        "flow_mag_p90_px_mean": 43.700533, "radial_flow_p90_px_mean": 35.155069,
        "delta_depth_close_5m_pp_mean": np.nan, "delta_depth_close_5m_pp_std": 3.249952,
        "delta_abs_p90": 4.026670, "zero_rate_pct": 5.084746,
        "eventos": 79, "event_rate_pct": 66.949153,
    },
    {
        "leva": BATCH_LABELS[16], "runs": 16, "intervalos": 410,
        "dt_s_mean": 0.092029, "depth_age_s_mean": 0.042302,
        "flow_valid_points_mean": 80.814634, "flow_track_retention_pct_mean": 96.747574,
        "flow_mag_p90_px_mean": 40.780258, "radial_flow_p90_px_mean": 33.360588,
        "delta_depth_close_5m_pp_mean": 0.021713, "delta_depth_close_5m_pp_std": 2.735755,
        "delta_abs_p90": 3.889082, "zero_rate_pct": 3.658537,
        "eventos": 281, "event_rate_pct": 68.536585,
    },
    {
        "leva": BATCH_LABELS[25], "runs": 25, "intervalos": 720,
        "dt_s_mean": 0.090783, "depth_age_s_mean": 0.043322,
        "flow_valid_points_mean": 80.040278, "flow_track_retention_pct_mean": 96.892998,
        "flow_mag_p90_px_mean": 39.500362, "radial_flow_p90_px_mean": 32.902471,
        "delta_depth_close_5m_pp_mean": -0.020764, "delta_depth_close_5m_pp_std": 2.610670,
        "delta_abs_p90": 3.832938, "zero_rate_pct": 2.638889,
        "eventos": 506, "event_rate_pct": 70.277778,
    },
]

REFERENCE_MLP_TEST = [
    (BATCH_LABELS[16], "MLP", "delta_depth_close_10m_pp", 1.854696, 2.690347),
    (BATCH_LABELS[16], "Media treino", "delta_depth_close_10m_pp", 0.841406, 1.225439),
    (BATCH_LABELS[16], "MLP", "delta_depth_close_2m_pp", 2.467895, 3.868087),
    (BATCH_LABELS[16], "Media treino", "delta_depth_close_2m_pp", 1.251268, 1.910592),
    (BATCH_LABELS[16], "MLP", "delta_depth_close_5m_pp", 3.175409, 4.729990),
    (BATCH_LABELS[16], "Media treino", "delta_depth_close_5m_pp", 1.751631, 2.411158),
    (BATCH_LABELS[16], "MLP", "delta_depth_p10_m", 0.149927, 0.257732),
    (BATCH_LABELS[16], "Media treino", "delta_depth_p10_m", 0.082042, 0.146651),
    (BATCH_LABELS[16], "MLP", "delta_depth_p50_m", 0.424316, 0.605935),
    (BATCH_LABELS[16], "Media treino", "delta_depth_p50_m", 0.249817, 0.349569),
    (BATCH_LABELS[16], "MLP", "delta_depth_p90_m", 1.746266, 2.661626),
    (BATCH_LABELS[16], "Media treino", "delta_depth_p90_m", 1.115172, 1.937910),
    (BATCH_LABELS[25], "MLP", "delta_depth_close_10m_pp", 2.041496, 2.906099),
    (BATCH_LABELS[25], "Media treino", "delta_depth_close_10m_pp", 0.952055, 1.406515),
    (BATCH_LABELS[25], "MLP", "delta_depth_close_2m_pp", 2.859533, 4.801528),
    (BATCH_LABELS[25], "Media treino", "delta_depth_close_2m_pp", 1.416045, 2.475855),
    (BATCH_LABELS[25], "MLP", "delta_depth_close_5m_pp", 3.736108, 5.168131),
    (BATCH_LABELS[25], "Media treino", "delta_depth_close_5m_pp", 1.710887, 2.365598),
    (BATCH_LABELS[25], "MLP", "delta_depth_p10_m", 0.220102, 0.467165),
    (BATCH_LABELS[25], "Media treino", "delta_depth_p10_m", 0.080154, 0.136932),
    (BATCH_LABELS[25], "MLP", "delta_depth_p50_m", 0.443824, 0.642170),
    (BATCH_LABELS[25], "Media treino", "delta_depth_p50_m", 0.235744, 0.355026),
    (BATCH_LABELS[25], "MLP", "delta_depth_p90_m", 1.723759, 3.344502),
    (BATCH_LABELS[25], "Media treino", "delta_depth_p90_m", 1.200840, 2.994017),
]

REFERENCE_EVENT_RESULTS = [
    ("16 runs - classificador", "balanced accuracy", 0.622925),
    ("16 runs - classificador", "precisao", 0.770492),
    ("16 runs - classificador", "recall", 0.854545),
    ("16 runs - classificador", "F1", 0.810345),
    ("16 runs - classificador", "average precision", 0.853243),
    ("16 runs - classificador", "falsos positivos", 14.0),
    ("16 runs - classificador", "falsos negativos", 8.0),
    ("16 runs - pos-gate", "MAE zero delta", 1.731648),
    ("16 runs - pos-gate", "MAE MLP evento + MLP delta", 2.419790),
    ("16 runs - pos-gate", "MAE evento real + MLP delta (diagnostico)", 2.051965),
    ("16 runs - pos-gate", "MAE MLP evento + mediana", 1.824273),
    ("25 runs - classificador", "balanced accuracy", 0.696233),
    ("25 runs - classificador", "precisao", 0.833333),
    ("25 runs - classificador", "recall", 0.797872),
    ("25 runs - classificador", "F1", 0.815217),
    ("25 runs - classificador", "average precision", 0.895586),
    ("25 runs - classificador", "falsos positivos", 15.0),
    ("25 runs - classificador", "falsos negativos", 19.0),
    ("25 runs - pos-gate", "MAE zero delta", 1.706795),
    ("25 runs - pos-gate", "MAE MLP evento + MLP delta", 2.757244),
    ("25 runs - pos-gate", "MAE evento real + MLP delta (diagnostico)", 2.762598),
    ("25 runs - pos-gate", "MAE MLP evento + mediana", 1.771411),
]

REFERENCE_RUN_EVENT_STATS = [
    ("run_20260521_174507", 16, 13, 0.812500),
    ("run_20260521_174545", 18, 14, 0.777778),
    ("run_20260521_175244", 27, 11, 0.407407),
    ("run_20260624_211302", 11, 7, 0.636364),
    ("run_20260624_224839", 14, 12, 0.857143),
    ("run_20260710_192442", 19, 14, 0.736842),
    ("run_20260710_193015", 13, 8, 0.615385),
    ("run_20260715_152840", 13, 11, 0.846154),
    ("run_20260715_152924", 17, 12, 0.705882),
    ("run_20260715_161701", 31, 25, 0.806452),
    ("run_20260715_161906", 48, 28, 0.583333),
    ("run_20260715_162101", 38, 23, 0.605263),
    ("run_20260715_163428", 33, 22, 0.666667),
    ("run_20260715_163825", 43, 29, 0.674419),
    ("run_20260715_174528", 27, 22, 0.814815),
    ("run_20260715_183633", 42, 30, 0.714286),
    ("run_20260715_215803", 46, 38, 0.826087),
    ("run_20260715_220110", 36, 24, 0.666667),
    ("run_20260715_220346", 30, 20, 0.666667),
    ("run_20260715_220437", 32, 21, 0.656250),
    ("run_20260716_221426", 32, 26, 0.812500),
    ("run_20260716_221504", 34, 18, 0.529412),
    ("run_20260716_221545", 39, 27, 0.692308),
    ("run_20260716_231519", 32, 28, 0.875000),
    ("run_20260716_231600", 29, 23, 0.793103),
]


COLORS = {
    "ink": "#172026",
    "muted": "#62717b",
    "line": "#d5dde3",
    "panel": "#ffffff",
    "page": "#f4f7f9",
    "route": "#1b8a5a",
    "drone": "#246bfe",
    "risk": "#c2410c",
    "accent": "#0f766e",
    "latest": "#7c3aed",
}
GRAPH_CONFIG = {"displaylogo": False}
BATCH_COLORS = {
    label: color
    for (_size, label), color in zip(
        BATCH_SPECS,
        (COLORS["drone"], "#0891b2", COLORS["accent"], "#a16207", COLORS["latest"]),
    )
}
BLOCK_COLORS = dict(zip(BLOCK_LABELS, BATCH_COLORS.values()))
EMPTY_FLIGHT_COLUMNS = (
    "timestamp",
    "x",
    "y",
    "z",
    "roll_speed",
    "pitch_speed",
    "yaw_speed",
)


def _empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_FLIGHT_COLUMNS)


def run_datetime(path: Path) -> datetime | None:
    """Extrai o instante de coleta do nome de uma run de log ou depth."""

    name = path.parent.name if path.name == "manifest.json" else path.name
    token = name.replace("voo_teste_", "").replace("run_", "").replace(".csv", "")
    try:
        return datetime.strptime(token, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _raw_log_manifest_files() -> list[Path]:
    return [
        path
        for path in LOG_DIR.glob("voo_teste_*/manifest.json")
        if "old" not in path.relative_to(LOG_DIR).parts
    ]


def _raw_depth_source_files() -> list[Path]:
    manifests = [
        path
        for path in DEPTH_GT_DIR.glob("run_*/manifest.json")
        if "old" not in path.relative_to(DEPTH_GT_DIR).parts
    ]
    if manifests:
        return manifests
    return [
        path
        for path in DEPTH_GT_DIR.glob("run_*/metadata.csv")
        if "old" not in path.relative_to(DEPTH_GT_DIR).parts
    ]


def paired_log_depth_files(max_delta_s: float = 120.0) -> list[tuple[Path, Path]]:
    """Pareia runs de voo e depth pelo timestamp mais proximo, sem reutilizar datasets."""

    logs = sorted(_raw_log_manifest_files(), key=lambda path: run_datetime(path) or datetime.min)
    depths = sorted(_raw_depth_source_files(), key=lambda path: run_datetime(path) or datetime.min)
    used_depths: set[Path] = set()
    pairs: list[tuple[Path, Path]] = []

    for log_path in logs:
        log_dt = run_datetime(log_path)
        if log_dt is None:
            continue
        candidates = []
        for depth_path in depths:
            if depth_path in used_depths:
                continue
            depth_dt = run_datetime(depth_path)
            if depth_dt is None:
                continue
            delta_s = abs((depth_dt - log_dt).total_seconds())
            if delta_s <= max_delta_s:
                candidates.append((delta_s, depth_path))
        if not candidates:
            continue
        _, selected_depth = min(candidates, key=lambda item: item[0])
        used_depths.add(selected_depth)
        pairs.append((log_path, selected_depth))

    return pairs


def list_log_files() -> list[Path]:
    """Lista os logs atuais, priorizando runs pareadas e ignorando ``logs/old``."""

    paired_logs = [log_path for log_path, _depth_path in paired_log_depth_files()]
    if paired_logs:
        return sorted(paired_logs, key=lambda path: run_datetime(path) or datetime.min, reverse=True)
    manifests = _raw_log_manifest_files()
    if manifests:
        return sorted(manifests, key=lambda path: run_datetime(path) or datetime.min, reverse=True)
    return sorted(LOG_DIR.glob("voo_teste_*.csv"), reverse=True)


def list_depth_metadata_files() -> list[Path]:
    """Lista datasets atuais de depth, com ``metadata.csv`` como fallback legado."""

    paired_depths = [depth_path for _log_path, depth_path in paired_log_depth_files()]
    if paired_depths:
        return sorted(paired_depths, key=lambda path: run_datetime(path) or datetime.min, reverse=True)
    sources = _raw_depth_source_files()
    return sorted(sources, key=lambda path: run_datetime(path) or datetime.min, reverse=True)


def list_depth_interval_runs() -> list[Path]:
    return sorted(
        path.parent
        for path in DEPTH_GT_DIR.glob("run_*/manifest.json")
        if "old" not in path.relative_to(DEPTH_GT_DIR).parts
    )


def filter_synchronized_intervals(df: pd.DataFrame, require_depth_dt: bool = True) -> pd.DataFrame:
    """Mantem somente intervalos com timestamps RGB/depth coerentes."""

    if df.empty or "dt_s" not in df or "depth_age_s" not in df:
        return pd.DataFrame() if require_depth_dt else df.copy()
    if require_depth_dt and "depth_dt_s" not in df:
        return pd.DataFrame()

    dt_rgb = pd.to_numeric(df["dt_s"], errors="coerce")
    depth_age = pd.to_numeric(df["depth_age_s"], errors="coerce")
    mask = dt_rgb.gt(0.0) & dt_rgb.le(0.5) & depth_age.le(0.08)
    if "depth_dt_s" in df:
        dt_depth = pd.to_numeric(df["depth_dt_s"], errors="coerce")
        mask &= dt_depth.gt(0.0) & dt_depth.le(0.5)
    return df.loc[mask].reset_index(drop=True)


def load_depth_interval_run(run_dir: Path) -> pd.DataFrame:
    """Carrega os intervalos de uma run depth no formato memmap atual."""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return pd.DataFrame()

    with open(manifest_path, encoding="utf-8") as fp:
        manifest = json.load(fp)

    if manifest.get("schema_version") != "depth_interval_memmap_v1":
        return pd.DataFrame()

    intervals_name = manifest.get("arrays", {}).get("intervals")
    if not intervals_name:
        return pd.DataFrame()

    intervals_path = run_dir / intervals_name
    if not intervals_path.exists():
        return pd.DataFrame()

    n = int(manifest.get("num_samples", 0))
    interval_array = np.load(intervals_path, mmap_mode="r")
    df = pd.DataFrame.from_records(interval_array[:n]).copy()
    if df.empty:
        return pd.DataFrame()

    df = filter_synchronized_intervals(df)
    if df.empty:
        return pd.DataFrame()
    df["run_id"] = run_dir.name
    df["ordem_intervalo"] = np.arange(len(df))
    return df


def enrich_depth_interval_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona campos derivados usados pelas visualizacoes de depth e IMU."""

    if df.empty:
        return df

    out = df.copy()
    for col in out.columns:
        if col != "run_id":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["tempo_s"] = out["dt_s"].fillna(0.0).cumsum() if "dt_s" in out else np.arange(len(out), dtype=float)
    out["sample_id"] = out["sample_id"].astype(str).str.zfill(6) if "sample_id" in out else out.index.astype(str)
    gyro_cols = ["delta_gyro_x_rad_s", "delta_gyro_y_rad_s", "delta_gyro_z_rad_s"]
    integral_cols = ["gyro_x_integral_rad", "gyro_y_integral_rad", "gyro_z_integral_rad"]
    accel_cols = ["delta_accel_x_m_s2", "delta_accel_y_m_s2", "delta_accel_z_m_s2"]
    if all(col in out for col in gyro_cols):
        out["gyro_norm"] = np.sqrt(sum(out[col].fillna(0.0) ** 2 for col in gyro_cols))
    elif all(col in out for col in integral_cols):
        out["gyro_norm"] = np.sqrt(sum(out[col].fillna(0.0) ** 2 for col in integral_cols))
    else:
        out["gyro_norm"] = 0.0
    if all(col in out for col in accel_cols):
        out["accel_norm"] = np.sqrt(sum(out[col].fillna(0.0) ** 2 for col in accel_cols))
    else:
        out["accel_norm"] = 0.0
    if "pan_comp_delta_rad" not in out:
        out["pan_comp_delta_rad"] = np.nan
    out["source_format"] = "depth_interval_memmap"
    return out


def is_depth_interval_memmap(df: pd.DataFrame) -> bool:
    return (
        not df.empty
        and "source_format" in df.columns
        and str(df["source_format"].iloc[0]) == "depth_interval_memmap"
    )


def _decode_plotly_array(value) -> np.ndarray:
    """Decodifica arrays binarios embutidos pelo Plotly no notebook."""

    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, dict) or "bdata" not in value:
        return np.asarray(value)

    array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
    shape = value.get("shape")
    if shape:
        array = array.reshape(tuple(int(part.strip()) for part in str(shape).split(",")))
    return array.copy()


@lru_cache(maxsize=2)
def _load_notebook_interval_snapshot(_mtime_ns: int) -> pd.DataFrame:
    """Recupera as metricas exibidas no ultimo output do notebook.

    Os memmaps continuam sendo a fonte principal. Este snapshot permite que o dashboard
    represente runs cujo output foi salvo no notebook, mas cujas pastas locais nao estao
    presentes no ambiente que abriu o dashboard.
    """

    try:
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()

    scatter_figure = None
    sequence_figure = None
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            figure = output.get("data", {}).get("application/vnd.plotly.v1+json")
            if not figure:
                continue
            title = figure.get("layout", {}).get("title", {})
            title_text = title.get("text", "") if isinstance(title, dict) else str(title)
            if title_text == "Flow radial do intervalo x variacao de proximidade no depth":
                scatter_figure = figure
            elif title_text == "Sequencia dos intervalos: flow visual e delta de proximidade":
                sequence_figure = figure

    if scatter_figure is None:
        return pd.DataFrame()

    flow_values = np.array([], dtype=float)
    if sequence_figure is not None:
        flow_trace = next(
            (trace for trace in sequence_figure.get("data", []) if trace.get("name") == "P90 flow"),
            None,
        )
        if flow_trace is not None:
            flow_values = _decode_plotly_array(flow_trace.get("y", [])).astype(float)

    frames = []
    offset = 0
    for trace in scatter_figure.get("data", []):
        run_id = str(trace.get("name", ""))
        if not run_id.startswith("run_"):
            continue
        radial_flow = _decode_plotly_array(trace.get("x", [])).astype(float)
        delta_close_5m = _decode_plotly_array(trace.get("y", [])).astype(float)
        customdata = _decode_plotly_array(trace.get("customdata", []))
        marker_size = _decode_plotly_array(trace.get("marker", {}).get("size", [])).astype(float)
        n = len(delta_close_5m)
        if n == 0 or customdata.ndim != 2 or customdata.shape[0] != n:
            continue
        flow_slice = flow_values[offset : offset + n]
        if len(flow_slice) != n:
            flow_slice = np.full(n, np.nan)
        offset += n
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "sample_id": customdata[:, 0],
                    "dt_s": customdata[:, 1],
                    "depth_age_s": customdata[:, 2],
                    "pan_comp_delta_rad": customdata[:, 3],
                    "flow_valid_points": marker_size,
                    "flow_mag_p90_px": flow_slice,
                    "radial_flow_p90_px": radial_flow,
                    "delta_depth_close_5m_pp": delta_close_5m,
                    "source_format": "notebook_snapshot",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_notebook_interval_snapshot() -> pd.DataFrame:
    """Recupera os intervalos embutidos nos outputs salvos do notebook."""

    if not NOTEBOOK_PATH.exists():
        return pd.DataFrame()
    snapshot = _load_notebook_interval_snapshot(NOTEBOOK_PATH.stat().st_mtime_ns).copy()
    return filter_synchronized_intervals(snapshot, require_depth_dt=False)


def load_all_depth_intervals() -> pd.DataFrame:
    """Combina memmaps locais com runs ausentes recuperadas do notebook."""

    frames = []
    for run_dir in list_depth_interval_runs():
        try:
            df = load_depth_interval_run(run_dir)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)

    local_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    notebook_df = load_notebook_interval_snapshot()
    if notebook_df.empty:
        return local_df
    if local_df.empty:
        return notebook_df

    local_runs = set(local_df["run_id"].astype(str))
    missing_runs = notebook_df[~notebook_df["run_id"].isin(local_runs)]
    return pd.concat([local_df, missing_runs], ignore_index=True, sort=False)


def run_block_label(run_id: str) -> str:
    index = RUN_ORDER_INDEX.get(run_id)
    if index is None:
        return "Runs adicionais"
    for block_index, (size, _label) in enumerate(BATCH_SPECS):
        if index < size:
            return BLOCK_LABELS[block_index]
    return "Runs adicionais"


def batch_label(size: int) -> str:
    return BATCH_LABELS.get(size, f"Primeiras {size} runs")


def ordered_run_ids(df: pd.DataFrame) -> list[str]:
    """Ordena primeiro as 25 runs de referencia e depois qualquer run adicional."""

    available = set(df["run_id"].astype(str))
    reference = [run_id for run_id in REFERENCE_RUN_ORDER if run_id in available]
    return reference + sorted(available - set(reference))


def run_label(path: Path) -> str:
    if path.name == "manifest.json":
        return path.parent.name.replace("voo_teste_", "")
    if path.is_dir():
        return path.name.replace("voo_teste_", "")
    return path.stem.replace("voo_teste_", "")


def depth_run_label(path: Path) -> str:
    """Retorna o timestamp identificador de uma run depth."""
    return path.parent.name.replace("run_", "")


def matching_depth_metadata(log_path: Path, depth_paths: list[Path]) -> Path | None:
    """Encontra o dataset depth pareado; usa ordem de coleta como ultimo fallback."""

    if not depth_paths:
        return None

    for paired_log, paired_depth in paired_log_depth_files():
        if str(paired_log.resolve()) == str(log_path.resolve()):
            return paired_depth

    log_token = run_label(log_path)
    for depth_path in depth_paths:
        if log_token and log_token in depth_path.parent.name:
            return depth_path

    log_paths = list_log_files()
    normalized_logs = [str(path.resolve()) for path in log_paths if path.exists()]
    try:
        log_index = normalized_logs.index(str(log_path.resolve()))
    except ValueError:
        return depth_paths[0]

    if log_index < len(depth_paths):
        return depth_paths[log_index]
    return depth_paths[-1]


def resolve_depth_file(depth_path_value: str, run_dir: Path) -> Path:
    """Resolve paths NPY salvos no Windows ou no WSL para a pasta local da run."""

    original = Path(str(depth_path_value))
    if original.exists():
        return original
    return run_dir / "depth_m" / original.name


@lru_cache(maxsize=8)
def load_depth_metadata(metadata_path_value: str) -> pd.DataFrame:
    """Carrega uma run depth e calcula percentis e ocupacao por faixa de distancia."""

    metadata_path = Path(metadata_path_value)
    if not metadata_path.exists():
        return pd.DataFrame()

    if metadata_path.name == "manifest.json":
        return enrich_depth_interval_dashboard(load_depth_interval_run(metadata_path.parent))

    df = pd.read_csv(metadata_path).copy()
    if df.empty:
        return pd.DataFrame()

    numeric_cols = [
        "rgb_timestamp_s",
        "depth_timestamp_s",
        "depth_age_s",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "accel_x",
        "accel_y",
        "accel_z",
        "depth_min_m",
        "depth_mean_m",
        "depth_max_m",
        "pan_comp_delta_rad",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "pan_comp_delta_rad" not in df.columns:
        df["pan_comp_delta_rad"] = np.nan
    if "pan_comp_source" not in df.columns:
        df["pan_comp_source"] = "nao coletado"

    df = df.sort_values("rgb_timestamp_s").reset_index(drop=True)
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].astype(str).str.zfill(6)
    df["tempo_s"] = df["rgb_timestamp_s"] - df["rgb_timestamp_s"].iloc[0]
    df["gyro_norm"] = np.sqrt(df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2)
    df["accel_norm"] = np.sqrt(df["accel_x"] ** 2 + df["accel_y"] ** 2 + df["accel_z"] ** 2)

    run_dir = metadata_path.parent
    depth_p10 = []
    depth_p50 = []
    depth_p90 = []
    valid_px_pct = []
    close_2m_pct = []
    close_5m_pct = []
    close_10m_pct = []

    for _, row in df.iterrows():
        depth_file = resolve_depth_file(row.get("depth_path", ""), run_dir)
        if not depth_file.exists():
            depth_p10.append(np.nan)
            depth_p50.append(np.nan)
            depth_p90.append(np.nan)
            valid_px_pct.append(np.nan)
            close_2m_pct.append(np.nan)
            close_5m_pct.append(np.nan)
            close_10m_pct.append(np.nan)
            continue

        depth = np.load(depth_file, mmap_mode="r")
        valid = np.isfinite(depth) & (depth > 0.0)
        valid_values = np.asarray(depth[valid], dtype=float)
        if valid_values.size == 0:
            depth_p10.append(np.nan)
            depth_p50.append(np.nan)
            depth_p90.append(np.nan)
            valid_px_pct.append(0.0)
            close_2m_pct.append(np.nan)
            close_5m_pct.append(np.nan)
            close_10m_pct.append(np.nan)
            continue

        depth_p10.append(float(np.percentile(valid_values, 10)))
        depth_p50.append(float(np.percentile(valid_values, 50)))
        depth_p90.append(float(np.percentile(valid_values, 90)))
        valid_px_pct.append(float(valid.mean() * 100.0))
        close_2m_pct.append(float((valid_values < 2.0).mean() * 100.0))
        close_5m_pct.append(float((valid_values < 5.0).mean() * 100.0))
        close_10m_pct.append(float((valid_values < 10.0).mean() * 100.0))

    df["depth_p10_m"] = depth_p10
    df["depth_p50_m"] = depth_p50
    df["depth_p90_m"] = depth_p90
    df["valid_px_pct"] = valid_px_pct
    df["depth_close_2m_pct"] = close_2m_pct
    df["depth_close_5m_pct"] = close_5m_pct
    df["depth_close_10m_pct"] = close_10m_pct
    return df


def has_real_avoidance_metrics(df: pd.DataFrame) -> bool:
    """Verifica se a run possui os tres sinais reais de evasao visual."""
    required = {"obstacle_risk", "avoid_lateral_body", "avoid_brake"}
    return required.issubset(set(df.columns))


def add_real_avoidance_metrics(df: pd.DataFrame) -> None:
    """Mapeia sinais reais de evasao para a interface comum dos graficos."""

    for col in ("obstacle_risk", "avoid_lateral_body", "avoid_brake", "evasao_visual_ativa", "pan_comp_delta_rad"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["indice_desvio_reativo"] = df["obstacle_risk"].clip(0.0, 1.0)
    df["evento_desvio_reativo"] = df["indice_desvio_reativo"] > OBSTACLE_RISK_THRESHOLD
    df["lateral_reativo_m_s2"] = df["avoid_lateral_body"]
    df["freio_reativo"] = df["avoid_brake"].clip(0.0, 1.0)
    df["fonte_desvio_reativo"] = "metricas reais"


def load_flight_interval_memmap(manifest_path: Path) -> pd.DataFrame:
    """Carrega uma run memmap e reconstrui sua trajetoria relativa pelos deltas."""

    if not manifest_path.exists():
        return _empty_dataframe()

    with open(manifest_path, encoding="utf-8") as fp:
        manifest = json.load(fp)

    if manifest.get("schema_version") not in {
        "flight_interval_memmap_v1",
        "flight_interval_memmap_v2_spatial",
    }:
        return _empty_dataframe()

    n = int(manifest.get("num_samples", 0))
    intervals_name = manifest.get("arrays", {}).get("flight_intervals")
    if n <= 0 or not intervals_name:
        return _empty_dataframe()

    intervals_path = manifest_path.parent / intervals_name
    if not intervals_path.exists():
        return _empty_dataframe()

    interval_array = np.load(intervals_path, mmap_mode="r")
    df = pd.DataFrame.from_records(interval_array[:n]).copy()
    if df.empty:
        return _empty_dataframe()

    for col in df.columns:
        if col != "pan_comp_source_code":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp_s" in df and df["timestamp_s"].notna().any():
        df["timestamp"] = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    else:
        df["timestamp"] = df["dt_s"].fillna(0.0).cumsum()
    df["tempo_s"] = df["timestamp"]
    if {"x_m", "y_m", "z_m"}.issubset(df.columns):
        for axis in ("x", "y", "z"):
            absolute = df[f"{axis}_m"]
            df[axis] = absolute - absolute.iloc[0]
    else:
        df["x"] = df["delta_x_m"].fillna(0.0).cumsum()
        df["y"] = df["delta_y_m"].fillna(0.0).cumsum()
        df["z"] = df["delta_z_m"].fillna(0.0).cumsum()
    df["altitude"] = -df["z"]
    df["dt"] = df["dt_s"].replace(0, np.nan)

    for axis in ("x", "y", "z"):
        delta_col = f"delta_{axis}_m"
        df[f"v_{axis}"] = df[delta_col] / df["dt"]
    df[["v_x", "v_y", "v_z"]] = df[["v_x", "v_y", "v_z"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["roll_speed"] = df.get("delta_roll_speed_rad_s", 0.0)
    df["pitch_speed"] = df.get("delta_pitch_speed_rad_s", 0.0)
    df["yaw_speed"] = df.get("delta_yaw_speed_rad_s", 0.0)
    add_motion_metrics(df)

    risk_delta = (
        df["delta_obstacle_risk"].fillna(0.0)
        if "delta_obstacle_risk" in df
        else pd.Series(0.0, index=df.index)
    )
    lateral_delta = (
        df["delta_avoid_lateral_body"].fillna(0.0)
        if "delta_avoid_lateral_body" in df
        else pd.Series(0.0, index=df.index)
    )
    brake_delta = (
        df["delta_avoid_brake"].fillna(0.0)
        if "delta_avoid_brake" in df
        else pd.Series(0.0, index=df.index)
    )
    df["indice_desvio_reativo"] = normalize(risk_delta.abs())
    df["evento_desvio_reativo"] = df["indice_desvio_reativo"] > OBSTACLE_RISK_THRESHOLD
    df["lateral_reativo_m_s2"] = lateral_delta
    df["freio_reativo"] = normalize(brake_delta.abs())
    df["fonte_desvio_reativo"] = "deltas memmap"
    df["run_id"] = manifest_path.parent.name
    return df


def load_run(path: Path) -> pd.DataFrame:
    """Carrega uma run de voo atual ou um CSV legado e calcula campos derivados."""

    if path.name == "manifest.json":
        return load_flight_interval_memmap(path)
    if path.is_dir():
        return load_flight_interval_memmap(path / "manifest.json")

    if not path.exists():
        return _empty_dataframe()

    df = pd.read_csv(path).copy()
    if df.empty:
        return _empty_dataframe()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["tempo_s"] = df["timestamp"] - df["timestamp"].iloc[0]
    df["altitude"] = -df["z"]
    df["dt"] = df["tempo_s"].diff().replace(0, np.nan)

    for axis in ("x", "y", "z"):
        df[f"v_{axis}"] = df[axis].diff() / df["dt"]

    df[["v_x", "v_y", "v_z"]] = df[["v_x", "v_y", "v_z"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    add_motion_metrics(df)
    if has_real_avoidance_metrics(df):
        add_real_avoidance_metrics(df)
    else:
        add_reactive_proxy(df)
    return df


def build_route(df: pd.DataFrame) -> np.ndarray:
    start = df.loc[0, ["x", "y", "z"]].to_numpy(dtype=float)
    waypoints = start + WAYPOINTS_RELATIVOS
    return np.vstack([start, waypoints])


def add_route_metrics(df: pd.DataFrame, route: np.ndarray) -> None:
    """Projeta a trajetoria na rota e adiciona desvio, progresso e velocidades."""

    points = df[["x", "y"]].to_numpy(dtype=float)
    segments = route[1:, :2] - route[:-1, :2]
    starts = route[:-1, :2]
    seg_len = np.linalg.norm(segments, axis=1)
    seg_len_safe = np.where(seg_len == 0, 1.0, seg_len)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])

    closest_dist = np.full(len(df), np.inf)
    signed_dist = np.zeros(len(df))
    along_distance = np.zeros(len(df))
    segment_index = np.zeros(len(df), dtype=int)

    for idx, (start, seg, length, length_safe) in enumerate(zip(starts, segments, seg_len, seg_len_safe)):
        rel = points - start
        t = np.clip(np.sum(rel * seg, axis=1) / (length_safe**2), 0.0, 1.0)
        projection = start + t[:, None] * seg
        diff = points - projection
        dist = np.linalg.norm(diff, axis=1)
        sign = np.sign(seg[0] * diff[:, 1] - seg[1] * diff[:, 0])
        sign = np.where(sign == 0, 1.0, sign)

        better = dist < closest_dist
        closest_dist[better] = dist[better]
        signed_dist[better] = dist[better] * sign[better]
        along_distance[better] = cumulative[idx] + t[better] * length
        segment_index[better] = idx

    df["desvio_rota_m"] = closest_dist
    df["desvio_rota_assinado_m"] = signed_dist
    df["progresso_rota_m"] = along_distance
    df["segmento_rota"] = segment_index + 1

    dir_x = np.zeros(len(df))
    dir_y = np.zeros(len(df))
    for idx, seg in enumerate(segments):
        mask = df["segmento_rota"].to_numpy() == idx + 1
        length = seg_len_safe[idx]
        dir_x[mask] = seg[0] / length
        dir_y[mask] = seg[1] / length

    df["vel_along_rota"] = df["v_x"] * dir_x + df["v_y"] * dir_y
    df["vel_lateral_rota"] = -df["v_x"] * dir_y + df["v_y"] * dir_x

    final_xy = route[-1, :2]
    df["distancia_final_xy"] = np.linalg.norm(points - final_xy, axis=1)
    df["evasao_habilitada_modelo"] = df["distancia_final_xy"] > RAIO_DESATIVA_EVASAO_FINAL


def add_motion_metrics(df: pd.DataFrame) -> None:
    """Calcula velocidade, atividade angular e distancia relativa a rota."""

    df["vel_horizontal"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2)
    df["vel_3d"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2 + df["v_z"] ** 2)
    df["vel_horizontal_suave"] = (
        df["vel_horizontal"].rolling(15, min_periods=1, center=True).median()
    )
    df["angular_norm"] = np.sqrt(
        df["roll_speed"] ** 2 + df["pitch_speed"] ** 2 + df["yaw_speed"] ** 2
    )
    df["angular_norm_suave"] = (
        df["angular_norm"].rolling(15, min_periods=1, center=True).median()
    )
    add_route_metrics(df, build_route(df))


def normalize(series: pd.Series, high_quantile: float = 0.95) -> pd.Series:
    """Normaliza magnitudes pelo quantil informado, limitando o resultado a [0, 1]."""

    clean = series.replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
    scale = float(clean.quantile(high_quantile))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = float(clean.max())
    if not math.isfinite(scale) or scale <= 1e-9:
        return clean * 0.0
    return (clean / scale).clip(0.0, 1.0)


def add_reactive_proxy(df: pd.DataFrame) -> None:
    """Estima a atividade reativa para CSVs legados sem sinais reais de evasao."""

    lateral = normalize(df["vel_lateral_rota"].rolling(9, min_periods=1, center=True).median())
    angular = normalize(df["angular_norm_suave"])
    deviation = (df["desvio_rota_m"] / max(RAIO_DE_ACEITACAO, 1.0)).clip(0.0, 1.0)

    speed_reference = df["vel_horizontal_suave"].rolling(45, min_periods=1, center=True).quantile(0.8)
    speed_reference = speed_reference.clip(lower=0.01)
    braking = ((speed_reference - df["vel_horizontal_suave"]) / speed_reference).clip(0.0, 1.0)

    proxy = (0.42 * lateral) + (0.28 * angular) + (0.20 * deviation) + (0.10 * braking)
    proxy = proxy.where(df["evasao_habilitada_modelo"], 0.0)

    df["proxy_desvio_reativo"] = proxy.clip(0.0, 1.0)
    df["evento_desvio_proxy"] = df["proxy_desvio_reativo"] > OBSTACLE_RISK_THRESHOLD
    df["lado_desvio_proxy"] = np.where(df["vel_lateral_rota"] >= 0, "direita/anti-horario", "esquerda/horario")
    df["indice_desvio_reativo"] = df["proxy_desvio_reativo"]
    df["evento_desvio_reativo"] = df["evento_desvio_proxy"]
    df["lateral_reativo_m_s2"] = df["vel_lateral_rota"].rolling(9, min_periods=1, center=True).median()
    df["freio_reativo"] = braking
    df["fonte_desvio_reativo"] = "proxy estimado"


def fmt_number(value: float, suffix: str = "", precision: int = 1) -> str:
    if value is None or not math.isfinite(float(value)):
        return "-"
    return f"{value:.{precision}f}{suffix}"


def metric_card(label: str, value: str, detail: str = "") -> html.Div:
    return html.Div(
        className="metric-card",
        children=[
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(detail, className="metric-detail"),
        ],
    )


def run_summary(df: pd.DataFrame) -> list[html.Div]:
    """Resume duracao, trajeto e atividade reativa da run selecionada."""

    if df.empty:
        return [metric_card("Sem dados", "-", "Nenhum CSV encontrado em logs/.")]

    total_distance = float(
        np.sqrt(
            df["x"].diff().fillna(0.0) ** 2
            + df["y"].diff().fillna(0.0) ** 2
            + df["z"].diff().fillna(0.0) ** 2
        ).sum()
    )
    event_share = float(df["evento_desvio_reativo"].mean() * 100.0)
    max_reactive = float(df["indice_desvio_reativo"].max())
    source = str(df["fonte_desvio_reativo"].iloc[0]) if "fonte_desvio_reativo" in df.columns else "proxy estimado"
    if source == "metricas reais":
        risk_label = "Risco visual max."
    elif source == "deltas memmap":
        risk_label = "Delta risco norm."
    else:
        risk_label = "Proxy reativo max."

    return [
        metric_card("Duracao", fmt_number(float(df["tempo_s"].max()), " s", 1), f"{len(df)} amostras"),
        metric_card("Distancia 3D", fmt_number(total_distance, " m", 1), "odometria acumulada"),
        metric_card("Altitude max.", fmt_number(float(df["altitude"].max()), " m", 2), "NED convertido para altitude"),
        metric_card(
            "Desvio rota max.",
            fmt_number(float(df["desvio_rota_m"].max()), " m", 2),
            "distancia lateral ao segmento",
        ),
        metric_card(risk_label, fmt_number(max_reactive, "", 2), f"{source}; limiar {OBSTACLE_RISK_THRESHOLD:.2f}"),
        metric_card("Tempo em desvio", fmt_number(event_share, "%", 1), "indice acima do limiar"),
    ]


def flight_notice(df: pd.DataFrame) -> html.Div:
    """Informa se os sinais reativos sao reais, deltas memmap ou proxy legado."""

    if not df.empty and has_real_avoidance_metrics(df):
        text = (
            "Esta run contem obstacle_risk, avoid_lateral_body e avoid_brake gravados no CSV. "
            "Os graficos de evasao usam as metricas reais calculadas pelo controlador."
        )
        return html.Div(text, className="notice notice-ok")

    source = (
        str(df["fonte_desvio_reativo"].iloc[0])
        if not df.empty and "fonte_desvio_reativo" in df
        else ""
    )
    if not df.empty and source == "deltas memmap":
        text = (
            "Esta run veio do formato novo em memmap. A trajetoria e relativa, reconstruida "
            "acumulando delta_x/delta_y/delta_z, e os graficos de IMU usam as variacoes "
            "entre amostras salvas no log."
        )
        return html.Div(text, className="notice notice-ok")

    text = (
        "Esta run foi coletada antes do logger salvar obstacle_risk, avoid_lateral_body e "
        "avoid_brake. O dashboard esta usando um proxy apenas para manter compatibilidade; "
        "novas runs gravadas com a versao atual passam a resolver este aviso."
    )
    return html.Div(text, className="notice")


def depth_summary(df: pd.DataFrame, metadata_path: Path | None) -> list[html.Div]:
    """Resume sincronizacao, flow e proximidade da run depth selecionada."""

    if df.empty or metadata_path is None:
        return [metric_card("Depth GT", "-", "Nenhuma run em datasets/depth_ground_truth/.")]

    if is_depth_interval_memmap(df):
        event_rate = 0.0
        if "delta_depth_close_5m_pp" in df.columns:
            event_rate = float((df["delta_depth_close_5m_pp"].abs() > 0.5).mean() * 100.0)
        return [
            metric_card("Dataset pareado", depth_run_label(metadata_path), f"{len(df)} intervalos depth/flow"),
            metric_card(
                "dt visual medio",
                fmt_number(float(df["dt_s"].mean() * 1000.0), " ms", 1),
                "entre atualizacoes visuais",
            ),
            metric_card(
                "RGB-depth medio",
                fmt_number(float(df["depth_age_s"].mean() * 1000.0), " ms", 1),
                "idade do depth usado",
            ),
            metric_card(
                "Flow valido",
                fmt_number(float(df["flow_valid_points"].mean()), " pts", 1),
                "media por intervalo",
            ),
            metric_card("Eventos prox.", fmt_number(event_rate, "%", 1), "|delta pixels < 5m| > 0,5 p.p."),
            metric_card(
                "Pan comp.",
                fmt_number(float(df["pan_comp_delta_rad"].abs().max()), " rad", 4),
                "maximo absoluto",
            ),
        ]

    pan_detail = "compensacao ausente nesta run"
    if df["pan_comp_delta_rad"].notna().any():
        pan_detail = f"pan mediano {float(df['pan_comp_delta_rad'].abs().median()):.4f} rad"

    return [
        metric_card("Depth run", depth_run_label(metadata_path), f"{len(df)} pares RGB/depth"),
        metric_card("Sincronia media", fmt_number(float(df["depth_age_s"].mean() * 1000.0), " ms", 1), "RGB x depth"),
        metric_card("Depth P10 med.", fmt_number(float(df["depth_p10_m"].median()), " m", 2), "percentil 10 por frame"),
        metric_card(
            "Pixels < 5 m",
            fmt_number(float(df["depth_close_5m_pct"].median()), "%", 1),
            "mediana da area valida",
        ),
        metric_card("Giro mediano", fmt_number(float(df["gyro_norm"].median()), " rad/s", 3), "norma do giroscopio"),
        metric_card("Pan comp.", fmt_number(float(df["pan_comp_delta_rad"].abs().max()), " rad", 4), pan_detail),
    ]


def apply_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Aplica o estilo visual compartilhado pelos graficos do dashboard."""

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16)),
        template="plotly_white",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        height=height,
        margin=dict(l=48, r=28, t=72, b=46),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.82)",
            bordercolor="rgba(213, 221, 227, 0.72)",
            borderwidth=1,
        ),
        font=dict(family="Segoe UI, Arial, sans-serif", color=COLORS["ink"]),
        hovermode="closest",
    )
    return fig


def table_figure(
    headers: list[str],
    columns: list[list],
    title: str,
    height: int,
    header_color: str = "#e8eef3",
) -> go.Figure:
    """Cria uma tabela Plotly com a identidade visual do dashboard."""

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color=header_color,
                    align="left",
                    font=dict(size=13),
                ),
                cells=dict(values=columns, fill_color="#ffffff", align="left", height=28),
            )
        ]
    )
    return apply_layout(fig, title, height)


def rows_to_columns(rows: list[list], column_count: int) -> list[list]:
    return list(map(list, zip(*rows))) if rows else [[] for _ in range(column_count)]


def format_table_number(value: float, digits: int = 1) -> str:
    return "N/D" if pd.isna(value) else f"{float(value):.{digits}f}"


def figure_xy(df: pd.DataFrame) -> go.Figure:
    """Compara a trajetoria XY executada com a rota planejada."""

    route = build_route(df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=route[:, 0],
            y=route[:, 1],
            mode="lines+markers",
            name="rota planejada",
            line=dict(color=COLORS["route"], width=3, dash="dash"),
            marker=dict(size=8, color=COLORS["route"]),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra>rota</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            name="trajetoria real",
            marker=dict(
                size=5,
                color=df["indice_desvio_reativo"],
                colorscale=[[0, "#2563eb"], [0.45, "#f59e0b"], [1, "#c2410c"]],
                cmin=0,
                cmax=1,
                colorbar=dict(title="evasao"),
            ),
            customdata=np.stack(
                [
                    df["tempo_s"],
                    df["desvio_rota_m"],
                    df["vel_lateral_rota"],
                    df["indice_desvio_reativo"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "t=%{customdata[0]:.2f}s<br>"
                "x=%{x:.2f}<br>y=%{y:.2f}<br>"
                "desvio=%{customdata[1]:.2f}m<br>"
                "vel lateral=%{customdata[2]:.2f}m/s<br>"
                "indice evasao=%{customdata[3]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return apply_layout(fig, "Trajetoria XY: rota planejada x desvio reativo", 520)


def figure_3d(df: pd.DataFrame) -> go.Figure:
    route = build_route(df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=route[:, 0],
            y=route[:, 1],
            z=-route[:, 2],
            mode="lines+markers",
            name="rota planejada",
            line=dict(color=COLORS["route"], width=5, dash="dash"),
            marker=dict(size=4, color=COLORS["route"]),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["altitude"],
            mode="lines",
            name="trajetoria real",
            line=dict(color=COLORS["drone"], width=5),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[df["x"].iloc[0], df["x"].iloc[-1]],
            y=[df["y"].iloc[0], df["y"].iloc[-1]],
            z=[df["altitude"].iloc[0], df["altitude"].iloc[-1]],
            mode="markers+text",
            text=["inicio", "fim"],
            textposition="top center",
            name="marcadores",
            marker=dict(size=5, color=["#16a34a", "#dc2626"]),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Altitude (m)",
            camera=dict(eye=dict(x=1.45, y=1.45, z=0.75)),
        )
    )
    return apply_layout(fig, "Trajetoria 3D do VANT", 520)


def figure_reactive_timeseries(df: pd.DataFrame) -> go.Figure:
    """Mostra risco, freio e comando lateral reativo ao longo da run."""

    source = str(df["fonte_desvio_reativo"].iloc[0]) if "fonte_desvio_reativo" in df.columns else "proxy estimado"
    if source == "metricas reais":
        index_name = "obstacle_risk"
        brake_name = "avoid_brake"
        lateral_name = "|avoid_lateral_body|"
    elif source == "deltas memmap":
        index_name = "|delta obstacle_risk| normalizado"
        brake_name = "|delta avoid_brake| normalizado"
        lateral_name = "|delta avoid_lateral_body| normalizado"
    else:
        index_name = "proxy desvio reativo"
        brake_name = "freio estimado"
        lateral_name = "|lateral estimado|"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=df["indice_desvio_reativo"],
            name=index_name,
            line=dict(color=COLORS["risk"], width=2),
            fill="tozeroy",
            fillcolor="rgba(194, 65, 12, 0.16)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=df["freio_reativo"],
            name=brake_name,
            line=dict(color=COLORS["accent"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=normalize(df["lateral_reativo_m_s2"]),
            name=lateral_name,
            line=dict(color="#7c3aed", width=1.8, dash="dash"),
        )
    )
    fig.add_hline(
        y=OBSTACLE_RISK_THRESHOLD,
        line_dash="dot",
        line_color="#991b1b",
        annotation_text="limiar de acionamento visual",
        annotation_position="top left",
    )
    fig.update_yaxes(title="indice normalizado", range=[0, max(1.05, float(df["indice_desvio_reativo"].max()) + 0.1)])
    fig.update_xaxes(title="Tempo de voo (s)")
    return apply_layout(fig, "Sinais derivados da evasao reativa", 390)


def figure_control(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in [
        ("roll_speed", "#2563eb"),
        ("pitch_speed", "#16a34a"),
        ("yaw_speed", "#c2410c"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["tempo_s"],
                y=df[col],
                name=col,
                mode="lines",
                line=dict(width=1.8, color=color),
            )
        )
    source = str(df["fonte_desvio_reativo"].iloc[0]) if "fonte_desvio_reativo" in df.columns else ""
    y_title = "Delta velocidade angular (rad/s)" if source == "deltas memmap" else "Velocidade angular (rad/s)"
    title = "Variacao das velocidades angulares" if source == "deltas memmap" else "Esforco de controle angular"
    fig.update_xaxes(title="Tempo de voo (s)")
    fig.update_yaxes(title=y_title)
    return apply_layout(fig, title, 390)


def figure_speed(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=df["vel_horizontal_suave"],
            name="velocidade horizontal",
            line=dict(color=COLORS["drone"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=df["vel_lateral_rota"].rolling(9, min_periods=1, center=True).median(),
            name="velocidade lateral ao corredor",
            line=dict(color=COLORS["risk"], width=2),
        )
    )
    fig.add_hline(
        y=VELOCIDADE_MAXIMA,
        line_dash="dot",
        line_color=COLORS["muted"],
        annotation_text="velocidade maxima do controlador",
        annotation_position="top left",
    )
    fig.update_xaxes(title="Tempo de voo (s)")
    fig.update_yaxes(title="m/s")
    return apply_layout(fig, "Velocidade e manobra lateral", 390)


def figure_altitude(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=df["altitude"],
            name="altitude real",
            line=dict(color=COLORS["drone"], width=2),
        )
    )
    fig.update_xaxes(title="Tempo de voo (s)")
    fig.update_yaxes(title="Altitude (m)")
    return apply_layout(fig, "Altitude ao longo do voo", 320)


def figure_depth_timeseries(df: pd.DataFrame) -> go.Figure:
    """Mostra a evolucao temporal dos indicadores de profundidade e proximidade."""

    if is_depth_interval_memmap(df):
        fig = go.Figure()
        for col, label, color in [
            ("delta_depth_p10_m", "delta P10", COLORS["risk"]),
            ("delta_depth_p50_m", "delta P50", COLORS["accent"]),
            ("delta_depth_p90_m", "delta P90", COLORS["drone"]),
        ]:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["tempo_s"],
                        y=df[col],
                        name=label,
                        mode="lines",
                        line=dict(color=color, width=2),
                    )
                )
        if "delta_depth_close_5m_pp" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["tempo_s"],
                    y=df["delta_depth_close_5m_pp"],
                    name="delta pixels < 5 m",
                    mode="lines",
                    yaxis="y2",
                    line=dict(color="#7c3aed", width=2, dash="dot"),
                )
            )
        fig.update_layout(
            yaxis=dict(title="Delta profundidade (m)"),
            yaxis2=dict(title="Delta pixels < 5 m (p.p.)", overlaying="y", side="right"),
        )
        fig.update_xaxes(title="Tempo acumulado dos intervalos (s)")
        return apply_layout(fig, "Intervalos depth/flow: variacao de profundidade e proximidade", 420)

    fig = go.Figure()
    for col, label, color in [
        ("depth_mean_m", "media", COLORS["drone"]),
        ("depth_p10_m", "P10", COLORS["risk"]),
        ("depth_p50_m", "P50", COLORS["accent"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["tempo_s"],
                y=df[col],
                name=f"depth {label}",
                mode="lines",
                line=dict(color=color, width=2),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=df["depth_close_5m_pct"],
            name="pixels < 5 m",
            mode="lines",
            yaxis="y2",
            line=dict(color="#7c3aed", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        yaxis=dict(title="Profundidade (m)"),
        yaxis2=dict(title="Pixels < 5 m (%)", overlaying="y", side="right", rangemode="tozero"),
    )
    fig.update_xaxes(title="Tempo desde o primeiro par (s)")
    return apply_layout(fig, "Depth ground truth: distancia e ocupacao proxima", 420)


def figure_depth_imu(df: pd.DataFrame) -> go.Figure:
    """Relaciona movimento visual ou proximidade com atividade inercial."""

    if is_depth_interval_memmap(df):
        x = df["radial_flow_p90_px"] if "radial_flow_p90_px" in df.columns else pd.Series(0.0, index=df.index)
        y = df["delta_depth_close_5m_pp"] if "delta_depth_close_5m_pp" in df.columns else pd.Series(0.0, index=df.index)
        color = df["flow_valid_points"] if "flow_valid_points" in df.columns else df["gyro_norm"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=8,
                    color=color,
                    colorscale=[[0, "#2563eb"], [0.55, "#f59e0b"], [1, "#c2410c"]],
                    colorbar=dict(title="flow pts"),
                    line=dict(color="white", width=0.6),
                ),
                customdata=np.stack(
                    [
                        df["sample_id"],
                        df["tempo_s"],
                        df["gyro_norm"],
                        df["pan_comp_delta_rad"].fillna(0.0),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "amostra=%{customdata[0]}<br>"
                    "t=%{customdata[1]:.2f}s<br>"
                    "flow radial P90=%{x:.2f}px<br>"
                    "delta <5m=%{y:.2f}p.p.<br>"
                    "gyro delta=%{customdata[2]:.3f}<br>"
                    "pan comp=%{customdata[3]:.4f}rad<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(title="P90 do flow radial (px)")
        fig.update_yaxes(title="Delta pixels < 5 m (p.p.)")
        return apply_layout(fig, "Flow radial x variacao de proximidade no depth", 420)

    color = df["pan_comp_delta_rad"].abs() if df["pan_comp_delta_rad"].notna().any() else df["depth_mean_m"]
    color_title = "|pan comp| rad" if df["pan_comp_delta_rad"].notna().any() else "depth media"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["depth_close_5m_pct"],
            y=df["gyro_norm"],
            mode="markers",
            marker=dict(
                size=8,
                color=color,
                colorscale=[[0, "#2563eb"], [0.55, "#f59e0b"], [1, "#c2410c"]],
                colorbar=dict(title=color_title),
                line=dict(color="white", width=0.6),
            ),
            customdata=np.stack(
                [
                    df["sample_id"],
                    df["tempo_s"],
                    df["depth_p10_m"],
                    df["accel_norm"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "amostra=%{customdata[0]}<br>"
                "t=%{customdata[1]:.2f}s<br>"
                "pixels < 5m=%{x:.2f}%<br>"
                "gyro=%{y:.3f}rad/s<br>"
                "depth P10=%{customdata[2]:.2f}m<br>"
                "accel=%{customdata[3]:.2f}m/s2<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Pixels validos com depth < 5 m (%)")
    fig.update_yaxes(title="Norma do giroscopio (rad/s)")
    return apply_layout(fig, "Depth x IMU: frames criticos para compensacao", 420)


def figure_depth_ranking(df: pd.DataFrame) -> go.Figure:
    """Lista os intervalos depth/flow mais informativos da run."""

    if is_depth_interval_memmap(df):
        ranking = (
            df.assign(abs_delta_close_5m=lambda frame: frame["delta_depth_close_5m_pp"].abs())
            .sort_values(["abs_delta_close_5m", "radial_flow_p90_px"], ascending=False)
            .head(15)
        )
        cols = [
            ("sample_id", "Amostra"),
            ("tempo_s", "Tempo (s)"),
            ("dt_s", "dt (s)"),
            ("flow_valid_points", "Flow pts"),
            ("radial_flow_p90_px", "Flow radial P90"),
            ("delta_depth_p50_m", "Delta P50 (m)"),
            ("delta_depth_close_5m_pp", "Delta <5m (p.p.)"),
        ]

        values = []
        for col, _ in cols:
            if col == "sample_id":
                values.append(ranking[col].astype(str).tolist())
            else:
                values.append([fmt_number(float(value), "", 3) for value in ranking[col]])

        return table_figure(
            [label for _, label in cols],
            values,
            "Intervalos mais informativos de depth/flow",
            360,
            header_color="#ece7ff",
        )

    ranking = df.sort_values(["depth_close_5m_pct", "gyro_norm"], ascending=False).head(15)
    cols = [
        ("sample_id", "Amostra"),
        ("tempo_s", "Tempo (s)"),
        ("depth_p10_m", "Depth P10 (m)"),
        ("depth_p50_m", "Depth P50 (m)"),
        ("depth_close_5m_pct", "Pixels < 5m (%)"),
        ("gyro_norm", "Gyro (rad/s)"),
        ("pan_comp_delta_rad", "Pan comp. (rad)"),
    ]

    values = []
    for col, _ in cols:
        if col == "sample_id":
            values.append(ranking[col].astype(str).tolist())
        elif col == "pan_comp_delta_rad" and ranking[col].isna().all():
            values.append(["nao coletado"] * len(ranking))
        else:
            values.append([fmt_number(float(value), "", 3) for value in ranking[col]])

    return table_figure(
        [label for _, label in cols],
        values,
        "Frames prioritarios para inspecao e treino",
        360,
        header_color="#ece7ff",
    )


def batch_interval_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Monta visoes cumulativas dos intervalos para cada marco de runs."""

    if df.empty:
        return pd.DataFrame()

    frames = []
    run_ids = ordered_run_ids(df)
    for size, label in BATCH_SPECS:
        if len(run_ids) < size:
            continue
        part = df[df["run_id"].isin(run_ids[:size])].copy()
        if part.empty:
            continue
        part["leva"] = label
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_batch_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Resume sincronizacao, flow e eventos de cada conjunto cumulativo."""

    if df.empty:
        return pd.DataFrame(REFERENCE_BATCH_STATS)

    rows = []
    run_ids = ordered_run_ids(df)
    for size, label in BATCH_SPECS:
        if len(run_ids) < size:
            continue
        part = df[df["run_id"].isin(run_ids[:size])]
        if part.empty:
            continue
        event_rate = 0.0
        event_count = 0
        delta_abs_p90 = np.nan
        zero_rate = np.nan
        if "delta_depth_close_5m_pp" in part.columns:
            delta_abs = part["delta_depth_close_5m_pp"].abs()
            event_count = int((delta_abs > 0.5).sum())
            event_rate = float((delta_abs > 0.5).mean() * 100.0)
            delta_abs_p90 = float(np.nanpercentile(delta_abs, 90))
            zero_rate = float((delta_abs < 1e-9).mean() * 100.0)
        rows.append(
            {
                "leva": label,
                "runs": int(part["run_id"].nunique()),
                "intervalos": int(len(part)),
                "dt_s_mean": float(part["dt_s"].mean()) if "dt_s" in part else np.nan,
                "depth_age_s_mean": float(part["depth_age_s"].mean()) if "depth_age_s" in part else np.nan,
                "flow_valid_points_mean": (
                    float(part["flow_valid_points"].mean())
                    if "flow_valid_points" in part
                    else np.nan
                ),
                "flow_track_retention_pct_mean": (
                    float(part["flow_track_retention_pct"].mean())
                    if "flow_track_retention_pct" in part
                    else np.nan
                ),
                "flow_mag_p90_px_mean": float(part["flow_mag_p90_px"].mean()) if "flow_mag_p90_px" in part else np.nan,
                "radial_flow_p90_px_mean": (
                    float(part["radial_flow_p90_px"].mean())
                    if "radial_flow_p90_px" in part
                    else np.nan
                ),
                "delta_depth_close_5m_pp_mean": (
                    float(part["delta_depth_close_5m_pp"].mean())
                    if "delta_depth_close_5m_pp" in part
                    else np.nan
                ),
                "delta_depth_close_5m_pp_std": (
                    float(part["delta_depth_close_5m_pp"].std())
                    if "delta_depth_close_5m_pp" in part
                    else np.nan
                ),
                "delta_abs_p90": delta_abs_p90,
                "zero_rate_pct": zero_rate,
                "eventos": event_count,
                "event_rate_pct": event_rate,
            }
        )
    return pd.DataFrame(rows)


def reference_mlp_df() -> pd.DataFrame:
    """Retorna os resultados controlados da MLP ou a referencia embutida."""

    fixed = load_validated_csv(
        MLP_FIXED_RESULTS_PATH,
        {"marco_runs", "modelo", "split", "alvo_delta", "MAE", "RMSE"},
    )
    if not fixed.empty:
        fixed = fixed[fixed["split"] == "teste"].copy()
        fixed["marco_runs"] = pd.to_numeric(fixed["marco_runs"], errors="coerce")
        fixed["leva"] = fixed["marco_runs"].map(lambda value: batch_label(int(value)))
        return fixed[["leva", "modelo", "alvo_delta", "MAE", "RMSE"]]
    return pd.DataFrame(
        REFERENCE_MLP_TEST,
        columns=["leva", "modelo", "alvo_delta", "MAE", "RMSE"],
    )


def reference_event_df() -> pd.DataFrame:
    """Retorna metricas do classificador fixo e do modelo pos-gate."""

    fixed = load_validated_csv(
        EVENT_FIXED_RESULTS_PATH,
        {
            "marco_runs", "split", "balanced_acc", "precision_evento",
            "recall_evento", "f1_evento", "avg_precision", "fp", "fn",
        },
    )
    if not fixed.empty:
        metrics = {
            "balanced_acc": "balanced accuracy",
            "precision_evento": "precisao",
            "recall_evento": "recall",
            "f1_evento": "F1",
            "avg_precision": "average precision",
            "fp": "falsos positivos",
            "fn": "falsos negativos",
        }
        rows = []
        for _, result in fixed[fixed["split"] == "teste"].iterrows():
            group = f"{int(result['marco_runs'])} runs - classificador fixo"
            rows.extend((group, label, float(result[column])) for column, label in metrics.items())
        if rows:
            return pd.DataFrame(rows, columns=["grupo", "metrica", "valor"])
    return pd.DataFrame(
        REFERENCE_EVENT_RESULTS,
        columns=["grupo", "metrica", "valor"],
    )


def reference_run_event_df() -> pd.DataFrame:
    df = pd.DataFrame(
        REFERENCE_RUN_EVENT_STATS,
        columns=["run_id", "intervalos", "eventos", "taxa_evento"],
    )
    df["leva"] = df["run_id"].map(run_block_label)
    return df


def batch_summary_cards(stats: pd.DataFrame, mlp_df: pd.DataFrame) -> list[html.Div]:
    """Cria os indicadores principais da comparacao entre marcos de runs."""

    if stats.empty:
        return [metric_card("Comparacao", "-", "Sem dados ou referencias para comparar.")]

    stats = stats.sort_values("runs").reset_index(drop=True)
    first_row = stats.iloc[0]
    previous_row = stats.iloc[-2] if len(stats) > 1 else first_row
    current_row = stats.iloc[-1]
    current_label = str(current_row["leva"])
    first_label = str(first_row["leva"])

    current_mlp = mlp_df[(mlp_df["leva"] == current_label) & (mlp_df["modelo"] == "MLP")]
    current_base = mlp_df[(mlp_df["leva"] == current_label) & (mlp_df["modelo"] == "Media treino")]
    joined = current_mlp.merge(current_base, on="alvo_delta", suffixes=("_mlp", "_base"))
    beats_baseline = int((joined["MAE_mlp"] < joined["MAE_base"]).sum()) if not joined.empty else 0

    first_mlp = mlp_df[(mlp_df["leva"] == first_label) & (mlp_df["modelo"] == "MLP")]
    mlp_change = current_mlp.merge(first_mlp, on="alvo_delta", suffixes=("_current", "_first"))
    improved_targets = int(
        (mlp_change["MAE_current"] < mlp_change["MAE_first"]).sum()
    ) if not mlp_change.empty else 0

    def incremental_event_rate(previous: pd.Series, current: pd.Series) -> float:
        interval_delta = int(current.get("intervalos", 0)) - int(previous.get("intervalos", 0))
        event_delta = int(current.get("eventos", 0)) - int(previous.get("eventos", 0))
        return event_delta / interval_delta * 100.0 if interval_delta > 0 else np.nan

    latest_event_rate = incremental_event_rate(previous_row, current_row)
    added_intervals = int(current_row["intervalos"]) - int(previous_row["intervalos"])
    current_runs = int(current_row["runs"])
    previous_runs = int(previous_row["runs"])
    first_runs = int(first_row["runs"])

    return [
        metric_card(
            "Intervalos",
            f"{int(current_row['intervalos'])}",
            " -> ".join(str(int(value)) for value in stats["intervalos"]),
        ),
        metric_card(
            f"Runs {previous_runs + 1}-{current_runs}",
            f"+{added_intervals}",
            f"intervalos validos sobre as primeiras {previous_runs}",
        ),
        metric_card(
            "Eventos no bloco novo",
            fmt_number(latest_event_rate, "%", 1),
            "taxa incremental, sem misturar os blocos anteriores",
        ),
        metric_card(
            "MLP x baseline",
            f"{beats_baseline}/6 alvos",
            f"resultado atual com {current_runs} runs",
        ),
        metric_card(
            f"MLP {first_runs} -> {current_runs}",
            f"{improved_targets}/6 alvos",
            "alvos com reducao de MAE no mesmo teste fixo",
        ),
        metric_card(
            "Eventos acumulados",
            fmt_number(float(current_row["event_rate_pct"]), "%", 1),
            f"primeiro marco: {float(first_row['event_rate_pct']):.1f}%",
        ),
    ]


def analysis_note(title: str, paragraphs: list[str], tone: str = "") -> html.Div:
    class_name = "analysis-note" + (f" analysis-note-{tone}" if tone else "")
    return html.Div(
        className=class_name,
        children=[html.H3(title), *[html.P(paragraph) for paragraph in paragraphs]],
    )


def batch_analysis_notes(
    stats: pd.DataFrame, mlp_df: pd.DataFrame, event_df: pd.DataFrame
) -> tuple[html.Div, ...]:
    """Gera os blocos textuais de interpretacao exibidos na aba comparativa."""

    stats = stats.sort_values("runs").reset_index(drop=True)
    first_row, current_row = stats.iloc[0], stats.iloc[-1]
    previous_row = stats.iloc[-2] if len(stats) > 1 else first_row
    first_runs, current_runs = int(first_row["runs"]), int(current_row["runs"])
    current_label = str(current_row["leva"])
    current_mlp = mlp_df[(mlp_df["leva"] == current_label) & (mlp_df["modelo"] == "MLP")]
    current_base = mlp_df[(mlp_df["leva"] == current_label) & (mlp_df["modelo"] == "Media treino")]
    joined = current_mlp.merge(current_base, on="alvo_delta", suffixes=("_mlp", "_base"))
    worst_target = "indisponivel"
    if not joined.empty:
        joined["gap"] = joined["MAE_mlp"] - joined["MAE_base"]
        worst_target = str(joined.sort_values("gap", ascending=False).iloc[0]["alvo_delta"])
    balanced = event_df[event_df["metrica"] == "balanced accuracy"].copy()
    classifier_summary = "As metricas do classificador nao estao disponiveis."
    if not balanced.empty:
        best = balanced.loc[balanced["valor"].idxmax()]
        latest = balanced.iloc[-1]
        classifier_summary = (
            f"A balanced accuracy atual e {float(latest['valor']):.3f}; o melhor marco foi "
            f"{best['grupo']} com {float(best['valor']):.3f}."
        )

    growth = analysis_note(
        "Comparar o desempenho conforme o conjunto cresce",
        [
            (
                f"Os intervalos validos cresceram de {int(first_row['intervalos'])} para "
                f"{int(current_row['intervalos'])} entre {first_runs} e {current_runs} runs. "
                f"O ultimo bloco adicionou {int(current_row['intervalos']) - int(previous_row['intervalos'])} intervalos."
            ),
            "Os graficos abaixo mantem os mesmos marcos do notebook e se atualizam a partir dos CSVs e memmaps disponiveis.",
        ],
        "ok",
    )
    stability = analysis_note(
        "Verificar se a melhora continua ou comeca a estabilizar",
        [
            (
                f"A taxa acumulada de eventos passou de {float(first_row['event_rate_pct']):.1f}% para "
                f"{float(current_row['event_rate_pct']):.1f}%. A curva por marco permite verificar se a distribuicao estabilizou."
            ),
            (
                f"No ultimo incremento, o P90 do delta absoluto foi de {float(current_row['delta_abs_p90']):.2f} p.p. "
                f"e os deltas zerados representaram {float(current_row.get('zero_rate_pct', np.nan)):.1f}%."
            ),
        ],
    )
    split = analysis_note(
        "Separar ganho real de variacao causada pela divisao treino/teste",
        [
            "A validacao usa duas runs fixas e o teste usa tres runs fixas desde o primeiro marco. Apenas o conjunto de treino cresce em 9, 17, 25, 33 e 40 runs.",
            (
                "Os CSVs controlados foram carregados pelo dashboard. Assim, a variacao atual de MAE e balanced accuracy nao inclui mudanca na composicao do teste."
                if has_fixed_comparison_results()
                else "Reexecute o notebook para gerar os CSVs da curva controlada."
            ),
        ],
        "warning",
    )
    errors = analysis_note(
        "Identificar quais alvos e eventos ainda concentram os erros",
        [
            f"No marco atual, o maior gap entre MLP e media do treino esta em {worst_target}.",
            classifier_summary,
            "A aba de modelo detalha separadamente a curva de loss, a representacao angular e as entradas associadas aos maiores erros.",
        ],
        "warning",
    )
    professor = analysis_note(
        "Resposta sugerida ao professor",
        [
            (
                f"Professor, atualizei a comparacao controlada para {', '.join(str(size) for size, _ in BATCH_SPECS)} runs. "
                f"O conjunto atual tem {int(current_row['intervalos'])} intervalos validos e mantem validacao e teste fixos."
            ),
            (
                "Tambem passei a acompanhar as curvas de loss e a explicabilidade por gradiente na validacao. "
                "A comparacao angular mostrou menor dependencia das entradas angulares, mas sem ganho consistente de MAE no teste fixo."
            ),
        ],
        "professor",
    )
    return growth, stability, split, errors, professor


def batch_notice(intervals_df: pd.DataFrame) -> html.Div:
    """Informa as fontes usadas e a cobertura de runs da comparacao atual."""

    if intervals_df.empty:
        text = (
            "Nao encontrei os memmaps de depth/flow em datasets/depth_ground_truth no ambiente atual. "
            "Esta aba esta usando os valores salvos no notebook para a comparacao estatistica e de MLP."
        )
        return html.Div(text, className="notice")

    run_count = intervals_df["run_id"].nunique()
    interval_count = len(intervals_df)
    snapshot_runs = 0
    if "source_format" in intervals_df:
        snapshot_runs = intervals_df.loc[
            intervals_df["source_format"] == "notebook_snapshot", "run_id"
        ].nunique()
    source_text = (
        f"; {snapshot_runs} runs recuperadas dos plots salvos no notebook"
        if snapshot_runs
        else "; todas lidas dos memmaps"
    )
    model_text = (
        "As curvas de MLP e classificacao com splits fixos foram carregadas dos CSVs gerados pelo notebook."
        if has_fixed_comparison_results()
        else (
            "O notebook esta configurado para treinar 9, 17, 25, 33 e 40 runs com validacao e teste fixos; "
            "rerode-o com todas as pastas para atualizar a curva controlada."
        )
    )
    text = (
        f"Comparacao cumulativa {' -> '.join(str(size) for size, _ in BATCH_SPECS)}: "
        f"{interval_count} intervalos em {run_count} runs"
        f"{source_text}. As metricas cumulativas descrevem a evolucao dos dados. {model_text}"
    )
    return html.Div(text, className="notice notice-ok")


def figure_batch_metric_facets(stats: pd.DataFrame) -> go.Figure:
    if stats.empty:
        return apply_layout(go.Figure(), "Resumo estatistico por leva")

    plot_df = stats[stats["leva"].isin(BATCH_LABEL_ORDER)].copy()
    if plot_df.empty:
        plot_df = stats.copy()
    plot_df["leva"] = pd.Categorical(plot_df["leva"], categories=BATCH_LABEL_ORDER, ordered=True)
    plot_df = plot_df.sort_values("leva")
    metrics = [
        ("intervalos", "Intervalos", ""),
        ("dt_s_mean", "dt medio", "ms"),
        ("depth_age_s_mean", "RGB-depth", "ms"),
        ("flow_valid_points_mean", "Flow valido", "pts"),
        ("flow_track_retention_pct_mean", "Retencao flow", "%"),
        ("event_rate_pct", "Eventos prox.", "%"),
    ]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[m[1] for m in metrics])
    for idx, (col, _label, unit) in enumerate(metrics):
        row = (idx // 3) + 1
        subplot_col = (idx % 3) + 1
        values = plot_df[col].astype(float)
        if unit == "ms":
            values = values * 1000.0
        fig.add_trace(
            go.Bar(
                x=plot_df["leva"],
                y=values,
                marker_color=[BATCH_COLORS.get(label, COLORS["drone"]) for label in plot_df["leva"]],
                text=[fmt_number(float(value), "", 1) for value in values],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{x}<br>%{y:.3f} " + unit + "<extra></extra>",
            ),
            row=row,
            col=subplot_col,
        )
        fig.update_yaxes(title=unit, row=row, col=subplot_col)

    return apply_layout(fig, "Evolucao cumulativa das metricas por marco", 620)


def figure_batch_distributions(batch_df: pd.DataFrame) -> go.Figure:
    if batch_df.empty:
        return apply_layout(go.Figure(), "Distribuicoes por leva")

    metrics = [
        ("dt_s", "dt entre intervalos", "ms", 1000.0),
        ("radial_flow_p90_px", "P90 flow radial", "px", 1.0),
        ("flow_valid_points", "Pontos validos", "pts", 1.0),
        ("delta_depth_close_5m_pp", "Delta pixels < 5m", "p.p.", 1.0),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[m[1] for m in metrics])
    labels = BATCH_LABEL_ORDER
    for idx, (col, _label, unit, scale) in enumerate(metrics):
        if col not in batch_df.columns:
            continue
        row = (idx // 2) + 1
        subplot_col = (idx % 2) + 1
        for label in labels:
            part = batch_df[batch_df["leva"] == label]
            if part.empty:
                continue
            fig.add_trace(
                go.Box(
                    y=part[col].astype(float) * scale,
                    name=label,
                    marker_color=BATCH_COLORS[label],
                    boxmean=True,
                    showlegend=idx == 0,
                    hovertemplate=label + "<br>%{y:.3f} " + unit + "<extra></extra>",
                ),
                row=row,
                col=subplot_col,
            )
        fig.update_yaxes(title=unit, row=row, col=subplot_col)

    return apply_layout(fig, "Distribuicoes cumulativas por marco", 700)


def figure_batch_event_rates(intervals_df: pd.DataFrame) -> go.Figure:
    if intervals_df.empty or "delta_depth_close_5m_pp" not in intervals_df.columns:
        summary = reference_run_event_df()
    else:
        summary = (
            intervals_df.assign(evento=lambda df: df["delta_depth_close_5m_pp"].abs() > 0.5)
            .groupby("run_id", as_index=False)
            .agg(intervalos=("evento", "size"), eventos=("evento", "sum"), taxa_evento=("evento", "mean"))
        )
        run_ids = ordered_run_ids(intervals_df)
        run_positions = {run_id: index for index, run_id in enumerate(run_ids)}

        def block_for_run(run_id: str) -> str:
            position = run_positions.get(run_id, len(run_positions))
            for block_index, (size, _label) in enumerate(BATCH_SPECS):
                if position < size:
                    return BLOCK_LABELS[block_index]
            return "Runs adicionais"

        summary["leva"] = summary["run_id"].map(block_for_run)
    summary = summary.sort_values("run_id")
    fig = go.Figure()
    for block_label in BLOCK_LABELS:
        part = summary[summary["leva"] == block_label]
        if part.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=part["run_id"].str.replace("run_", "", regex=False),
                y=part["taxa_evento"] * 100.0,
                name=block_label,
                marker_color=BLOCK_COLORS[block_label],
                customdata=np.stack([part["intervalos"], part["eventos"]], axis=-1),
                hovertemplate=(
                    "run=%{x}<br>eventos=%{customdata[1]} / %{customdata[0]}<br>"
                    "taxa=%{y:.1f}%<extra>" + block_label + "</extra>"
                ),
            )
        )
    fig.update_xaxes(title="Run", tickangle=-25)
    fig.update_yaxes(title="Eventos com |delta pixels < 5m| > 0,5 p.p. (%)")
    return apply_layout(fig, "Taxa de eventos de proximidade por run", 420)


def figure_mlp_reference_mae(mlp_df: pd.DataFrame) -> go.Figure:
    if mlp_df.empty:
        return apply_layout(go.Figure(), "MAE no teste: MLP x baseline")

    target_order = [
        "delta_depth_p10_m",
        "delta_depth_p50_m",
        "delta_depth_p90_m",
        "delta_depth_close_2m_pp",
        "delta_depth_close_5m_pp",
        "delta_depth_close_10m_pp",
    ]
    fig = go.Figure()
    available_labels = [label for _size, label in BATCH_SPECS if label in set(mlp_df["leva"])]
    if not available_labels:
        available_labels = list(dict.fromkeys(mlp_df["leva"]))
    series = [
        (label, "MLP", BATCH_COLORS.get(label, COLORS["drone"]))
        for label in available_labels
    ]
    series.append((available_labels[-1], "Media treino", COLORS["risk"]))
    for leva, modelo, color in series:
        part = mlp_df[(mlp_df["leva"] == leva) & (mlp_df["modelo"] == modelo)].set_index("alvo_delta")
        values = [float(part.loc[target, "MAE"]) if target in part.index else np.nan for target in target_order]
        fig.add_trace(
            go.Bar(
                x=target_order,
                y=values,
                name=f"{leva} - {modelo}",
                marker_color=color,
                hovertemplate="%{x}<br>MAE=%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.add_annotation(
        text=(
            "Validacao e teste fixos; apenas o treino cresce"
            if MLP_FIXED_RESULTS_PATH.exists()
            else "Referencias anteriores; rerode o notebook para atualizar o teste fixo"
        ),
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.12,
        showarrow=False,
        xanchor="right",
        font=dict(color=COLORS["muted"], size=12),
    )
    fig.update_xaxes(title="Alvo delta", tickangle=-20)
    fig.update_yaxes(title="MAE no teste")
    title = "MAE no teste fixo por marco de runs"
    return apply_layout(fig, title, 520)


def figure_mlp_reference_table(mlp_df: pd.DataFrame) -> go.Figure:
    if mlp_df.empty:
        return apply_layout(go.Figure(), "Tabela de MAE do notebook")

    rows = []
    targets = sorted(mlp_df["alvo_delta"].unique())

    def mae_value(target: str, batch_label: str, model: str) -> float:
        part = mlp_df[
            (mlp_df["leva"] == batch_label)
            & (mlp_df["modelo"] == model)
            & (mlp_df["alvo_delta"] == target)
        ]
        return float(part["MAE"].iloc[0]) if not part.empty else np.nan

    labels = [label for _size, label in BATCH_SPECS if label in set(mlp_df["leva"])]
    sizes = [size for size, label in BATCH_SPECS if label in labels]
    if not labels:
        labels = list(dict.fromkeys(mlp_df["leva"]))
        sizes = list(range(1, len(labels) + 1))
    for target in targets:
        mlp_values = [mae_value(target, label, "MLP") for label in labels]
        baseline = mae_value(target, labels[-1], "Media treino")
        change = (mlp_values[-1] / mlp_values[0] - 1.0) * 100.0 if mlp_values[0] > 0 else np.nan
        gap = mlp_values[-1] - baseline
        rows.append([
            target,
            *[f"{value:.3f}" for value in mlp_values],
            f"{change:+.1f}%",
            f"{baseline:.3f}",
            f"{gap:+.3f}",
        ])

    headers = [
        "Alvo",
        *[f"MLP {size}" for size in sizes],
        f"Variacao {sizes[0]}-{sizes[-1]}",
        f"Media {sizes[-1]}",
        "Gap atual",
    ]
    return table_figure(
        headers,
        rows_to_columns(rows, len(headers)),
        (
            "Curva de erro no teste fixo por marco"
            if MLP_FIXED_RESULTS_PATH.exists()
            else "Erros registrados antes da curva com split fixo"
        ),
        340,
    )


def figure_event_reference_table(event_df: pd.DataFrame) -> go.Figure:
    if event_df.empty:
        return apply_layout(go.Figure(), "Modelo em duas etapas")

    values = [
        event_df["grupo"].tolist(),
        event_df["metrica"].tolist(),
        [f"{float(value):.3f}" for value in event_df["valor"]],
    ]
    return table_figure(
        ["Grupo", "Metrica", "Valor no teste atual"],
        values,
        "Classificador de eventos no teste fixo por marco",
        620,
    )


def figure_batch_stats_table(stats: pd.DataFrame) -> go.Figure:
    if stats.empty:
        return apply_layout(go.Figure(), "Tabela de estatisticas por leva")

    rows = []
    for _, row in stats.iterrows():
        rows.append(
            [
                row["leva"],
                f"{int(row['runs'])}",
                f"{int(row['intervalos'])}",
                format_table_number(row["dt_s_mean"] * 1000.0),
                format_table_number(row["depth_age_s_mean"] * 1000.0),
                format_table_number(row["flow_valid_points_mean"]),
                format_table_number(row["event_rate_pct"]),
                format_table_number(row.get("zero_rate_pct", np.nan)),
                format_table_number(row.get("delta_abs_p90", np.nan), 2),
                format_table_number(row["delta_depth_close_5m_pp_std"], 2),
            ]
        )
    headers = [
        "Leva",
        "Runs",
        "Intervalos",
        "dt medio (ms)",
        "RGB-depth (ms)",
        "Flow valido",
        "Eventos (%)",
        "Delta zero (%)",
        "P90 |delta|",
        "Std delta <5m",
    ]
    return table_figure(
        headers,
        rows_to_columns(rows, len(headers)),
        "Tabela estatistica das levas",
        340,
    )


def load_model_diagnostics() -> dict[str, pd.DataFrame]:
    """Carrega os artefatos de treino e explicabilidade exportados pelo notebook."""

    return {
        "angular": load_validated_csv(
            ANGULAR_COMPARISON_PATH,
            {"modelo", "split", "alvo_delta", "MAE", "marco_runs", "representacao_angular"},
        ),
        "loss": load_validated_csv(
            LOSS_CURVES_PATH,
            {"epoca", "split", "mse_padronizado", "marco_runs"},
        ),
        "importance": load_validated_csv(
            GRADIENT_IMPORTANCE_PATH,
            {"alvo_delta", "feature", "grupo", "importancia_pct"},
        ),
        "grouped_importance": load_validated_csv(
            GROUPED_GRADIENT_IMPORTANCE_PATH,
            {"alvo_delta", "feature", "grupo", "importancia_pct"},
        ),
        "errors": load_validated_csv(
            LARGEST_GRADIENT_ERRORS_PATH,
            {
                "run_id", "sample_id", "real", "predito", "erro_absoluto",
                "feature_1", "importancia_1_pct", "feature_2", "importancia_2_pct",
                "feature_3", "importancia_3_pct",
            },
        ),
    }


def model_diagnostic_cards(data: dict[str, pd.DataFrame]) -> list[html.Div]:
    """Resume configuracao, parada do treino e cobertura das explicacoes."""

    loss_df = data["loss"]
    errors_df = data["errors"]
    latest_mark = int(loss_df["marco_runs"].max()) if not loss_df.empty else 0
    validation = loss_df[loss_df["split"] == "validacao"]
    best_epoch = int(validation.loc[validation["mse_padronizado"].idxmin(), "epoca"]) if not validation.empty else 0
    stopped_epoch = int(loss_df["epoca"].max()) if not loss_df.empty else 0
    zero_gradients = 0
    if not errors_df.empty:
        importance_cols = ["importancia_1_pct", "importancia_2_pct", "importancia_3_pct"]
        zero_gradients = int((errors_df[importance_cols].fillna(0.0).abs().sum(axis=1) <= 1e-12).sum())
    return [
        metric_card("Marco atual", f"{latest_mark} runs", "validacao e teste fixos"),
        metric_card("Arquitetura", "64 -> 16", "ReLU e dropout entre camadas"),
        metric_card("Dropout", "0,30", "aplicado nas duas camadas ocultas"),
        metric_card("Early stopping", f"epoca {best_epoch}", f"interrompido na epoca {stopped_epoch}; patience 50"),
        metric_card("Criterio", "loss validacao", "melhora minima de 0,0001"),
        metric_card("Gradiente zero", str(zero_gradients), "maiores erros sem ranking local valido"),
    ]


def figure_loss_curves(loss_df: pd.DataFrame) -> go.Figure:
    """Mostra as losses por epoca e a epoca restaurada pelo early stopping."""

    if loss_df.empty:
        return apply_layout(go.Figure(), "Curvas de loss indisponiveis")
    latest = int(loss_df["marco_runs"].max())
    plot_df = loss_df[loss_df["marco_runs"] == latest].sort_values("epoca")
    colors = {"treino": COLORS["drone"], "validacao": COLORS["risk"], "teste": COLORS["accent"]}
    fig = go.Figure()
    for split in ("treino", "validacao", "teste"):
        part = plot_df[plot_df["split"] == split]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["epoca"], y=part["mse_padronizado"], mode="lines",
            name=split.capitalize(), line=dict(color=colors[split], width=2),
            hovertemplate="epoca=%{x}<br>MSE=%{y:.4f}<extra>" + split + "</extra>",
        ))
    validation = plot_df[plot_df["split"] == "validacao"]
    if not validation.empty:
        best = validation.loc[validation["mse_padronizado"].idxmin()]
        fig.add_vline(x=float(best["epoca"]), line_dash="dash", line_color=COLORS["risk"])
        fig.add_annotation(
            x=float(best["epoca"]), y=float(best["mse_padronizado"]),
            text=f"melhor validacao: epoca {int(best['epoca'])}", showarrow=True,
            arrowhead=2, bgcolor="white",
        )
    fig.update_xaxes(title="Epoca")
    fig.update_yaxes(title="MSE padronizado")
    return apply_layout(fig, f"Curvas de loss com {latest} runs", 470)


def figure_angular_comparison(angular_df: pd.DataFrame, split: str) -> go.Figure:
    """Compara valor angular e seno/cosseno no mesmo split e marco."""

    if angular_df.empty:
        return apply_layout(go.Figure(), "Comparacao angular indisponivel")
    latest = int(pd.to_numeric(angular_df["marco_runs"], errors="coerce").max())
    plot_df = angular_df[
        (angular_df["split"] == split)
        & (pd.to_numeric(angular_df["marco_runs"], errors="coerce") == latest)
        & (angular_df["modelo"] == "MLP")
    ].copy()
    target_order = [
        "delta_depth_p10_m", "delta_depth_p50_m", "delta_depth_p90_m",
        "delta_depth_close_2m_pp", "delta_depth_close_5m_pp", "delta_depth_close_10m_pp",
    ]
    titles = ["P10", "P50", "P90", "Pixels < 2 m", "Pixels < 5 m", "Pixels < 10 m"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles)
    representations = (("Valor angular", COLORS["drone"]), ("Seno/cosseno", "#f59e42"))
    for index, target in enumerate(target_order):
        row, col = index // 3 + 1, index % 3 + 1
        part = plot_df[plot_df["alvo_delta"] == target]
        for representation, color in representations:
            value = part.loc[part["representacao_angular"] == representation, "MAE"]
            if value.empty:
                continue
            fig.add_trace(go.Bar(
                x=[representation], y=[float(value.iloc[0])], name=representation,
                legendgroup=representation, showlegend=index == 0, marker_color=color,
                text=[f"{float(value.iloc[0]):.3f}"], textposition="outside",
                hovertemplate=f"{representation}<br>MAE=%{{y:.4f}}<extra></extra>",
            ), row=row, col=col)
        fig.update_yaxes(title="MAE", row=row, col=col)
    fig = apply_layout(fig, f"Representacao angular na {split} fixa ({latest} runs)", 650)
    fig.update_layout(
        margin=dict(l=48, r=28, t=105, b=46),
        legend=dict(y=1.08, yanchor="bottom", x=0.5, xanchor="center"),
    )
    return fig


def figure_gradient_importance(importance_df: pd.DataFrame, target: str) -> go.Figure:
    """Exibe as features e grupos mais sensiveis no erro de validacao."""

    part = importance_df[importance_df["alvo_delta"] == target].copy()
    if part.empty:
        return apply_layout(go.Figure(), "Importancia por gradiente indisponivel")
    top = part.nlargest(12, "importancia_pct").sort_values("importancia_pct")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Features", "Grupos"), column_widths=[0.65, 0.35])
    fig.add_trace(go.Bar(
        x=top["importancia_pct"], y=top["feature"], orientation="h",
        marker_color=COLORS["drone"], showlegend=False,
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    ), row=1, col=1)
    groups = part.groupby("grupo", as_index=False)["importancia_pct"].sum().sort_values("importancia_pct")
    fig.add_trace(go.Bar(
        x=groups["importancia_pct"], y=groups["grupo"], orientation="h",
        marker_color=COLORS["accent"], showlegend=False,
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    ), row=1, col=2)
    fig.update_xaxes(title="Importancia no gradiente (%)")
    return apply_layout(fig, f"Entradas associadas ao erro de validacao: {target}", 560)


def figure_largest_gradient_errors(errors_df: pd.DataFrame) -> go.Figure:
    """Lista maiores erros sem inventar ranking quando o gradiente local e zero."""

    if errors_df.empty:
        return apply_layout(go.Figure(), "Maiores erros indisponiveis")
    rows = []
    for _, row in errors_df.sort_values("erro_absoluto", ascending=False).iterrows():
        importances = [float(row[f"importancia_{index}_pct"]) for index in range(1, 4)]
        if sum(abs(value) for value in importances) <= 1e-12:
            explanation = "Gradiente local zero - sem ranking valido"
        else:
            explanation = " | ".join(
                f"{row[f'feature_{index}']} ({row[f'importancia_{index}_pct']:.2f}%)"
                for index in range(1, 4)
            )
        rows.append([
            str(row["run_id"]).replace("run_", ""), int(row["sample_id"]),
            f"{float(row['real']):.2f}", f"{float(row['predito']):.2f}",
            f"{float(row['erro_absoluto']):.2f}", explanation,
        ])
    headers = ["Run", "Amostra", "Real", "Predito", "Erro abs.", "Entradas mais influentes no erro local"]
    return table_figure(
        headers, rows_to_columns(rows, len(headers)),
        "Maiores erros de validacao em pixels < 5 m", 560,
    )


def comparison_table(paths: list[Path]) -> go.Figure:
    """Compara duracao, distancia e atividade reativa entre runs de voo."""

    rows = []
    for path in paths:
        df = load_run(path)
        if df.empty:
            continue
        dist = float(
            np.sqrt(
                df["x"].diff().fillna(0.0) ** 2
                + df["y"].diff().fillna(0.0) ** 2
                + df["z"].diff().fillna(0.0) ** 2
            ).sum()
        )
        rows.append(
            [
                run_label(path),
                f"{float(df['tempo_s'].max()):.1f}",
                f"{dist:.1f}",
                f"{float(df['altitude'].max()):.2f}",
                f"{float(df['desvio_rota_m'].max()):.2f}",
                f"{float(df['indice_desvio_reativo'].max()):.2f}",
                f"{float(df['evento_desvio_reativo'].mean() * 100.0):.1f}",
                str(df["fonte_desvio_reativo"].iloc[0]),
            ]
        )

    headers = [
        "Run",
        "Duracao (s)",
        "Dist. 3D (m)",
        "Alt. max (m)",
        "Desvio max (m)",
        "Indice reativo max",
        "Tempo desvio (%)",
        "Fonte evasao",
    ]
    return table_figure(
        headers,
        rows_to_columns(rows, len(headers)),
        "Comparativo entre runs",
        320,
    )


def graph(component_id: str) -> dcc.Graph:
    return dcc.Graph(id=component_id, config=GRAPH_CONFIG)


def graph_grid(*component_ids: str) -> html.Div:
    return html.Div(
        className="graph-grid graph-grid-two",
        children=[graph(component_id) for component_id in component_ids],
    )


def analysis_section(title: str, note_id: str, children: list) -> html.Section:
    return html.Section(
        className="analysis-section",
        children=[html.H2(title), html.Div(id=note_id), *children],
    )


def dashboard_tab(label: str, value: str, children: list) -> dcc.Tab:
    return dcc.Tab(
        label=label,
        value=value,
        className="tab",
        selected_className="tab tab-selected",
        children=[html.Div(className="tab-panel", children=children)],
    )


def overview_tab() -> dcc.Tab:
    return dashboard_tab(
        "Trajetoria e IMU",
        "overview-tab",
        [
            html.Div(id="metrics", className="metrics-grid"),
            html.Div(id="flight-notice"),
            graph_grid("xy-graph", "trajectory-3d"),
            graph_grid("control-graph", "altitude-graph"),
        ],
    )


def reactive_tab() -> dcc.Tab:
    return dashboard_tab(
        "Evasao e voo",
        "reactive-tab",
        [
            graph_grid("reactive-graph", "speed-graph"),
            graph("comparison-table"),
        ],
    )


def depth_tab(depth_options: list[dict], depth_default: str) -> dcc.Tab:
    return dashboard_tab(
        "Depth ground truth",
        "depth-tab",
        [
            html.Div(
                className="selector selector-inline",
                children=[
                    html.Label("Dataset pareado em depth_ground_truth"),
                    dcc.Dropdown(
                        id="depth-run-select",
                        options=depth_options,
                        value=depth_default,
                        clearable=False,
                    ),
                ],
            ),
            html.Div(
                className="notice notice-depth",
                children=(
                    "Esta aba usa o mapa renderizado pelo Gazebo como ground truth "
                    "sintetico para analise/treino futuro. Ele nao representa uma "
                    "depth camera embarcada no drone."
                ),
            ),
            html.Div(id="depth-metrics", className="metrics-grid"),
            graph_grid("depth-timeseries", "depth-imu-graph"),
            graph("depth-ranking-table"),
        ],
    )


def batch_tab() -> dcc.Tab:
    return dashboard_tab(
        "Comparar levas",
        "batch-tab",
        [
            html.Div(id="batch-comparison-notice"),
            html.Div(id="batch-comparison-metrics", className="metrics-grid"),
            analysis_section(
                "Desempenho conforme o conjunto cresce",
                "batch-growth-analysis",
                [graph_grid("batch-metric-facets", "batch-stats-table")],
            ),
            analysis_section(
                "Continuidade ou estabilizacao",
                "batch-stability-analysis",
                [graph_grid("batch-event-rates", "batch-distributions")],
            ),
            analysis_section(
                "Ganho real x variacao do split",
                "batch-split-analysis",
                [graph("batch-mlp-mae")],
            ),
            analysis_section(
                "Alvos e eventos que concentram erros",
                "batch-error-analysis",
                [graph_grid("batch-mlp-table", "batch-event-table")],
            ),
            analysis_section(
                "Resposta para o professor",
                "batch-professor-message",
                [],
            ),
        ],
    )


def model_tab() -> dcc.Tab:
    targets = [
        "delta_depth_p10_m", "delta_depth_p50_m", "delta_depth_p90_m",
        "delta_depth_close_2m_pp", "delta_depth_close_5m_pp", "delta_depth_close_10m_pp",
    ]
    return dashboard_tab(
        "Modelo e explicabilidade",
        "model-tab",
        [
            html.Div(id="model-diagnostic-metrics", className="metrics-grid"),
            analysis_section(
                "Treino, validacao e teste",
                "model-loss-note",
                [graph("model-loss-curves")],
            ),
            analysis_section(
                "Representacao das entradas angulares",
                "model-angular-note",
                [
                    dcc.RadioItems(
                        id="angular-split-select",
                        options=[
                            {"label": "Validacao", "value": "validacao"},
                            {"label": "Teste", "value": "teste"},
                        ],
                        value="validacao",
                        inline=True,
                        className="inline-options",
                    ),
                    graph("model-angular-comparison"),
                ],
            ),
            analysis_section(
                "Entradas relacionadas ao erro",
                "model-gradient-note",
                [
                    html.Div(
                        className="selector selector-inline",
                        children=[
                            html.Label("Alvo explicado"),
                            dcc.Dropdown(
                                id="gradient-target-select",
                                options=[{"label": target, "value": target} for target in targets],
                                value="delta_depth_close_5m_pp",
                                clearable=False,
                            ),
                        ],
                    ),
                    graph("model-gradient-importance"),
                    graph("model-largest-errors"),
                ],
            ),
        ],
    )


def dropdown_options(
    paths: list[Path], label_factory: Callable[[Path], str]
) -> list[dict[str, str]]:
    return [{"label": label_factory(path), "value": str(path)} for path in paths]


def selected_dropdown_value(
    options: list[dict[str, str]], *preferred_values: str | None
) -> str:
    valid_values = {option["value"] for option in options}
    for preferred in preferred_values:
        if preferred in valid_values:
            return str(preferred)
    return options[0]["value"] if options else ""


def layout(log_paths: list[Path], depth_paths: list[Path]) -> html.Div:
    """Monta as abas do dashboard e seus seletores iniciais."""

    flight_options = dropdown_options(log_paths, run_label)
    flight_default = flight_options[0]["value"] if flight_options else ""
    depth_options = dropdown_options(depth_paths, depth_run_label)
    paired_depth = matching_depth_metadata(Path(flight_default), depth_paths) if flight_default else None
    depth_default = (
        str(paired_depth)
        if paired_depth is not None
        else (depth_options[0]["value"] if depth_options else "")
    )

    return html.Div(
        className="app-shell",
        children=[
            html.Div(
                className="topbar",
                children=[
                    html.Div(
                        children=[
                            html.H1("Dashboard de metricas do desvio reativo"),
                            html.P(
                                "Analise organizada por trajetoria/IMU, evasao, "
                                "depth ground truth, comparacao das levas e diagnostico do modelo."
                            ),
                        ]
                    ),
                    html.Div(
                        className="selector selector-top",
                        children=[
                            html.Label("Run de voo"),
                            dcc.Dropdown(
                                id="run-select",
                                options=flight_options,
                                value=flight_default,
                                clearable=False,
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="dashboard-tabs",
                value="overview-tab",
                className="tabs",
                children=[
                    overview_tab(), reactive_tab(), depth_tab(depth_options, depth_default),
                    batch_tab(), model_tab(),
                ],
            ),
            dcc.Interval(id="refresh-data", interval=10000, n_intervals=0),
        ],
    )


INDEX_TEMPLATE = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                * { box-sizing: border-box; }
                body {
                    margin: 0;
                    background: #f4f7f9;
                    color: #172026;
                    font-family: "Segoe UI", Arial, sans-serif;
                }
                .app-shell {
                    width: min(1480px, calc(100vw - 40px));
                    margin: 0 auto;
                    padding: 24px 0 40px;
                }
                .topbar {
                    display: grid;
                    grid-template-columns: minmax(0, 1fr) minmax(260px, 360px);
                    gap: 20px;
                    align-items: end;
                    padding: 12px 0 18px;
                    border-bottom: 1px solid #d5dde3;
                }
                h1 {
                    margin: 0 0 8px;
                    font-size: 30px;
                    line-height: 1.15;
                    letter-spacing: 0;
                }
                p {
                    margin: 0;
                    color: #62717b;
                    font-size: 15px;
                    line-height: 1.45;
                }
                .selector label {
                    display: block;
                    margin-bottom: 6px;
                    color: #43515b;
                    font-size: 13px;
                    font-weight: 650;
                }
                .selector-inline {
                    width: min(420px, 100%);
                    margin: 18px 0 4px;
                }
                .selector-top {
                    width: 100%;
                    align-self: end;
                }
                .tabs {
                    margin-top: 18px;
                }
                .tab {
                    border: 1px solid #d5dde3 !important;
                    border-bottom: none !important;
                    background: #eef3f6 !important;
                    color: #43515b !important;
                    padding: 12px 18px !important;
                    font-weight: 650;
                }
                .tab-selected {
                    background: #ffffff !important;
                    color: #172026 !important;
                    border-top: 3px solid #0f766e !important;
                }
                .tab-panel {
                    background: #ffffff;
                    border: 1px solid #d5dde3;
                    border-top: none;
                    padding: 1px 14px 18px;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(6, minmax(150px, 1fr));
                    gap: 12px;
                    margin: 18px 0;
                }
                .metric-card {
                    background: #ffffff;
                    border: 1px solid #d5dde3;
                    border-radius: 8px;
                    padding: 14px 14px 12px;
                    min-height: 104px;
                }
                .metric-label {
                    color: #62717b;
                    font-size: 12px;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 0;
                }
                .metric-value {
                    margin-top: 8px;
                    font-size: 27px;
                    font-weight: 750;
                    line-height: 1.05;
                }
                .metric-detail {
                    margin-top: 8px;
                    color: #62717b;
                    font-size: 12px;
                    line-height: 1.25;
                }
                .notice {
                    margin: 0 0 16px;
                    padding: 11px 14px;
                    border-left: 4px solid #0f766e;
                    background: #edf7f5;
                    color: #244a45;
                    font-size: 13px;
                    line-height: 1.4;
                }
                .notice-depth {
                    border-left-color: #7c3aed;
                    background: #f4f0ff;
                    color: #3b2f63;
                }
                .notice-ok {
                    border-left-color: #1b8a5a;
                    background: #edf8f1;
                    color: #1f5134;
                }
                .analysis-section {
                    margin-top: 26px;
                    padding-top: 20px;
                    border-top: 1px solid #d5dde3;
                }
                .analysis-section h2 {
                    margin: 0 0 10px;
                    font-size: 19px;
                    line-height: 1.25;
                    letter-spacing: 0;
                }
                .analysis-note {
                    margin-bottom: 14px;
                    padding: 12px 14px;
                    border-left: 4px solid #246bfe;
                    background: #f3f7ff;
                    color: #26394f;
                }
                .analysis-note h3 {
                    margin: 0 0 6px;
                    font-size: 14px;
                    letter-spacing: 0;
                }
                .analysis-note p {
                    margin: 5px 0 0;
                    color: inherit;
                    font-size: 13px;
                    line-height: 1.45;
                }
                .analysis-note-ok {
                    border-left-color: #1b8a5a;
                    background: #edf8f1;
                    color: #1f5134;
                }
                .analysis-note-warning {
                    border-left-color: #c2410c;
                    background: #fff6ed;
                    color: #6b3416;
                }
                .analysis-note-professor {
                    border-left-color: #0f766e;
                    background: #edf7f5;
                    color: #244a45;
                }
                .graph-grid {
                    display: grid;
                    gap: 14px;
                    margin-bottom: 14px;
                }
                .graph-grid-two {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .graph-grid-three {
                    grid-template-columns: repeat(3, minmax(0, 1fr));
                }
                .dash-graph {
                    background: #ffffff;
                    border: 1px solid #d5dde3;
                    border-radius: 8px;
                    overflow: hidden;
                }
                .inline-options {
                    display: flex;
                    gap: 18px;
                    margin: 8px 0 14px;
                    color: #43515b;
                    font-size: 14px;
                }
                .inline-options label { margin-right: 16px; }
                @media (max-width: 1180px) {
                    .metrics-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
                    .graph-grid-three { grid-template-columns: 1fr; }
                }
                @media (max-width: 820px) {
                    .app-shell { width: min(100vw - 24px, 760px); padding-top: 14px; }
                    .topbar { grid-template-columns: 1fr; align-items: start; }
                    h1 { font-size: 24px; }
                    .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                    .graph-grid-two { grid-template-columns: 1fr; }
                }
                @media (max-width: 520px) {
                    .metrics-grid { grid-template-columns: 1fr; }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """


def register_callbacks(app: Dash) -> None:
    """Registra callbacks de atualizacao das abas de voo, depth e comparacao."""

    @app.callback(
        Output("run-select", "options"),
        Output("run-select", "value"),
        Input("refresh-data", "n_intervals"),
        State("run-select", "value"),
    )
    def refresh_flight_run_options(_n_intervals: int, selected_path: str):
        current_paths = list_log_files()
        options = dropdown_options(current_paths, run_label)
        return options, selected_dropdown_value(options, selected_path)

    @app.callback(
        Output("depth-run-select", "options"),
        Output("depth-run-select", "value"),
        Input("refresh-data", "n_intervals"),
        Input("run-select", "value"),
        State("depth-run-select", "value"),
    )
    def refresh_depth_run_options(_n_intervals: int, selected_log_path: str, selected_path: str):
        current_paths = list_depth_metadata_files()
        options = dropdown_options(current_paths, depth_run_label)
        paired = None
        if selected_log_path:
            paired_path = matching_depth_metadata(Path(selected_log_path), current_paths)
            paired = str(paired_path) if paired_path is not None else None
        return options, selected_dropdown_value(options, paired, selected_path)

    @app.callback(
        Output("metrics", "children"),
        Output("flight-notice", "children"),
        Output("xy-graph", "figure"),
        Output("trajectory-3d", "figure"),
        Output("reactive-graph", "figure"),
        Output("control-graph", "figure"),
        Output("speed-graph", "figure"),
        Output("altitude-graph", "figure"),
        Output("comparison-table", "figure"),
        Input("run-select", "value"),
    )
    def update_flight_dashboard(selected_path: str):
        current_log_paths = list_log_files()
        path = Path(selected_path) if selected_path else (current_log_paths[0] if current_log_paths else Path())
        df = load_run(path)
        if df.empty:
            empty_fig = apply_layout(go.Figure(), "Sem dados")
            return (
                [metric_card("Sem dados", "-", "Nenhum CSV encontrado.")],
                flight_notice(df),
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                comparison_table(current_log_paths),
            )

        return (
            run_summary(df),
            flight_notice(df),
            figure_xy(df),
            figure_3d(df),
            figure_reactive_timeseries(df),
            figure_control(df),
            figure_speed(df),
            figure_altitude(df),
            comparison_table(current_log_paths),
        )

    @app.callback(
        Output("depth-metrics", "children"),
        Output("depth-timeseries", "figure"),
        Output("depth-imu-graph", "figure"),
        Output("depth-ranking-table", "figure"),
        Input("depth-run-select", "value"),
    )
    def update_depth_dashboard(selected_path: str):
        current_depth_paths = list_depth_metadata_files()
        depth_path = Path(selected_path) if selected_path else (current_depth_paths[0] if current_depth_paths else None)
        depth_df = load_depth_metadata(str(depth_path)) if depth_path is not None else pd.DataFrame()
        if depth_df.empty:
            empty_fig = apply_layout(go.Figure(), "Sem depth ground truth")
            return (
                [metric_card("Depth GT", "-", "Nenhuma run em datasets/depth_ground_truth/.")],
                empty_fig,
                empty_fig,
                empty_fig,
            )

        return (
            depth_summary(depth_df, depth_path),
            figure_depth_timeseries(depth_df),
            figure_depth_imu(depth_df),
            figure_depth_ranking(depth_df),
        )

    @app.callback(
        Output("batch-comparison-notice", "children"),
        Output("batch-comparison-metrics", "children"),
        Output("batch-metric-facets", "figure"),
        Output("batch-event-rates", "figure"),
        Output("batch-distributions", "figure"),
        Output("batch-mlp-mae", "figure"),
        Output("batch-mlp-table", "figure"),
        Output("batch-stats-table", "figure"),
        Output("batch-event-table", "figure"),
        Output("batch-growth-analysis", "children"),
        Output("batch-stability-analysis", "children"),
        Output("batch-split-analysis", "children"),
        Output("batch-error-analysis", "children"),
        Output("batch-professor-message", "children"),
        Input("refresh-data", "n_intervals"),
    )
    def update_batch_comparison(_n_intervals: int):
        intervals_df = load_all_depth_intervals()
        stats_df = summarize_batch_intervals(intervals_df)
        batch_df = batch_interval_frames(intervals_df)
        mlp_df = reference_mlp_df()
        event_df = reference_event_df()
        notes = batch_analysis_notes(stats_df, mlp_df, event_df)

        return (
            batch_notice(intervals_df),
            batch_summary_cards(stats_df, mlp_df),
            figure_batch_metric_facets(stats_df),
            figure_batch_event_rates(intervals_df),
            figure_batch_distributions(batch_df),
            figure_mlp_reference_mae(mlp_df),
            figure_mlp_reference_table(mlp_df),
            figure_batch_stats_table(stats_df),
            figure_event_reference_table(event_df),
            *notes,
        )

    @app.callback(
        Output("model-diagnostic-metrics", "children"),
        Output("model-loss-curves", "figure"),
        Output("model-angular-comparison", "figure"),
        Output("model-gradient-importance", "figure"),
        Output("model-largest-errors", "figure"),
        Output("model-loss-note", "children"),
        Output("model-angular-note", "children"),
        Output("model-gradient-note", "children"),
        Input("refresh-data", "n_intervals"),
        Input("angular-split-select", "value"),
        Input("gradient-target-select", "value"),
    )
    def update_model_diagnostics(
        _n_intervals: int, angular_split: str, gradient_target: str
    ):
        data = load_model_diagnostics()
        loss_df = data["loss"]
        angular_df = data["angular"]
        importance_df = (
            data["grouped_importance"]
            if not data["grouped_importance"].empty
            else data["importance"]
        )
        errors_df = data["errors"]

        validation = loss_df[loss_df["split"] == "validacao"]
        best_epoch = int(validation.loc[validation["mse_padronizado"].idxmin(), "epoca"]) if not validation.empty else 0
        stopped_epoch = int(loss_df["epoca"].max()) if not loss_df.empty else 0
        loss_note = analysis_note(
            "Como interpretar",
            [
                f"O early stopping escolheu a epoca {best_epoch} pela menor loss de validacao e o treino terminou na epoca {stopped_epoch}.",
                "A curva de teste e apenas diagnostica: ela nao participa da escolha da epoca nem dos pesos restaurados.",
            ],
            "warning",
        )

        angular_note = analysis_note(
            "Leitura do experimento angular",
            [
                "Valor angular e seno/cosseno usam as mesmas runs, splits, semente, dropout e early stopping.",
                "A representacao ciclica reduziu a importancia conjunta de algumas entradas angulares, mas nao melhorou de forma consistente o MAE no teste fixo.",
            ],
        )

        zero_gradients = 0
        if not errors_df.empty:
            cols = ["importancia_1_pct", "importancia_2_pct", "importancia_3_pct"]
            zero_gradients = int((errors_df[cols].fillna(0.0).abs().sum(axis=1) <= 1e-12).sum())
        gradient_note = analysis_note(
            "Limite da explicacao local",
            [
                "As importancias sao calculadas somente na validacao fixa; o teste permanece reservado para avaliacao final.",
                f"Em {zero_gradients} dos maiores erros, o gradiente local e zero. Nesses casos o dashboard informa que nao existe ranking valido, em vez de exibir nomes arbitrarios com 0%.",
            ],
            "warning" if zero_gradients else "ok",
        )

        return (
            model_diagnostic_cards(data),
            figure_loss_curves(loss_df),
            figure_angular_comparison(angular_df, angular_split or "validacao"),
            figure_gradient_importance(importance_df, gradient_target or "delta_depth_close_5m_pp"),
            figure_largest_gradient_errors(errors_df),
            loss_note,
            angular_note,
            gradient_note,
        )

def create_app() -> Dash:
    """Cria e configura a aplicacao Dash pronta para execucao."""

    app = Dash(__name__)
    app.title = "Metricas do desvio reativo"
    app.layout = layout(list_log_files(), list_depth_metadata_files())
    app.index_string = INDEX_TEMPLATE
    register_callbacks(app)
    return app


if __name__ == "__main__":
    create_app().run(debug=False, host="127.0.0.1", port=8050)
