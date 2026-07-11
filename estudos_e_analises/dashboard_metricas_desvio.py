from __future__ import annotations

import math
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dcc, html


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"


def find_depth_ground_truth_dir() -> Path:
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

FIRST_BATCH_RUN_IDS = {
    "run_20260521_161617",
    "run_20260521_174507",
    "run_20260521_174545",
    "run_20260521_175244",
}
FIRST_BATCH_LABEL = "Primeira leva (4 runs)"
NEW_BATCH_LABEL = "Runs novas"
ALL_BATCH_LABEL = "Todas as runs atuais"

REFERENCE_BATCH_STATS = [
    {
        "leva": FIRST_BATCH_LABEL,
        "runs": 4,
        "intervalos": 205,
        "dt_s_mean": 0.055239,
        "depth_age_s_mean": 0.057249,
        "flow_valid_points_mean": 82.717073,
        "flow_track_retention_pct_mean": 97.636475,
        "flow_mag_p90_px_mean": 31.688473,
        "radial_flow_p90_px_mean": 26.264482,
        "delta_depth_close_5m_pp_mean": 0.075290,
        "delta_depth_close_5m_pp_std": 2.976323,
        "event_rate_pct": 49 / 205 * 100.0,
    },
    {
        "leva": ALL_BATCH_LABEL,
        "runs": 9,
        "intervalos": 382,
        "dt_s_mean": 0.055602,
        "depth_age_s_mean": 0.057173,
        "flow_valid_points_mean": 81.848168,
        "flow_track_retention_pct_mean": 97.476318,
        "flow_mag_p90_px_mean": 34.574924,
        "radial_flow_p90_px_mean": 28.885937,
        "delta_depth_close_5m_pp_mean": 0.075447,
        "delta_depth_close_5m_pp_std": 2.405984,
        "event_rate_pct": 27.2,
    },
]

REFERENCE_MLP_TEST = [
    ("Primeira leva", "MLP", "delta_depth_close_10m_pp", 3.299956, 13.520034),
    ("Primeira leva", "Media treino", "delta_depth_close_10m_pp", 0.593824, 1.484924),
    ("Primeira leva", "MLP", "delta_depth_close_2m_pp", 4.059252, 17.529010),
    ("Primeira leva", "Media treino", "delta_depth_close_2m_pp", 1.592609, 6.444213),
    ("Primeira leva", "MLP", "delta_depth_close_5m_pp", 3.907923, 14.620979),
    ("Primeira leva", "Media treino", "delta_depth_close_5m_pp", 1.445566, 4.530197),
    ("Primeira leva", "MLP", "delta_depth_p10_m", 0.171427, 0.591393),
    ("Primeira leva", "Media treino", "delta_depth_p10_m", 0.097618, 0.375058),
    ("Primeira leva", "MLP", "delta_depth_p50_m", 0.170319, 0.350195),
    ("Primeira leva", "Media treino", "delta_depth_p50_m", 0.251893, 0.872944),
    ("Primeira leva", "MLP", "delta_depth_p90_m", 1.863624, 4.800333),
    ("Primeira leva", "Media treino", "delta_depth_p90_m", 0.847543, 2.119142),
    ("Todas atuais", "MLP", "delta_depth_close_10m_pp", 1.038949, 1.919759),
    ("Todas atuais", "Media treino", "delta_depth_close_10m_pp", 0.548046, 0.957294),
    ("Todas atuais", "MLP", "delta_depth_close_2m_pp", 1.753198, 3.781931),
    ("Todas atuais", "Media treino", "delta_depth_close_2m_pp", 0.733926, 1.388800),
    ("Todas atuais", "MLP", "delta_depth_close_5m_pp", 2.027266, 4.154152),
    ("Todas atuais", "Media treino", "delta_depth_close_5m_pp", 0.798785, 1.540230),
    ("Todas atuais", "MLP", "delta_depth_p10_m", 0.066686, 0.141854),
    ("Todas atuais", "Media treino", "delta_depth_p10_m", 0.033849, 0.065558),
    ("Todas atuais", "MLP", "delta_depth_p50_m", 0.263102, 0.563510),
    ("Todas atuais", "Media treino", "delta_depth_p50_m", 0.122928, 0.255040),
    ("Todas atuais", "MLP", "delta_depth_p90_m", 1.298594, 3.082840),
    ("Todas atuais", "Media treino", "delta_depth_p90_m", 0.591928, 1.817458),
]

REFERENCE_EVENT_RESULTS = [
    ("Classificador evento", "balanced accuracy", 0.847078),
    ("Classificador evento", "precisao", 0.766667),
    ("Classificador evento", "recall", 0.821429),
    ("Classificador evento", "F1", 0.793103),
    ("Regressao pos-gate", "MAE zero delta", 0.736902),
    ("Regressao pos-gate", "MAE MLP evento + MLP delta", 1.055266),
    ("Regressao pos-gate", "MAE oracle evento + MLP delta", 0.932840),
]

REFERENCE_RUN_EVENT_STATS = [
    ("run_20260521_161617", 36, 11, 0.305556),
    ("run_20260521_174507", 41, 13, 0.317073),
    ("run_20260521_174545", 47, 14, 0.297872),
    ("run_20260521_175244", 81, 11, 0.135802),
    ("run_20260624_211120", 23, 7, 0.304348),
    ("run_20260624_211302", 34, 7, 0.205882),
    ("run_20260624_224839", 37, 12, 0.324324),
    ("run_20260624_230625", 34, 8, 0.235294),
    ("run_20260624_230757", 49, 21, 0.428571),
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
}


def _empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "x",
            "y",
            "z",
            "roll_speed",
            "pitch_speed",
            "yaw_speed",
        ]
    )


def run_datetime(path: Path) -> datetime | None:
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
    """
    Lista os logs de voo novos do mais recente para o mais antigo.

    As runs atuais ficam em logs/voo_teste_*/manifest.json. A pasta logs/old guarda CSVs
    historicos e e ignorada para evitar misturar formatos antigos com os memmaps novos.

    Fonte:
    [Python pathlib] https://docs.python.org/3/library/pathlib.html
    """

    paired_logs = [log_path for log_path, _depth_path in paired_log_depth_files()]
    if paired_logs:
        return sorted(paired_logs, key=lambda path: run_datetime(path) or datetime.min, reverse=True)
    manifests = _raw_log_manifest_files()
    if manifests:
        return sorted(manifests, key=lambda path: run_datetime(path) or datetime.min, reverse=True)
    return sorted(LOG_DIR.glob("voo_teste_*.csv"), reverse=True)


def list_depth_metadata_files() -> list[Path]:
    """
    Lista os arquivos novos de depth/flow produzidos pelo dataset de profundidade do Gazebo.

    O formato atual usa datasets/depth_ground_truth/run_*/manifest.json com intervalos em
    memmap. O formato antigo metadata.csv continua aceito como fallback, mas a pasta old/
    e ignorada quando existir.

    Fontes:
    [Python pathlib] https://docs.python.org/3/library/pathlib.html
    [Gazebo DepthCamera] https://gazebosim.org/api/rendering/7/classgz_1_1rendering_1_1DepthCamera.html
    [Artigo - RealTimeMonocular2022] https://doi.org/10.1109/TITS.2022.3160741
    """

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


def load_depth_interval_run(run_dir: Path) -> pd.DataFrame:
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

    df["run_id"] = run_dir.name
    df["ordem_intervalo"] = np.arange(len(df))
    return df


def enrich_depth_interval_dashboard(df: pd.DataFrame) -> pd.DataFrame:
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


def load_all_depth_intervals() -> pd.DataFrame:
    frames = []
    for run_dir in list_depth_interval_runs():
        try:
            df = load_depth_interval_run(run_dir)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def batch_label_for_run(run_id: str) -> str:
    return FIRST_BATCH_LABEL if run_id in FIRST_BATCH_RUN_IDS else NEW_BATCH_LABEL


def run_label(path: Path) -> str:
    if path.name == "manifest.json":
        return path.parent.name.replace("voo_teste_", "")
    if path.is_dir():
        return path.name.replace("voo_teste_", "")
    return path.stem.replace("voo_teste_", "")


def depth_run_label(path: Path) -> str:
    """
    Retorna o identificador curto de uma run de depth ground truth.

    O metadata.csv fica dentro de uma pasta run_<timestamp>. Esta funcao preserva esse
    timestamp para que a run possa ser pareada visualmente com os logs voo_teste_<timestamp>.

    Fonte:
    [Python pathlib] https://docs.python.org/3/library/pathlib.html
    """

    return path.parent.name.replace("run_", "")


def matching_depth_metadata(log_path: Path, depth_paths: list[Path]) -> Path | None:
    """
    Escolhe a run de depth mais compativel com o log selecionado no dashboard.

    Quando o timestamp do log aparece no nome da pasta run_<timestamp>, esse metadata.csv e
    usado. Caso contrario, o dashboard usa a run de depth mais recente para ainda expor a
    analise dos dados novos sem quebrar a visualizacao.

    Fonte:
    [Python pathlib] https://docs.python.org/3/library/pathlib.html
    """

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
    """
    Resolve o caminho de um arquivo NPY de profundidade no Windows ou no WSL.

    O CSV pode guardar caminhos absolutos do ambiente WSL, como /home/prograf4080/..., que
    nao existem quando a analise e aberta pelo Windows. Nesses casos, a funcao aproveita o
    nome do arquivo e reconstrui o caminho local dentro de run_dir/depth_m.

    Fontes:
    [Python pathlib] https://docs.python.org/3/library/pathlib.html
    [NumPy NPY format] https://numpy.org/doc/stable/reference/generated/numpy.save.html
    """

    original = Path(str(depth_path_value))
    if original.exists():
        return original
    return run_dir / "depth_m" / original.name


@lru_cache(maxsize=8)
def load_depth_metadata(metadata_path_value: str) -> pd.DataFrame:
    """
    Carrega e enriquece uma run de depth ground truth para uso no dashboard.

    Alem dos campos do metadata.csv, a funcao le os mapas .npy quando eles existem e calcula
    percentis de profundidade e porcentagem de pixels mais proximos que 2 m, 5 m e 10 m.
    Esses indicadores ajudam a transformar o z-buffer/depth do Gazebo em um alvo analisavel
    para a futura rede monocular, sem mudar o controlador em tempo real.

    Fontes:
    [Pandas read_csv] https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    [NumPy load] https://numpy.org/doc/stable/reference/generated/numpy.load.html
    [Artigo - RealTimeMonocular2022] https://doi.org/10.1109/TITS.2022.3160741
    [Artigo - Vyas2022] https://doi.org/10.48550/arXiv.2205.01399
    """

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
    """
    Indica se o log de voo contem metricas reais da evasao visual.

    Runs novas gravadas por main.py incluem obstacle_risk, avoid_lateral_body e avoid_brake.
    Quando esses campos existem, o dashboard deixa de usar o proxy derivado de odometria e
    passa a mostrar os sinais calculados diretamente pelo controlador.

    Fontes:
    [Pandas DataFrame columns] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html
    [Python set] https://docs.python.org/3/library/stdtypes.html#set
    """

    required = {"obstacle_risk", "avoid_lateral_body", "avoid_brake"}
    return required.issubset(set(df.columns))


def add_real_avoidance_metrics(df: pd.DataFrame) -> None:
    """
    Padroniza as metricas reais de evasao para os graficos do dashboard.

    A funcao cria colunas genericas usadas pela visualizacao: indice_desvio_reativo,
    evento_desvio_reativo, lateral_reativo_m_s2 e freio_reativo. Assim, os graficos podem
    tratar logs novos e antigos com a mesma interface de dados.

    Fontes:
    [Pandas to_numeric] https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
    [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
    """

    for col in ("obstacle_risk", "avoid_lateral_body", "avoid_brake", "evasao_visual_ativa", "pan_comp_delta_rad"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["indice_desvio_reativo"] = df["obstacle_risk"].clip(0.0, 1.0)
    df["evento_desvio_reativo"] = df["indice_desvio_reativo"] > OBSTACLE_RISK_THRESHOLD
    df["lateral_reativo_m_s2"] = df["avoid_lateral_body"]
    df["freio_reativo"] = df["avoid_brake"].clip(0.0, 1.0)
    df["fonte_desvio_reativo"] = "metricas reais"


def load_flight_interval_memmap(manifest_path: Path) -> pd.DataFrame:
    """
    Carrega o formato atual de logs/voo_teste_*/manifest.json.

    O logger novo salva apenas variacoes entre amostras. Para visualizacao, o dashboard
    reconstrui uma trajetoria relativa acumulando os deltas de deslocamento e usa os deltas
    de velocidade angular como sinais de IMU, igual ao notebook de metricas.

    Fontes:
    [NumPy load] https://numpy.org/doc/stable/reference/generated/numpy.load.html
    [Pandas DataFrame] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    """

    if not manifest_path.exists():
        return _empty_dataframe()

    with open(manifest_path, encoding="utf-8") as fp:
        manifest = json.load(fp)

    if manifest.get("schema_version") != "flight_interval_memmap_v1":
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

    df["timestamp"] = df["dt_s"].fillna(0.0).cumsum()
    df["tempo_s"] = df["timestamp"]
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
    df["vel_horizontal"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2)
    df["vel_3d"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2 + df["v_z"] ** 2)
    df["vel_horizontal_suave"] = df["vel_horizontal"].rolling(15, min_periods=1, center=True).median()
    df["angular_norm"] = np.sqrt(df["roll_speed"] ** 2 + df["pitch_speed"] ** 2 + df["yaw_speed"] ** 2)
    df["angular_norm_suave"] = df["angular_norm"].rolling(15, min_periods=1, center=True).median()

    rota = build_route(df)
    add_route_metrics(df, rota)

    risk_delta = df["delta_obstacle_risk"].fillna(0.0) if "delta_obstacle_risk" in df else pd.Series(0.0, index=df.index)
    lateral_delta = df["delta_avoid_lateral_body"].fillna(0.0) if "delta_avoid_lateral_body" in df else pd.Series(0.0, index=df.index)
    brake_delta = df["delta_avoid_brake"].fillna(0.0) if "delta_avoid_brake" in df else pd.Series(0.0, index=df.index)
    df["indice_desvio_reativo"] = normalize(risk_delta.abs())
    df["evento_desvio_reativo"] = df["indice_desvio_reativo"] > OBSTACLE_RISK_THRESHOLD
    df["lateral_reativo_m_s2"] = lateral_delta
    df["freio_reativo"] = normalize(brake_delta.abs())
    df["fonte_desvio_reativo"] = "deltas memmap"
    df["run_id"] = manifest_path.parent.name
    return df


def load_run(path: Path) -> pd.DataFrame:
    """
    Carrega um CSV de voo e calcula metricas derivadas para o dashboard.

    Logs novos usam diretamente obstacle_risk, avoid_lateral_body e avoid_brake gravados
    pelo DataLogger. Logs antigos continuam recebendo um proxy baseado em desvio lateral,
    frenagem relativa e atividade angular para manter comparabilidade historica.

    Fontes:
    [Pandas read_csv] https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    [NumPy hypot] https://numpy.org/doc/stable/reference/generated/numpy.hypot.html
    [PX4 VehicleOdometry] https://docs.px4.io/main/en/msg_docs/VehicleOdometry
    """

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
    df["vel_horizontal"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2)
    df["vel_3d"] = np.sqrt(df["v_x"] ** 2 + df["v_y"] ** 2 + df["v_z"] ** 2)
    df["vel_horizontal_suave"] = df["vel_horizontal"].rolling(15, min_periods=1, center=True).median()
    df["angular_norm"] = np.sqrt(
        df["roll_speed"] ** 2 + df["pitch_speed"] ** 2 + df["yaw_speed"] ** 2
    )
    df["angular_norm_suave"] = df["angular_norm"].rolling(15, min_periods=1, center=True).median()

    rota = build_route(df)
    add_route_metrics(df, rota)
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


def normalize(series: pd.Series, high_quantile: float = 0.95) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
    scale = float(clean.quantile(high_quantile))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = float(clean.max())
    if not math.isfinite(scale) or scale <= 1e-9:
        return clean * 0.0
    return (clean / scale).clip(0.0, 1.0)


def add_reactive_proxy(df: pd.DataFrame) -> None:
    """
    Calcula um proxy de evasao reativa para logs antigos sem metricas visuais reais.

    O proxy combina velocidade lateral ao corredor, atividade angular, desvio da rota e
    frenagem relativa. Ele fica marcado como "proxy estimado" para o dashboard avisar que a
    run foi coletada antes do logger salvar obstacle_risk, avoid_lateral_body e avoid_brake.

    Fontes:
    [Pandas rolling] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
    [PX4 VehicleOdometry] https://docs.px4.io/main/en/msg_docs/VehicleOdometry
    """

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
    """
    Cria os cards principais da aba de metricas de voo.

    Para logs novos, o card de risco usa obstacle_risk real gravado pelo controlador. Para
    logs antigos, o mesmo espaco mostra o proxy estimado e informa a fonte no detalhe.

    Fontes:
    [Pandas Series max] https://pandas.pydata.org/docs/reference/api/pandas.Series.max.html
    [PX4 VehicleOdometry] https://docs.px4.io/main/en/msg_docs/VehicleOdometry
    """

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
        metric_card("Desvio rota max.", fmt_number(float(df["desvio_rota_m"].max()), " m", 2), "distancia lateral ao segmento"),
        metric_card(risk_label, fmt_number(max_reactive, "", 2), f"{source}; limiar {OBSTACLE_RISK_THRESHOLD:.2f}"),
        metric_card("Tempo em desvio", fmt_number(event_share, "%", 1), "indice acima do limiar"),
    ]


def flight_notice(df: pd.DataFrame) -> html.Div:
    """
    Gera o aviso da aba de voo conforme a qualidade do log selecionado.

    Runs novas exibem uma confirmacao de que obstacle_risk, avoid_lateral_body e avoid_brake
    vieram do CSV. Runs antigas mantem o aviso de que o dashboard precisou estimar um proxy,
    apontando exatamente o que falta coletar.

    Fontes:
    [Dash HTML components] https://dash.plotly.com/dash-html-components
    [Python all] https://docs.python.org/3/library/functions.html#all
    """

    if not df.empty and has_real_avoidance_metrics(df):
        text = (
            "Esta run contem obstacle_risk, avoid_lateral_body e avoid_brake gravados no CSV. "
            "Os graficos de evasao usam as metricas reais calculadas pelo controlador."
        )
        return html.Div(text, className="notice notice-ok")

    if not df.empty and "fonte_desvio_reativo" in df.columns and str(df["fonte_desvio_reativo"].iloc[0]) == "deltas memmap":
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
    """
    Monta cards resumidos para a run de profundidade do Gazebo.

    Os cards destacam sincronizacao RGB/depth, profundidade central da cena e fracao de
    pixels proximos. A leitura permanece diagnostica: esses dados sao ground truth do
    simulador para analise e treino futuro, nao uma depth camera embarcada.

    Fontes:
    [Gazebo DepthCamera] https://gazebosim.org/api/rendering/7/classgz_1_1rendering_1_1DepthCamera.html
    [Pandas DataFrame] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    [Artigo - RealTimeMonocular2022] https://doi.org/10.1109/TITS.2022.3160741
    """

    if df.empty or metadata_path is None:
        return [metric_card("Depth GT", "-", "Nenhuma run em datasets/depth_ground_truth/.")]

    if "source_format" in df.columns and str(df["source_format"].iloc[0]) == "depth_interval_memmap":
        event_rate = 0.0
        if "delta_depth_close_5m_pp" in df.columns:
            event_rate = float((df["delta_depth_close_5m_pp"].abs() > 0.5).mean() * 100.0)
        return [
            metric_card("Dataset pareado", depth_run_label(metadata_path), f"{len(df)} intervalos depth/flow"),
            metric_card("dt visual medio", fmt_number(float(df["dt_s"].mean() * 1000.0), " ms", 1), "entre atualizacoes visuais"),
            metric_card("RGB-depth medio", fmt_number(float(df["depth_age_s"].mean() * 1000.0), " ms", 1), "idade do depth usado"),
            metric_card("Flow valido", fmt_number(float(df["flow_valid_points"].mean()), " pts", 1), "media por intervalo"),
            metric_card("Eventos prox.", fmt_number(event_rate, "%", 1), "|delta pixels < 5m| > 0,5 p.p."),
            metric_card("Pan comp.", fmt_number(float(df["pan_comp_delta_rad"].abs().max()), " rad", 4), "maximo absoluto"),
        ]

    pan_detail = "compensacao ausente nesta run"
    if df["pan_comp_delta_rad"].notna().any():
        pan_detail = f"pan mediano {float(df['pan_comp_delta_rad'].abs().median()):.4f} rad"

    return [
        metric_card("Depth run", depth_run_label(metadata_path), f"{len(df)} pares RGB/depth"),
        metric_card("Sincronia media", fmt_number(float(df["depth_age_s"].mean() * 1000.0), " ms", 1), "RGB x depth"),
        metric_card("Depth P10 med.", fmt_number(float(df["depth_p10_m"].median()), " m", 2), "percentil 10 por frame"),
        metric_card("Pixels < 5 m", fmt_number(float(df["depth_close_5m_pct"].median()), "%", 1), "mediana da area valida"),
        metric_card("Giro mediano", fmt_number(float(df["gyro_norm"].median()), " rad/s", 3), "norma do giroscopio"),
        metric_card("Pan comp.", fmt_number(float(df["pan_comp_delta_rad"].abs().max()), " rad", 4), pan_detail),
    ]


def apply_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
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


def figure_xy(df: pd.DataFrame) -> go.Figure:
    """
    Mostra a trajetoria XY colorida pelo indice de evasao reativa.

    O indice vem de obstacle_risk real em logs novos ou do proxy em logs antigos. A rota
    planejada continua sendo reconstruida a partir dos waypoints relativos do controlador.

    Fontes:
    [Plotly scatter] https://plotly.com/python/line-and-scatter/
    [PX4 Offboard Mode] https://docs.px4.io/main/en/flight_modes/offboard
    """

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
    """
    Plota os sinais de evasao reativa ao longo do tempo.

    Em logs novos, a figura mostra obstacle_risk, avoid_lateral_body e avoid_brake reais.
    Em logs antigos, mostra o indice de proxy e sinais derivados equivalentes para manter a
    leitura temporal da manobra.

    Fontes:
    [Plotly line charts] https://plotly.com/python/line-charts/
    [OpenCV Optical Flow] https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
    """

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
    """
    Cria o grafico temporal de profundidade e pixels proximos da run do Gazebo.

    O eixo principal mostra a profundidade media e os percentis P10/P50. O eixo secundario
    mostra a porcentagem de pixels validos com profundidade menor que 5 m, que funciona como
    um indicador simples de ocupacao proxima no campo visual.

    Fontes:
    [Plotly multiple axes] https://plotly.com/python/multiple-axes/
    [NumPy percentile] https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    [Artigo - Vyas2022] https://doi.org/10.48550/arXiv.2205.01399
    """

    if "source_format" in df.columns and str(df["source_format"].iloc[0]) == "depth_interval_memmap":
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
    """
    Relaciona proximidade visual do depth ground truth com atividade inercial.

    Cada ponto representa um par RGB/depth. A leitura ajuda a identificar trechos em que a
    cena estava proxima e o drone tambem girava bastante, justamente os casos em que a
    compensacao IMU antes do fluxo optico tende a ser mais importante.

    Fontes:
    [Plotly scatter] https://plotly.com/python/line-and-scatter/
    [PX4 SensorCombined] https://docs.px4.io/main/en/msg_docs/SensorCombined.html
    [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
    """

    if "source_format" in df.columns and str(df["source_format"].iloc[0]) == "depth_interval_memmap":
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
    """
    Cria uma tabela com os frames mais interessantes da run de depth.

    O ranking prioriza frames com maior porcentagem de pixels abaixo de 5 m e maior giro
    inercial. Esses casos tendem a ser uteis para inspecionar colisao iminente, paralaxe e
    efeito da compensacao IMU antes do treino monocular.

    Fontes:
    [Plotly Table] https://plotly.com/python/table/
    [Pandas sort_values] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
    [Artigo - Tarrio2015] https://doi.org/10.1109/iccv.2015.87
    """

    if "source_format" in df.columns and str(df["source_format"].iloc[0]) == "depth_interval_memmap":
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

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=[label for _, label in cols],
                        fill_color="#ece7ff",
                        align="left",
                        font=dict(size=13),
                    ),
                    cells=dict(values=values, fill_color="#ffffff", align="left", height=28),
                )
            ]
        )
        return apply_layout(fig, "Intervalos mais informativos de depth/flow", 360)

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

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[label for _, label in cols],
                    fill_color="#ece7ff",
                    align="left",
                    font=dict(size=13),
                ),
                cells=dict(values=values, fill_color="#ffffff", align="left", height=28),
            )
        ]
    )
    return apply_layout(fig, "Frames prioritarios para inspecao e treino", 360)


def batch_interval_frames(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    frames = []
    first_df = df[df["run_id"].isin(FIRST_BATCH_RUN_IDS)].copy()
    new_df = df[~df["run_id"].isin(FIRST_BATCH_RUN_IDS)].copy()
    all_df = df.copy()
    for label, part in [
        (FIRST_BATCH_LABEL, first_df),
        (NEW_BATCH_LABEL, new_df),
        (ALL_BATCH_LABEL, all_df),
    ]:
        if part.empty:
            continue
        part = part.copy()
        part["leva"] = label
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_batch_intervals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(REFERENCE_BATCH_STATS)

    rows = []
    for label, part in [
        (FIRST_BATCH_LABEL, df[df["run_id"].isin(FIRST_BATCH_RUN_IDS)]),
        (NEW_BATCH_LABEL, df[~df["run_id"].isin(FIRST_BATCH_RUN_IDS)]),
        (ALL_BATCH_LABEL, df),
    ]:
        if part.empty:
            continue
        event_rate = 0.0
        if "delta_depth_close_5m_pp" in part.columns:
            event_rate = float((part["delta_depth_close_5m_pp"].abs() > 0.5).mean() * 100.0)
        rows.append(
            {
                "leva": label,
                "runs": int(part["run_id"].nunique()),
                "intervalos": int(len(part)),
                "dt_s_mean": float(part["dt_s"].mean()) if "dt_s" in part else np.nan,
                "depth_age_s_mean": float(part["depth_age_s"].mean()) if "depth_age_s" in part else np.nan,
                "flow_valid_points_mean": float(part["flow_valid_points"].mean()) if "flow_valid_points" in part else np.nan,
                "flow_track_retention_pct_mean": float(part["flow_track_retention_pct"].mean()) if "flow_track_retention_pct" in part else np.nan,
                "flow_mag_p90_px_mean": float(part["flow_mag_p90_px"].mean()) if "flow_mag_p90_px" in part else np.nan,
                "radial_flow_p90_px_mean": float(part["radial_flow_p90_px"].mean()) if "radial_flow_p90_px" in part else np.nan,
                "delta_depth_close_5m_pp_mean": float(part["delta_depth_close_5m_pp"].mean()) if "delta_depth_close_5m_pp" in part else np.nan,
                "delta_depth_close_5m_pp_std": float(part["delta_depth_close_5m_pp"].std()) if "delta_depth_close_5m_pp" in part else np.nan,
                "event_rate_pct": event_rate,
            }
        )
    return pd.DataFrame(rows)


def reference_mlp_df() -> pd.DataFrame:
    return pd.DataFrame(
        REFERENCE_MLP_TEST,
        columns=["leva", "modelo", "alvo_delta", "MAE", "RMSE"],
    )


def reference_event_df() -> pd.DataFrame:
    return pd.DataFrame(
        REFERENCE_EVENT_RESULTS,
        columns=["grupo", "metrica", "valor"],
    )


def reference_run_event_df() -> pd.DataFrame:
    df = pd.DataFrame(
        REFERENCE_RUN_EVENT_STATS,
        columns=["run_id", "intervalos", "eventos", "taxa_evento"],
    )
    df["leva"] = df["run_id"].map(batch_label_for_run)
    return df


def batch_summary_cards(stats: pd.DataFrame, mlp_df: pd.DataFrame) -> list[html.Div]:
    if stats.empty:
        return [metric_card("Comparacao", "-", "Sem dados ou referencias para comparar.")]

    first = stats[stats["leva"] == FIRST_BATCH_LABEL]
    all_runs = stats[stats["leva"] == ALL_BATCH_LABEL]
    first_row = first.iloc[0] if not first.empty else stats.iloc[0]
    all_row = all_runs.iloc[0] if not all_runs.empty else stats.iloc[-1]

    close5 = mlp_df[(mlp_df["alvo_delta"] == "delta_depth_close_5m_pp") & (mlp_df["modelo"] == "MLP")]
    close5_first = close5[close5["leva"] == "Primeira leva"]["MAE"]
    close5_all = close5[close5["leva"] == "Todas atuais"]["MAE"]
    close5_detail = "referencia do notebook"
    close5_value = "-"
    if not close5_first.empty and not close5_all.empty:
        diff = float(close5_first.iloc[0] - close5_all.iloc[0])
        close5_value = fmt_number(float(close5_all.iloc[0]), " p.p.", 2)
        close5_detail = f"antes {float(close5_first.iloc[0]):.2f}; melhora {diff:.2f}"

    return [
        metric_card(
            "Intervalos",
            f"{int(all_row['intervalos'])}",
            f"antes {int(first_row['intervalos'])}; runs {int(first_row['runs'])} -> {int(all_row['runs'])}",
        ),
        metric_card(
            "dt medio",
            fmt_number(float(all_row["dt_s_mean"] * 1000.0), " ms", 1),
            f"antes {float(first_row['dt_s_mean'] * 1000.0):.1f} ms",
        ),
        metric_card(
            "RGB-depth",
            fmt_number(float(all_row["depth_age_s_mean"] * 1000.0), " ms", 1),
            f"antes {float(first_row['depth_age_s_mean'] * 1000.0):.1f} ms",
        ),
        metric_card(
            "Eventos prox.",
            fmt_number(float(all_row["event_rate_pct"]), "%", 1),
            f"antes {float(first_row['event_rate_pct']):.1f}% com |delta| > 0,5 p.p.",
        ),
        metric_card(
            "Variacao < 5m",
            fmt_number(float(all_row["delta_depth_close_5m_pp_std"]), " p.p.", 2),
            f"desvio padrao antes {float(first_row['delta_depth_close_5m_pp_std']):.2f}",
        ),
        metric_card("MLP close 5m", close5_value, close5_detail),
    ]


def batch_notice(intervals_df: pd.DataFrame) -> html.Div:
    if intervals_df.empty:
        text = (
            "Nao encontrei os memmaps de depth/flow em datasets/depth_ground_truth no ambiente atual. "
            "Esta aba esta usando os valores salvos no notebook para a comparacao estatistica e de MLP."
        )
        return html.Div(text, className="notice")

    run_count = intervals_df["run_id"].nunique()
    interval_count = len(intervals_df)
    text = (
        f"Comparacao calculada diretamente dos memmaps encontrados: {interval_count} intervalos em "
        f"{run_count} runs. A primeira leva usa as 4 runs de 20260521; 'todas atuais' inclui "
        "as runs disponiveis agora."
    )
    return html.Div(text, className="notice notice-ok")


def figure_batch_metric_facets(stats: pd.DataFrame) -> go.Figure:
    if stats.empty:
        return apply_layout(go.Figure(), "Resumo estatistico por leva")

    plot_df = stats[stats["leva"].isin([FIRST_BATCH_LABEL, ALL_BATCH_LABEL])].copy()
    metrics = [
        ("intervalos", "Intervalos", ""),
        ("dt_s_mean", "dt medio", "ms"),
        ("depth_age_s_mean", "RGB-depth", "ms"),
        ("flow_valid_points_mean", "Flow valido", "pts"),
        ("flow_track_retention_pct_mean", "Retencao flow", "%"),
        ("event_rate_pct", "Eventos prox.", "%"),
    ]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[m[1] for m in metrics])
    colors = {FIRST_BATCH_LABEL: COLORS["muted"], ALL_BATCH_LABEL: COLORS["accent"]}

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
                marker_color=[colors.get(label, COLORS["drone"]) for label in plot_df["leva"]],
                text=[fmt_number(float(value), "", 1) for value in values],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{x}<br>%{y:.3f} " + unit + "<extra></extra>",
            ),
            row=row,
            col=subplot_col,
        )
        fig.update_yaxes(title=unit, row=row, col=subplot_col)

    return apply_layout(fig, "Resumo visual: primeira leva x todas as runs atuais", 620)


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
    labels = [FIRST_BATCH_LABEL, NEW_BATCH_LABEL, ALL_BATCH_LABEL]
    colors = {FIRST_BATCH_LABEL: COLORS["drone"], NEW_BATCH_LABEL: COLORS["risk"], ALL_BATCH_LABEL: COLORS["accent"]}

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
                    marker_color=colors[label],
                    boxmean=True,
                    showlegend=idx == 0,
                    hovertemplate=label + "<br>%{y:.3f} " + unit + "<extra></extra>",
                ),
                row=row,
                col=subplot_col,
            )
        fig.update_yaxes(title=unit, row=row, col=subplot_col)

    return apply_layout(fig, "Distribuicoes: primeira leva, runs novas e conjunto atual", 700)


def figure_batch_event_rates(intervals_df: pd.DataFrame) -> go.Figure:
    if intervals_df.empty or "delta_depth_close_5m_pp" not in intervals_df.columns:
        summary = reference_run_event_df()
    else:
        summary = (
            intervals_df.assign(evento=lambda df: df["delta_depth_close_5m_pp"].abs() > 0.5)
            .groupby("run_id", as_index=False)
            .agg(intervalos=("evento", "size"), eventos=("evento", "sum"), taxa_evento=("evento", "mean"))
        )
        summary["leva"] = summary["run_id"].map(batch_label_for_run)
    summary = summary.sort_values("run_id")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["run_id"].str.replace("run_", "", regex=False),
            y=summary["taxa_evento"] * 100.0,
            marker_color=np.where(summary["leva"] == FIRST_BATCH_LABEL, COLORS["drone"], COLORS["risk"]),
            customdata=np.stack([summary["intervalos"], summary["eventos"], summary["leva"]], axis=-1),
            hovertemplate=(
                "run=%{x}<br>leva=%{customdata[2]}<br>"
                "eventos=%{customdata[1]} / %{customdata[0]}<br>"
                "taxa=%{y:.1f}%<extra></extra>"
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
    series = [
        ("Primeira leva", "MLP", COLORS["risk"]),
        ("Primeira leva", "Media treino", COLORS["muted"]),
        ("Todas atuais", "MLP", COLORS["drone"]),
        ("Todas atuais", "Media treino", COLORS["accent"]),
    ]
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
    fig.update_xaxes(title="Alvo delta", tickangle=-20)
    fig.update_yaxes(title="MAE no teste")
    return apply_layout(fig, "Resultado de modelo: primeira leva x notebook atual", 520)


def figure_mlp_reference_table(mlp_df: pd.DataFrame) -> go.Figure:
    if mlp_df.empty:
        return apply_layout(go.Figure(), "Tabela de MAE do notebook")

    rows = []
    targets = sorted(mlp_df["alvo_delta"].unique())
    for target in targets:
        old_mlp = mlp_df[(mlp_df["leva"] == "Primeira leva") & (mlp_df["modelo"] == "MLP") & (mlp_df["alvo_delta"] == target)]
        new_mlp = mlp_df[(mlp_df["leva"] == "Todas atuais") & (mlp_df["modelo"] == "MLP") & (mlp_df["alvo_delta"] == target)]
        old_base = mlp_df[(mlp_df["leva"] == "Primeira leva") & (mlp_df["modelo"] == "Media treino") & (mlp_df["alvo_delta"] == target)]
        new_base = mlp_df[(mlp_df["leva"] == "Todas atuais") & (mlp_df["modelo"] == "Media treino") & (mlp_df["alvo_delta"] == target)]
        if old_mlp.empty or new_mlp.empty or old_base.empty or new_base.empty:
            continue
        old_value = float(old_mlp["MAE"].iloc[0])
        new_value = float(new_mlp["MAE"].iloc[0])
        rows.append(
            [
                target,
                f"{old_value:.3f}",
                f"{new_value:.3f}",
                f"{old_value - new_value:+.3f}",
                f"{float(old_base['MAE'].iloc[0]):.3f}",
                f"{float(new_base['MAE'].iloc[0]):.3f}",
            ]
        )

    headers = ["Alvo", "MLP antiga", "MLP atual", "Delta MAE", "Media antiga", "Media atual"]
    values = list(map(list, zip(*rows))) if rows else [[] for _ in headers]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers, fill_color="#e8eef3", align="left", font=dict(size=13)),
                cells=dict(values=values, fill_color="#ffffff", align="left", height=28),
            )
        ]
    )
    return apply_layout(fig, "Tabela estatistica do teste MLP", 340)


def figure_event_reference_table(event_df: pd.DataFrame) -> go.Figure:
    if event_df.empty:
        return apply_layout(go.Figure(), "Modelo em duas etapas")

    values = [
        event_df["grupo"].tolist(),
        event_df["metrica"].tolist(),
        [f"{float(value):.3f}" for value in event_df["valor"]],
    ]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=["Grupo", "Metrica", "Valor no teste atual"], fill_color="#e8eef3", align="left", font=dict(size=13)),
                cells=dict(values=values, fill_color="#ffffff", align="left", height=28),
            )
        ]
    )
    return apply_layout(fig, "Resumo do modelo em duas etapas no notebook atual", 320)


def figure_batch_stats_table(stats: pd.DataFrame) -> go.Figure:
    if stats.empty:
        return apply_layout(go.Figure(), "Tabela de estatisticas por leva")

    table = stats.copy()
    rows = []
    for _, row in table.iterrows():
        rows.append(
            [
                row["leva"],
                f"{int(row['runs'])}",
                f"{int(row['intervalos'])}",
                f"{float(row['dt_s_mean'] * 1000.0):.1f}",
                f"{float(row['depth_age_s_mean'] * 1000.0):.1f}",
                f"{float(row['flow_valid_points_mean']):.1f}",
                f"{float(row['flow_track_retention_pct_mean']):.1f}",
                f"{float(row['event_rate_pct']):.1f}",
                f"{float(row['delta_depth_close_5m_pp_std']):.2f}",
            ]
        )
    headers = [
        "Leva",
        "Runs",
        "Intervalos",
        "dt medio (ms)",
        "RGB-depth (ms)",
        "Flow valido",
        "Retencao (%)",
        "Eventos (%)",
        "Std delta <5m",
    ]
    values = list(map(list, zip(*rows))) if rows else [[] for _ in headers]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers, fill_color="#e8eef3", align="left", font=dict(size=13)),
                cells=dict(values=values, fill_color="#ffffff", align="left", height=28),
            )
        ]
    )
    return apply_layout(fig, "Tabela estatistica das levas", 340)


def comparison_table(paths: list[Path]) -> go.Figure:
    """
    Gera a tabela comparativa das runs de voo.

    A coluna de indice reativo usa as metricas reais quando o CSV possui obstacle_risk,
    avoid_lateral_body e avoid_brake; caso contrario, usa o proxy historico e indica a
    fonte dos dados.

    Fontes:
    [Plotly Table] https://plotly.com/python/table/
    [Pandas DataFrame] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    """

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
    values = list(map(list, zip(*rows))) if rows else [[] for _ in headers]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=headers, fill_color="#e8eef3", align="left", font=dict(size=13)),
                cells=dict(values=values, fill_color="#ffffff", align="left", height=28),
            )
        ]
    )
    return apply_layout(fig, "Comparativo entre runs", 320)


def layout(log_paths: list[Path], depth_paths: list[Path]) -> html.Div:
    """
    Monta o layout do dashboard em abas por tema.

    O seletor de voo fica no topo porque as abas de trajetoria/IMU e evasao usam a mesma
    run. A aba de depth mantem seu seletor proprio, e a aba de comparacao agrega as levas
    de intervalos depth/flow usadas no notebook.

    Fontes:
    [Dash Tabs] https://dash.plotly.com/dash-core-components/tabs
    [Dash Dropdown] https://dash.plotly.com/dash-core-components/dropdown
    """

    flight_options = [{"label": run_label(path), "value": str(path)} for path in log_paths]
    flight_default = flight_options[0]["value"] if flight_options else ""
    depth_options = [{"label": depth_run_label(path), "value": str(path)} for path in depth_paths]
    paired_depth_default = matching_depth_metadata(Path(flight_default), depth_paths) if flight_default else None
    depth_default = str(paired_depth_default) if paired_depth_default is not None else (depth_options[0]["value"] if depth_options else "")

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
                                "Analise organizada por trajetoria/IMU, evasao, depth ground truth e comparacao das levas."
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
                    dcc.Tab(
                        label="Trajetoria e IMU",
                        value="overview-tab",
                        className="tab",
                        selected_className="tab tab-selected",
                        children=[
                            html.Div(
                                className="tab-panel",
                                children=[
                                    html.Div(id="metrics", className="metrics-grid"),
                                    html.Div(id="flight-notice"),
                                    html.Div(
                                        className="graph-grid graph-grid-two",
                                        children=[
                                            dcc.Graph(id="xy-graph", config={"displaylogo": False}),
                                            dcc.Graph(id="trajectory-3d", config={"displaylogo": False}),
                                        ],
                                    ),
                                    html.Div(
                                        className="graph-grid graph-grid-two",
                                        children=[
                                            dcc.Graph(id="control-graph", config={"displaylogo": False}),
                                            dcc.Graph(id="altitude-graph", config={"displaylogo": False}),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Evasao e voo",
                        value="reactive-tab",
                        className="tab",
                        selected_className="tab tab-selected",
                        children=[
                            html.Div(
                                className="tab-panel",
                                children=[
                                    html.Div(
                                        className="graph-grid graph-grid-two",
                                        children=[
                                            dcc.Graph(id="reactive-graph", config={"displaylogo": False}),
                                            dcc.Graph(id="speed-graph", config={"displaylogo": False}),
                                        ],
                                    ),
                                    dcc.Graph(id="comparison-table", config={"displaylogo": False}),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Depth ground truth",
                        value="depth-tab",
                        className="tab",
                        selected_className="tab tab-selected",
                        children=[
                            html.Div(
                                className="tab-panel",
                                children=[
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
                                    html.Div(
                                        className="graph-grid graph-grid-two",
                                        children=[
                                            dcc.Graph(id="depth-timeseries", config={"displaylogo": False}),
                                            dcc.Graph(id="depth-imu-graph", config={"displaylogo": False}),
                                        ],
                                    ),
                                    dcc.Graph(id="depth-ranking-table", config={"displaylogo": False}),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Comparar levas",
                        value="batch-tab",
                        className="tab",
                        selected_className="tab tab-selected",
                        children=[
                            html.Div(
                                className="tab-panel",
                                children=[
                                    html.Div(id="batch-comparison-notice"),
                                    html.Div(id="batch-comparison-metrics", className="metrics-grid"),
                                    html.Div(
                                        className="graph-grid graph-grid-two",
                                        children=[
                                            dcc.Graph(id="batch-metric-facets", config={"displaylogo": False}),
                                            dcc.Graph(id="batch-event-rates", config={"displaylogo": False}),
                                        ],
                                    ),
                                    dcc.Graph(id="batch-distributions", config={"displaylogo": False}),
                                    html.Div(
                                        className="graph-grid graph-grid-two",
                                        children=[
                                            dcc.Graph(id="batch-mlp-mae", config={"displaylogo": False}),
                                            dcc.Graph(id="batch-mlp-table", config={"displaylogo": False}),
                                        ],
                                    ),
                                    html.Div(
                                        className="graph-grid graph-grid-two",
                                        children=[
                                            dcc.Graph(id="batch-stats-table", config={"displaylogo": False}),
                                            dcc.Graph(id="batch-event-table", config={"displaylogo": False}),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    ),
                ],
            ),
            dcc.Interval(id="refresh-data", interval=10000, n_intervals=0),
        ],
    )


def create_app() -> Dash:
    log_paths = list_log_files()
    depth_paths = list_depth_metadata_files()
    app = Dash(__name__)
    app.title = "Metricas do desvio reativo"
    app.layout = layout(log_paths, depth_paths)

    app.index_string = """
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

    @app.callback(
        Output("run-select", "options"),
        Output("run-select", "value"),
        Input("refresh-data", "n_intervals"),
        State("run-select", "value"),
    )
    def refresh_flight_run_options(_n_intervals: int, selected_path: str):
        """
        Atualiza o seletor da aba de voo enquanto o dashboard esta aberto.

        A cada intervalo, a funcao relista logs/*.csv. Se a run selecionada ainda existir,
        ela e preservada; caso contrario, o dashboard abre na run mais recente.

        Fontes:
        [Dash callbacks] https://dash.plotly.com/basic-callbacks
        [Dash dcc.Interval] https://dash.plotly.com/dash-core-components/interval
        """

        current_paths = list_log_files()
        options = [{"label": run_label(path), "value": str(path)} for path in current_paths]
        valid_values = {option["value"] for option in options}
        selected = selected_path if selected_path in valid_values else (options[0]["value"] if options else "")
        return options, selected

    @app.callback(
        Output("depth-run-select", "options"),
        Output("depth-run-select", "value"),
        Input("refresh-data", "n_intervals"),
        Input("run-select", "value"),
        State("depth-run-select", "value"),
    )
    def refresh_depth_run_options(_n_intervals: int, selected_log_path: str, selected_path: str):
        """
        Atualiza o seletor da aba de depth enquanto o dashboard esta aberto.

        A funcao relista datasets/depth_ground_truth/run_*/metadata.csv e preserva a run
        selecionada quando possivel. Isso permite gerar uma nova coleta de depth e ve-la no
        dashboard sem reiniciar o servidor.

        Fontes:
        [Dash callbacks] https://dash.plotly.com/basic-callbacks
        [Dash dcc.Interval] https://dash.plotly.com/dash-core-components/interval
        """

        current_paths = list_depth_metadata_files()
        options = [{"label": depth_run_label(path), "value": str(path)} for path in current_paths]
        valid_values = {option["value"] for option in options}
        paired = None
        if selected_log_path:
            paired_path = matching_depth_metadata(Path(selected_log_path), current_paths)
            paired = str(paired_path) if paired_path is not None else None
        selected = paired if paired in valid_values else (selected_path if selected_path in valid_values else (options[0]["value"] if options else ""))
        return options, selected

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
        """
        Atualiza somente a aba de metricas de voo.

        O callback recebe uma run vinda de logs/*.csv e nao tenta carregar depth. Isso evita
        misturar os seletores e deixa claro se a evasao veio de metricas reais ou de proxy
        para logs antigos.

        Fontes:
        [Dash callbacks] https://dash.plotly.com/basic-callbacks
        [Plotly figures] https://plotly.com/python/creating-and-updating-figures/
        """

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
        """
        Atualiza somente a aba de depth ground truth.

        O callback carrega metadata.csv de datasets/depth_ground_truth/run_* e os arquivos
        .npy correspondentes. Assim, as runs de depth aparecem em uma lista propria e nao
        dependem da existencia de um CSV de voo com o mesmo timestamp.

        Fontes:
        [Dash callbacks] https://dash.plotly.com/basic-callbacks
        [NumPy load] https://numpy.org/doc/stable/reference/generated/numpy.load.html
        [Gazebo DepthCamera] https://gazebosim.org/api/rendering/7/classgz_1_1rendering_1_1DepthCamera.html
        """

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
        Input("refresh-data", "n_intervals"),
    )
    def update_batch_comparison(_n_intervals: int):
        intervals_df = load_all_depth_intervals()
        stats_df = summarize_batch_intervals(intervals_df)
        batch_df = batch_interval_frames(intervals_df)
        mlp_df = reference_mlp_df()
        event_df = reference_event_df()

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
        )

    return app


if __name__ == "__main__":
    create_app().run(debug=False, host="127.0.0.1", port=8050)
