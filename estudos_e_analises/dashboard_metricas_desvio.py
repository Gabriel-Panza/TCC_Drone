from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"
DEPTH_GT_DIR = PROJECT_ROOT / "datasets" / "depth_ground_truth"

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


def list_log_files() -> list[Path]:
    """
    Lista os CSVs de voo do mais recente para o mais antigo.

    A ordenacao descendente faz a run nova aparecer primeiro no seletor do dashboard e evita
    que a tela abra sempre no voo mais antigo.

    Fonte:
    [Python pathlib] https://docs.python.org/3/library/pathlib.html
    """

    return sorted(LOG_DIR.glob("*.csv"), reverse=True)


def list_depth_metadata_files() -> list[Path]:
    """
    Lista os arquivos metadata.csv produzidos pelo dataset de profundidade do Gazebo.

    Cada arquivo representa uma run gerada pelo parametro save_ground_truth_dataset do
    controlador. O dashboard usa esses metadados como ground truth sintetico para avaliar
    se havia objetos proximos no campo visual da camera monocular, sem tratar o depth como
    sensor embarcado.

    Fontes:
    [Python pathlib] https://docs.python.org/3/library/pathlib.html
    [Gazebo DepthCamera] https://gazebosim.org/api/rendering/7/classgz_1_1rendering_1_1DepthCamera.html
    [Artigo - RealTimeMonocular2022] https://doi.org/10.1109/TITS.2022.3160741
    """

    return sorted(DEPTH_GT_DIR.glob("run_*/metadata.csv"), reverse=True)


def run_label(path: Path) -> str:
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

    log_token = run_label(log_path)
    for depth_path in depth_paths:
        if log_token and log_token in depth_path.parent.name:
            return depth_path

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
    risk_label = "Risco visual max." if source == "metricas reais" else "Proxy reativo max."

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
    index_name = "obstacle_risk" if source == "metricas reais" else "proxy desvio reativo"
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
            name="avoid_brake" if source == "metricas reais" else "freio estimado",
            line=dict(color=COLORS["accent"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=normalize(df["lateral_reativo_m_s2"]),
            name="|avoid_lateral_body|" if source == "metricas reais" else "|lateral estimado|",
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
    fig.update_xaxes(title="Tempo de voo (s)")
    fig.update_yaxes(title="Velocidade angular (rad/s)")
    return apply_layout(fig, "Esforco de controle angular", 390)


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
    Monta o layout do dashboard com abas independentes de voo e depth.

    A aba de voo lista apenas arquivos logs/*.csv. A aba de depth lista apenas
    datasets/depth_ground_truth/run_*/metadata.csv, evitando que uma run de simulador fique
    escondida no seletor errado.

    Fontes:
    [Dash Tabs] https://dash.plotly.com/dash-core-components/tabs
    [Dash Dropdown] https://dash.plotly.com/dash-core-components/dropdown
    """

    flight_options = [{"label": run_label(path), "value": str(path)} for path in log_paths]
    flight_default = flight_options[0]["value"] if flight_options else ""
    depth_options = [{"label": depth_run_label(path), "value": str(path)} for path in depth_paths]
    depth_default = depth_options[0]["value"] if depth_options else ""

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
                                "Analise separada das metricas de voo e do depth ground truth renderizado pelo Gazebo."
                            ),
                        ]
                    ),
                ],
            ),
            dcc.Tabs(
                id="dashboard-tabs",
                value="flight-tab",
                className="tabs",
                children=[
                    dcc.Tab(
                        label="Metricas de voo",
                        value="flight-tab",
                        className="tab",
                        selected_className="tab tab-selected",
                        children=[
                            html.Div(
                                className="tab-panel",
                                children=[
                                    html.Div(
                                        className="selector selector-inline",
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
                                        className="graph-grid graph-grid-three",
                                        children=[
                                            dcc.Graph(id="reactive-graph", config={"displaylogo": False}),
                                            dcc.Graph(id="control-graph", config={"displaylogo": False}),
                                            dcc.Graph(id="speed-graph", config={"displaylogo": False}),
                                        ],
                                    ),
                                    dcc.Graph(id="altitude-graph", config={"displaylogo": False}),
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
                                            html.Label("Run de depth"),
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
        State("depth-run-select", "value"),
    )
    def refresh_depth_run_options(_n_intervals: int, selected_path: str):
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
        selected = selected_path if selected_path in valid_values else (options[0]["value"] if options else "")
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

    return app


if __name__ == "__main__":
    create_app().run(debug=False, host="127.0.0.1", port=8050)
