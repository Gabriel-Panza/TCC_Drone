from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"

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
    return sorted(LOG_DIR.glob("*.csv"))


def run_label(path: Path) -> str:
    return path.stem.replace("voo_teste_", "")


def load_run(path: Path) -> pd.DataFrame:
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
    if df.empty:
        return [metric_card("Sem dados", "-", "Nenhum CSV encontrado em logs/.")]

    total_distance = float(
        np.sqrt(
            df["x"].diff().fillna(0.0) ** 2
            + df["y"].diff().fillna(0.0) ** 2
            + df["z"].diff().fillna(0.0) ** 2
        ).sum()
    )
    event_share = float(df["evento_desvio_proxy"].mean() * 100.0)
    max_proxy = float(df["proxy_desvio_reativo"].max())

    return [
        metric_card("Duracao", fmt_number(float(df["tempo_s"].max()), " s", 1), f"{len(df)} amostras"),
        metric_card("Distancia 3D", fmt_number(total_distance, " m", 1), "odometria acumulada"),
        metric_card("Altitude max.", fmt_number(float(df["altitude"].max()), " m", 2), "NED convertido para altitude"),
        metric_card("Desvio rota max.", fmt_number(float(df["desvio_rota_m"].max()), " m", 2), "distancia lateral ao segmento"),
        metric_card("Proxy reativo max.", fmt_number(max_proxy, "", 2), f"limiar visual: {OBSTACLE_RISK_THRESHOLD:.2f}"),
        metric_card("Tempo em desvio", fmt_number(event_share, "%", 1), "proxy acima do limiar"),
    ]


def apply_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=17)),
        template="plotly_white",
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        height=height,
        margin=dict(l=48, r=28, t=58, b=46),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Segoe UI, Arial, sans-serif", color=COLORS["ink"]),
        hovermode="closest",
    )
    return fig


def figure_xy(df: pd.DataFrame) -> go.Figure:
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
                color=df["proxy_desvio_reativo"],
                colorscale=[[0, "#2563eb"], [0.45, "#f59e0b"], [1, "#c2410c"]],
                cmin=0,
                cmax=1,
                colorbar=dict(title="proxy"),
            ),
            customdata=np.stack(
                [
                    df["tempo_s"],
                    df["desvio_rota_m"],
                    df["vel_lateral_rota"],
                    df["proxy_desvio_reativo"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "t=%{customdata[0]:.2f}s<br>"
                "x=%{x:.2f}<br>y=%{y:.2f}<br>"
                "desvio=%{customdata[1]:.2f}m<br>"
                "vel lateral=%{customdata[2]:.2f}m/s<br>"
                "proxy=%{customdata[3]:.2f}<extra></extra>"
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
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=df["proxy_desvio_reativo"],
            name="proxy desvio reativo",
            line=dict(color=COLORS["risk"], width=2),
            fill="tozeroy",
            fillcolor="rgba(194, 65, 12, 0.16)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["tempo_s"],
            y=df["desvio_rota_m"] / max(RAIO_DE_ACEITACAO, 1.0),
            name="desvio / raio aceitacao",
            line=dict(color=COLORS["accent"], width=2),
        )
    )
    fig.add_hline(
        y=OBSTACLE_RISK_THRESHOLD,
        line_dash="dot",
        line_color="#991b1b",
        annotation_text="limiar de acionamento visual",
        annotation_position="top left",
    )
    fig.update_yaxes(title="indice normalizado", range=[0, max(1.05, float(df["proxy_desvio_reativo"].max()) + 0.1)])
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


def comparison_table(paths: list[Path]) -> go.Figure:
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
                f"{float(df['proxy_desvio_reativo'].max()):.2f}",
                f"{float(df['evento_desvio_proxy'].mean() * 100.0):.1f}",
            ]
        )

    headers = [
        "Run",
        "Duracao (s)",
        "Dist. 3D (m)",
        "Alt. max (m)",
        "Desvio max (m)",
        "Proxy max",
        "Tempo desvio (%)",
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


def layout(log_paths: list[Path]) -> html.Div:
    options = [{"label": run_label(path), "value": str(path)} for path in log_paths]
    default_value = options[0]["value"] if options else ""

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
                                "Analise dos logs de odometria usando a rota e os parametros da abordagem reativa atual."
                            ),
                        ]
                    ),
                    html.Div(
                        className="selector",
                        children=[
                            html.Label("Run"),
                            dcc.Dropdown(
                                id="run-select",
                                options=options,
                                value=default_value,
                                clearable=False,
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(id="metrics", className="metrics-grid"),
            html.Div(
                className="notice",
                children=(
                    "Os sinais obstacle_risk, avoid_lateral_body e avoid_brake ainda nao sao gravados no CSV. "
                    "Por isso, o indice de desvio reativo aqui e um proxy inferido por desvio lateral, "
                    "velocidade lateral, frenagem relativa e atividade angular."
                ),
            ),
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


def create_app() -> Dash:
    log_paths = list_log_files()
    app = Dash(__name__)
    app.title = "Metricas do desvio reativo"
    app.layout = layout(log_paths)

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
        Output("metrics", "children"),
        Output("xy-graph", "figure"),
        Output("trajectory-3d", "figure"),
        Output("reactive-graph", "figure"),
        Output("control-graph", "figure"),
        Output("speed-graph", "figure"),
        Output("altitude-graph", "figure"),
        Output("comparison-table", "figure"),
        Input("run-select", "value"),
    )
    def update_dashboard(selected_path: str):
        path = Path(selected_path) if selected_path else (log_paths[0] if log_paths else Path())
        df = load_run(path)
        if df.empty:
            empty_fig = apply_layout(go.Figure(), "Sem dados")
            return [metric_card("Sem dados", "-", "Nenhum CSV encontrado.")], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, comparison_table(log_paths)

        return (
            run_summary(df),
            figure_xy(df),
            figure_3d(df),
            figure_reactive_timeseries(df),
            figure_control(df),
            figure_speed(df),
            figure_altitude(df),
            comparison_table(log_paths),
        )

    return app


if __name__ == "__main__":
    create_app().run(debug=False, host="127.0.0.1", port=8050)
