"""Carregamento e graficos dos experimentos espaciais finais."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def latest_final_analysis(root: Path = PROJECT_ROOT) -> Path:
    """Retorna o relatorio espacial final mais recente."""
    candidates = sorted(
        path.parent for path in
        (root / "logs" / "spatial_final_analysis").glob("final_*/final_report.json")
    )
    return candidates[-1] if candidates else root / "logs" / "spatial_final_analysis"


def load_spatial_results(root: Path = PROJECT_ROOT):
    """Carrega resumo, runs de referencia e comparacao monocular."""
    analysis = latest_final_analysis(root)
    report_path = analysis / "final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    reference_path = analysis / "reference_runs.csv"
    monocular_path = analysis / "monocular_comparison.csv"
    reference = pd.read_csv(reference_path) if reference_path.exists() else pd.DataFrame()
    monocular = pd.read_csv(monocular_path) if monocular_path.exists() else pd.DataFrame()
    return report, reference, monocular


def load_run_trajectory(run_dir: Path) -> pd.DataFrame:
    """Extrai posicoes NED gravadas em events.jsonl."""
    event_path = Path(run_dir) / "events.jsonl"
    rows = []
    if not event_path.exists():
        return pd.DataFrame(columns=["sample", "north_m", "east_m", "down_m", "event"])
    for sample, line in enumerate(event_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        position = event.get("position_ned_m")
        if position is None and event.get("event") == "plan":
            position = (event.get("plan", {}).get("diagnostics") or {}).get(
                "current_position_ned_m"
            )
        if position is None or len(position) < 3:
            continue
        rows.append({
            "sample": sample, "north_m": float(position[0]),
            "east_m": float(position[1]), "down_m": float(position[2]),
            "event": event.get("event", ""),
        })
    return pd.DataFrame(rows)


def reference_runs_figure(reference: pd.DataFrame) -> go.Figure:
    """Resume duracao, planejamento, sucesso e falso espaco livre por run."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Duracao da missao", "Tempo medio de planejamento",
        "Taxa de sucesso dos planos", "Taxa de falso espaco livre",
    ))
    if reference.empty:
        return fig
    run = reference["run"].astype(str)
    success = reference["plan_successes"] / reference["plan_attempts"].clip(lower=1)
    values = (
        (reference["duration_s"], "s"),
        (reference["planning_mean_ms"], "ms"),
        (success * 100.0, "%"),
        (reference["false_free_rate"] * 100.0, "%"),
    )
    for index, (series, unit) in enumerate(values):
        row, col = index // 2 + 1, index % 2 + 1
        fig.add_trace(go.Bar(
            x=run, y=series, text=[f"{value:.1f}" for value in series],
            textposition="outside", cliponaxis=False, showlegend=False,
            hovertemplate=f"run=%{{x}}<br>%{{y:.2f}} {unit}<extra></extra>",
        ), row=row, col=col)
        fig.update_yaxes(title=unit, rangemode="tozero", automargin=True, row=row, col=col)
    fig.update_layout(title="Bateria espacial com profundidade de referencia",
                      template="plotly_white", height=780,
                      margin=dict(t=105, b=85), bargap=0.18)
    return fig


def monocular_models_figure(monocular: pd.DataFrame) -> go.Figure:
    """Compara erro por pixel com os criterios espaciais v12--v19."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "MAE de profundidade", "Sucesso do planejamento",
        "Falso espaco livre", "Colisoes no replay",
    ))
    if monocular.empty:
        return fig
    model = monocular["model"]
    series = (
        (monocular["depth_mae_m"], "m"),
        (monocular["plan_success_rate"] * 100.0, "%"),
        (monocular["false_free_rate"] * 100.0, "%"),
        (monocular["collisions"], "colisoes"),
    )
    for index, (values, unit) in enumerate(series):
        row, col = index // 2 + 1, index % 2 + 1
        fig.add_trace(go.Bar(
            x=model, y=values, text=[f"{value:.2f}" for value in values],
            textposition="outside", cliponaxis=False, showlegend=False,
            hovertemplate=f"%{{x}}<br>%{{y:.3f}} {unit}<extra></extra>",
        ), row=row, col=col)
        fig.update_yaxes(title=unit, rangemode="tozero", automargin=True, row=row, col=col)
    fig.update_layout(title="Monocular: pixel versus seguranca espacial",
                      template="plotly_white", height=780,
                      margin=dict(t=105, b=85), bargap=0.18)
    return fig


def reference_trajectories_figure(reference: pd.DataFrame) -> go.Figure:
    """Sobrepoe no plano NED as trajetorias da bateria de referencia."""
    fig = go.Figure()
    for _, row in reference.iterrows():
        trajectory = load_run_trajectory(Path(row["dataset"]))
        if trajectory.empty:
            continue
        fig.add_trace(go.Scatter(
            x=trajectory["east_m"], y=trajectory["north_m"], mode="lines",
            name=f"run {int(row['run'])}",
            customdata=trajectory[["down_m", "event"]],
            hovertemplate=("E=%{x:.1f} m<br>N=%{y:.1f} m<br>"
                           "D=%{customdata[0]:.2f} m<br>%{customdata[1]}<extra></extra>"),
        ))
    fig.update_xaxes(title="Leste (m)", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="Norte (m)")
    fig.update_layout(title="Trajetorias NED das dez runs de referencia",
                      template="plotly_white", height=650, legend_title="Execucao")
    return fig
