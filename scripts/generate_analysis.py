"""
CFB Model — Residual Analysis Generator
=========================================
Reads outputs/predictions/model_results.csv and produces a standalone
HTML report (outputs/charts/model_analysis.html) with four charts:

  1. Predicted vs. Actual spread margin scatter + regression line
  2. Team residual bar chart (biggest over/under-estimates)
  3. Week-by-week MAE line chart
  4. Predicted vs. Actual totals scatter

Run:
    python3 scripts/generate_analysis.py [--season 2025]

The same analysis is rendered live inside the Streamlit app's
"Model Analysis" tab via the shared build_analysis_figures() function.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_CSV  = ROOT / "outputs" / "predictions" / "model_results.csv"
OUT_HTML = ROOT / "outputs" / "charts" / "model_analysis.html"


# ─── SHARED FIGURE BUILDERS ───────────────────────────────────────────────────

DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1f2e"
BORDER    = "#252d3d"
TEXT      = "#d1d5db"
MUTED     = "#6b7280"
GOLD      = "#eab308"
GREEN     = "#22c55e"
RED       = "#ef4444"
BLUE      = "#3b82f6"
PURPLE    = "#8b5cf6"


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=14), x=0),
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT, size=11),
        margin=dict(l=50, r=20, t=48, b=50),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    )


def load_results(season: int | None = None) -> pd.DataFrame:
    """Load model_results.csv, filter to season if given, compute residuals."""
    if not OUT_CSV.exists():
        raise FileNotFoundError(
            f"model_results.csv not found at {OUT_CSV}. "
            "Run `python3 src/model.py` first."
        )
    df = pd.read_csv(OUT_CSV)

    # Coerce numeric columns
    for col in ["point_diff", "pred_spread", "total_points", "pred_total",
                "spread", "over_under"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if season is not None:
        df = df[df["season"] == season]

    # Core residuals
    # pred_spread = model's predicted home margin (positive = model likes home)
    # point_diff  = actual home margin
    df["residual"]       = df["pred_spread"] - df["point_diff"]   # >0 = overestimated home
    df["abs_residual"]   = df["residual"].abs()
    df["within_7"]       = (df["abs_residual"] <= 7.0).astype(int)

    if "pred_total" in df.columns and "total_points" in df.columns:
        df["total_residual"]     = df["pred_total"] - df["total_points"]
        df["total_abs_residual"] = df["total_residual"].abs()

    return df


def summary_stats(df: pd.DataFrame) -> dict:
    spread_mae  = df["abs_residual"].mean()
    spread_r2   = 1 - ((df["residual"] ** 2).sum() /
                       ((df["point_diff"] - df["point_diff"].mean()) ** 2).sum())
    within_7    = df["within_7"].mean()
    n           = len(df)

    stats = dict(
        n=n,
        spread_mae=spread_mae,
        spread_r2=spread_r2,
        within_7=within_7,
    )

    if "covered_spread" in df.columns:
        stats["ats_acc"] = df["covered_spread"].mean()
    if "went_over" in df.columns and "pred_total" in df.columns:
        over_pred = df["pred_total"] > df["over_under"]
        stats["totals_acc"] = (over_pred == df["went_over"].astype(bool)).mean()
    if "total_abs_residual" in df.columns:
        stats["totals_mae"] = df["total_abs_residual"].mean()

    return stats


def fig_scatter_spread(df: pd.DataFrame) -> go.Figure:
    """Predicted vs. actual spread margin with identity line + regression."""
    x = df["pred_spread"].dropna()
    y = df["point_diff"].reindex(x.index).dropna()
    x, y = x.align(y, join="inner")

    # OLS regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)

    # Colour each dot by whether model was right direction (residual sign matches)
    correct = ((x > 0) == (y > 0)).map({True: GREEN, False: RED})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=correct, size=5, opacity=0.7),
        text=df.reindex(x.index).apply(
            lambda r: f"{r['home_team']} vs {r['away_team']}<br>"
                      f"Pred: {r['pred_spread']:+.1f}  Actual: {r['point_diff']:+.1f}",
            axis=1),
        hoverinfo="text",
        name="Games",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b,
        mode="lines", line=dict(color=GOLD, width=2, dash="solid"),
        name=f"Regression (slope={m:.2f})",
    ))
    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()], y=[x.min(), x.max()],
        mode="lines", line=dict(color=MUTED, width=1, dash="dash"),
        name="Perfect prediction",
    ))
    fig.update_layout(
        **_base_layout("Predicted vs. Actual Home Margin"),
        xaxis_title="Model prediction (home pts)",
        yaxis_title="Actual margin (home pts)",
        legend=dict(bgcolor=PANEL_BG, bordercolor=BORDER, font=dict(size=10)),
        height=420,
    )
    return fig


def fig_team_residuals(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Mean residual per team (over all games as home or away).
    Positive residual = model overestimated the home team → model is too
    bullish on teams the dot appears for.
    """
    rows = []
    for _, r in df.iterrows():
        if pd.notna(r["residual"]):
            rows.append({"team": r["home_team"], "residual": r["residual"]})
            rows.append({"team": r["away_team"], "residual": -r["residual"]})

    team_df = (pd.DataFrame(rows)
               .groupby("team")["residual"]
               .agg(["mean", "count"])
               .reset_index()
               .rename(columns={"mean": "mean_residual", "count": "n_games"}))
    team_df = team_df[team_df["n_games"] >= 3]  # filter noise

    # Take the top_n most biased in each direction
    top_over  = team_df.nlargest(top_n // 2,  "mean_residual")
    top_under = team_df.nsmallest(top_n // 2, "mean_residual")
    plot_df   = pd.concat([top_over, top_under]).sort_values("mean_residual")

    colors = [GREEN if v > 0 else RED for v in plot_df["mean_residual"]]

    fig = go.Figure(go.Bar(
        x=plot_df["mean_residual"],
        y=plot_df["team"],
        orientation="h",
        marker_color=colors,
        text=plot_df["mean_residual"].apply(lambda v: f"{v:+.1f}"),
        textposition="outside",
        hovertemplate="%{y}<br>Avg residual: %{x:+.1f} pts<br>Games: %{customdata}<extra></extra>",
        customdata=plot_df["n_games"],
    ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(
        **_base_layout("Team Residuals — Model Bias by Team"),
        xaxis_title="Mean residual (pts)  |  + = model overestimates team",
        yaxis_title="",
        height=max(360, len(plot_df) * 22 + 80),
        showlegend=False,
    )
    return fig


def fig_mae_by_week(df: pd.DataFrame) -> go.Figure:
    """MAE and game count by week — shows where accuracy degrades."""
    wk = (df.groupby("week")
            .agg(mae=("abs_residual", "mean"), n=("abs_residual", "count"))
            .reset_index()
            .sort_values("week"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wk["week"], y=wk["mae"],
        mode="lines+markers",
        line=dict(color=GOLD, width=2),
        marker=dict(size=7, color=GOLD),
        text=wk.apply(lambda r: f"Wk {int(r['week'])}<br>MAE: {r['mae']:.1f} pts<br>Games: {int(r['n'])}", axis=1),
        hoverinfo="text",
        name="MAE",
    ))
    # Shade the band ±1 pt around the mean
    mean_mae = wk["mae"].mean()
    fig.add_hline(y=mean_mae, line_color=MUTED, line_width=1, line_dash="dash",
                  annotation_text=f"Season avg {mean_mae:.1f} pts",
                  annotation_font_color=MUTED)
    fig.update_layout(
        **_base_layout("Spread MAE by Week"),
        xaxis_title="Week",
        yaxis_title="Mean Absolute Error (pts)",
        height=340,
    )
    return fig


def fig_scatter_totals(df: pd.DataFrame) -> go.Figure:
    """Predicted vs. actual total score scatter."""
    if "pred_total" not in df.columns or "total_points" not in df.columns:
        return None

    x = df["pred_total"].dropna()
    y = df["total_points"].reindex(x.index).dropna()
    x, y = x.align(y, join="inner")

    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)

    # Colour by correct OVER/UNDER call
    over_pred   = (df.reindex(x.index)["pred_total"] > df.reindex(x.index)["over_under"])
    actual_over = df.reindex(x.index)["went_over"].astype(bool) if "went_over" in df.columns else over_pred
    correct     = (over_pred == actual_over).map({True: GREEN, False: RED})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=correct, size=5, opacity=0.7),
        text=df.reindex(x.index).apply(
            lambda r: f"{r['home_team']} vs {r['away_team']}<br>"
                      f"Pred: {r['pred_total']:.1f}  Actual: {r['total_points']:.1f}",
            axis=1),
        hoverinfo="text",
        name="Games",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b,
        mode="lines", line=dict(color=GOLD, width=2),
        name=f"Regression (slope={m:.2f})",
    ))
    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()], y=[x.min(), x.max()],
        mode="lines", line=dict(color=MUTED, width=1, dash="dash"),
        name="Perfect prediction",
    ))
    fig.update_layout(
        **_base_layout("Predicted vs. Actual Total Score"),
        xaxis_title="Model total prediction (pts)",
        yaxis_title="Actual combined score (pts)",
        legend=dict(bgcolor=PANEL_BG, bordercolor=BORDER, font=dict(size=10)),
        height=420,
    )
    return fig


def build_analysis_figures(season: int | None = None):
    """
    Entry point for both this script and the Streamlit app.
    Returns (df, stats, fig_spread, fig_teams, fig_week, fig_totals).
    fig_totals may be None if pred_total is unavailable.
    """
    df    = load_results(season)
    stats = summary_stats(df)
    return (
        df,
        stats,
        fig_scatter_spread(df),
        fig_team_residuals(df),
        fig_mae_by_week(df),
        fig_scatter_totals(df),
    )


# ─── STANDALONE HTML OUTPUT ───────────────────────────────────────────────────

def generate_html(season: int | None = None):
    df, stats, f_spread, f_teams, f_week, f_totals = build_analysis_figures(season)

    season_label = str(season) if season else "All seasons"
    mae    = stats["spread_mae"]
    r2     = stats["spread_r2"]
    within = stats["within_7"]
    n      = stats["n"]

    ats_row    = f"<li>ATS accuracy: <strong>{stats['ats_acc']:.1%}</strong></li>" if "ats_acc" in stats else ""
    totals_row = f"<li>Totals accuracy: <strong>{stats['totals_acc']:.1%}</strong> · MAE {stats['totals_mae']:.1f} pts</li>" if "totals_acc" in stats else ""

    html_parts = [f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>CFB Model Analysis — {season_label}</title>
<style>
  body {{ background:#0f1117; color:#d1d5db; font-family:system-ui,sans-serif; margin:0; padding:24px; }}
  h1 {{ color:#fff; font-size:1.3em; margin-bottom:4px; }}
  .subtitle {{ color:#6b7280; font-size:0.85em; margin-bottom:24px; }}
  .stats-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:28px; }}
  .stat {{ background:#1a1f2e; border:1px solid #252d3d; border-radius:10px; padding:14px 18px; }}
  .stat-label {{ color:#6b7280; font-size:0.7em; text-transform:uppercase; letter-spacing:.07em; }}
  .stat-value {{ color:#fff; font-size:1.5em; font-weight:700; margin-top:4px; }}
  .chart-wrap {{ background:#1a1f2e; border:1px solid #252d3d; border-radius:10px;
                  padding:16px; margin-bottom:20px; }}
  ul {{ color:#9ca3af; font-size:0.88em; margin:8px 0 0 0; padding-left:18px; }}
</style>
</head><body>
<h1>CFB Model Analysis</h1>
<div class="subtitle">Test set: {season_label} · {n:,} games</div>
<div class="stats-grid">
  <div class="stat"><div class="stat-label">Spread MAE</div><div class="stat-value">{mae:.1f} pts</div></div>
  <div class="stat"><div class="stat-label">R²</div><div class="stat-value">{r2:.3f}</div></div>
  <div class="stat"><div class="stat-label">Within 7 pts</div><div class="stat-value">{within:.0%}</div></div>
  <div class="stat"><div class="stat-label">Games</div><div class="stat-value">{n:,}</div></div>
</div>
<ul>
  {ats_row}
  {totals_row}
</ul>
"""]

    def add_fig(fig, title, note=""):
        if fig is None:
            return
        div = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        note_html = f'<div style="color:#6b7280;font-size:0.8em;margin-top:6px">{note}</div>' if note else ""
        html_parts.append(f'<div class="chart-wrap"><h3 style="color:#fff;margin:0 0 12px 0;font-size:1em">{title}</h3>{div}{note_html}</div>')

    add_fig(f_spread, "Predicted vs. Actual Spread Margin",
            "Green = model predicted correct direction. Dashed = perfect prediction.")
    add_fig(f_teams,  "Team Residuals",
            "Positive = model consistently overestimates team. Negative = underestimates.")
    add_fig(f_week,   "Spread MAE by Week",
            "High early = small sample. High late = garbage time / fatigue effects.")
    add_fig(f_totals, "Predicted vs. Actual Total Score",
            "Green = correct OVER/UNDER call. Dashed = perfect prediction.")

    html_parts.append("</body></html>")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("".join(html_parts))
    print(f"✅ Analysis written → {OUT_HTML}")
    print(f"   {n:,} games · MAE {mae:.1f} pts · R² {r2:.3f} · Within-7 {within:.0%}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CFB model residual analysis")
    parser.add_argument("--season", type=int, default=None,
                        help="Filter to a single season (e.g. 2025). Default: all seasons in CSV.")
    args = parser.parse_args()
    generate_html(season=args.season)
