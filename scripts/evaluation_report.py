"""Report market baselines, per-season performance and CORE uncertainty.

    python scripts/evaluation_report.py --input outputs/predictions/walk_forward_results.csv

All CORE returns assume -110. Historical CFBD lines have no decision-time
timestamp, so these are retrospective simulations, not executable ROI proof.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from betting import settle
import gates


def wilson_interval(wins, n):
    if not n:
        return [None, None]
    z = 1.95996398454
    p = wins / n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(centre-radius), float(centre+radius)]


def core_stats(d):
    selected = d[d.apply(gates.core_total, axis=1)].copy()
    settled = [settle(r.total_points, r.over_under, -1) for r in selected.itertuples()]
    wins = sum(r[0] == "win" for r in settled)
    losses = sum(r[0] == "loss" for r in settled)
    profit = float(sum(r[1] for r in settled))
    risk = float(sum(r[2] for r in settled))
    roi_ci = [None, None]
    if settled:
        selected["pnl"] = [r[1] for r in settled]
        selected["risk"] = [r[2] for r in settled]
        # Sample entire season/week blocks to preserve within-week correlation.
        blocks = selected.groupby(["season", "week"])[["pnl", "risk"]].sum().to_numpy()
        if len(blocks) >= 2:
            rng = np.random.default_rng(42)
            draws = blocks[rng.integers(0, len(blocks), size=(2000, len(blocks)))].sum(axis=1)
            roi_ci = np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist()
    return {"bets": len(settled), "wins": wins, "losses": losses,
            "pushes": len(settled)-wins-losses, "profit_units": profit,
            "roi": profit/risk if risk else None,
            "win_rate": wins/(wins+losses) if wins+losses else None,
            "win_rate_95pct_wilson": wilson_interval(wins, wins+losses),
            "roi_95pct_week_block_bootstrap": roi_ci}


def build_report(predictions, features):
    predictions = predictions.drop_duplicates()
    if predictions.game_id.duplicated().any():
        raise ValueError("Conflicting duplicate predictions")
    context = ["game_id", "home_conference", "away_conference", "wind_speed", "is_dome"]
    extra = [c for c in context if c not in predictions or c == "game_id"]
    source = features[extra].drop_duplicates()
    d = predictions.merge(source, on="game_id", how="left", validate="one_to_one")
    d["totals_edge"] = d.pred_total - d.over_under

    def metrics(g):
        return {"games": len(g),
                "spread_mae": float((g.pred_spread-g.point_diff).abs().mean()),
                "market_spread_mae": float((g.vegas_home_margin-g.point_diff).abs().mean()),
                "totals_mae": float((g.pred_total-g.total_points).abs().mean()),
                "market_totals_mae": float((g.over_under-g.total_points).abs().mean()),
                "brier": float(((g.pred_home_win_p-g.home_win)**2).mean()),
                "core": core_stats(g)}

    return {"evaluation_version": str(d.evaluation_version.iloc[0]) if "evaluation_version" in d else "legacy",
            "optimizer": sorted(d.optimizer.unique().tolist()) if "optimizer" in d else ["unknown"],
            "assumed_odds": -110,
            "limitations": ["CORE rules were selected retrospectively; confidence intervals do not correct strategy selection bias.",
                            "Historical market-line timestamps are unavailable; ROI is not proof of decision-time profitability.",
                            "Historical weather is observed weather, not archived forecasts.",
                            "Model architecture was developed using these seasons; this is not an untouched research holdout."],
            "overall": metrics(d),
            "by_season": {str(int(s)): metrics(g) for s, g in d.groupby("season")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "outputs/predictions/walk_forward_results.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/predictions/validation_summary.json")
    args = ap.parse_args()
    report = build_report(pd.read_csv(args.input), pd.read_csv(ROOT / "data/processed/feature_matrix.csv"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
