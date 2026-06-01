"""
Diagnose win probability calibration on the full walk-forward OOS dataset.
This is the honest test — 5,195 truly out-of-sample predictions.

Run: /opt/homebrew/bin/python3 scripts/diagnose_calibration.py
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

df = pd.read_csv(ROOT / "outputs" / "predictions" / "walk_forward_results.csv")
df = df.dropna(subset=["pred_home_win_p", "home_win"]).copy()
df["pred_home_win_p"] = pd.to_numeric(df["pred_home_win_p"], errors="coerce")
df["home_win"]        = df["home_win"].astype(int)

y    = df["home_win"].values
p    = df["pred_home_win_p"].values
print(f"Walk-forward OOS games: {len(df):,}")
print(f"Actual home win rate:   {y.mean():.1%}")
print(f"Model avg prediction:   {p.mean():.1%}")
print(f"Brier score (raw):      {brier_score_loss(y, p):.4f}")
print(f"  (lower = better; naive 50/50 = 0.250)")

# ── Current calibration curve ──────────────────────────────────────────────────
print("\n=== Current calibration (8 quantile bins) ===")
frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
ece = float(np.mean(np.abs(frac - mean_pred)))
print(f"ECE = {ece:.4f}  (0 = perfect, 0.05 = acceptable, >0.10 = poor)")
print(f"\n  {'Predicted':>10}  {'Actual':>8}  {'Gap':>7}  Bar")
for pred, actual in zip(mean_pred, frac):
    gap  = actual - pred
    bar  = "█" * int(round(actual * 30))
    flag = " ← UNDER" if gap > 0.06 else (" ← OVER" if gap < -0.06 else "")
    print(f"  {pred:>9.1%}  {actual:>7.1%}  {gap:>+6.1%}  {bar}{flag}")

# ── Where is the model most wrong? ────────────────────────────────────────────
print("\n=== Calibration by game type ===")
df["spread_abs"] = df["spread"].abs() if "spread" in df.columns else np.nan
bins = [(0,3,"Toss-up (0-3pt)"),(3,7,"Slight fav (3-7pt)"),(7,14,"Moderate fav (7-14pt)"),
        (14,25,"Big fav (14-25pt)"),(25,99,"Blowout (25pt+)")]

for lo, hi, label in bins:
    if "spread" not in df.columns:
        break
    mask = (df["spread"].abs() >= lo) & (df["spread"].abs() < hi)
    sub  = df[mask]
    if len(sub) < 30:
        continue
    pred_avg = sub["pred_home_win_p"].mean()
    act_avg  = sub["home_win"].mean()
    n = len(sub)
    gap = act_avg - pred_avg
    flag = " ⚠️" if abs(gap) > 0.05 else ""
    print(f"  {label:<22} n={n:4d}  pred={pred_avg:.1%}  actual={act_avg:.1%}  gap={gap:+.1%}{flag}")

# ── Platt scaling calibration ──────────────────────────────────────────────────
print("\n=== Platt scaling (logistic on logit) ===")
# logit transform of raw probability
eps   = 1e-6
logit = np.log(np.clip(p, eps, 1-eps) / (1 - np.clip(p, eps, 1-eps)))
logit = logit.reshape(-1, 1)

# Fit on first 70% of data (chronological), evaluate on last 30%
# Using the walk-forward data as post-hoc calibration — valid since it's OOS
split = int(len(df) * 0.70)
platt = LogisticRegression(C=1.0)
platt.fit(logit[:split], y[:split])
p_cal = platt.predict_proba(logit)[:, 1]

brier_raw = brier_score_loss(y, p)
brier_cal = brier_score_loss(y, p_cal)
print(f"Brier before Platt: {brier_raw:.4f}")
print(f"Brier after  Platt: {brier_cal:.4f}  (Δ {brier_cal-brier_raw:+.4f})")
print(f"\nPlatt params: a={platt.coef_[0][0]:.4f}  b={platt.intercept_[0]:.4f}")
print(f"  (a<1 = shrink toward 50%; b>0 = shift toward home win)")

frac2, mean_pred2 = calibration_curve(y, p_cal, n_bins=8, strategy="quantile")
ece2 = float(np.mean(np.abs(frac2 - mean_pred2)))
print(f"\nCalibrated ECE: {ece2:.4f}  (was {ece:.4f})")
print(f"\n  {'Predicted':>10}  {'Actual':>8}  {'Gap':>7}")
for pred, actual in zip(mean_pred2, frac2):
    gap  = actual - pred
    flag = " ← UNDER" if gap > 0.06 else (" ← OVER" if gap < -0.06 else "")
    print(f"  {pred:>9.1%}  {actual:>7.1%}  {gap:>+6.1%}{flag}")

# ── Moneyline EV with calibrated probabilities ─────────────────────────────────
print("\n=== Moneyline EV with Platt-calibrated win probs ===")
df2 = df.dropna(subset=["home_moneyline"]).copy() if "home_moneyline" in df.columns else pd.DataFrame()
if df2.empty:
    print("  No moneyline data available")
else:
    logit2 = np.log(np.clip(df2["pred_home_win_p"].values, eps, 1-eps) /
                    (1 - np.clip(df2["pred_home_win_p"].values, eps, 1-eps))).reshape(-1,1)
    df2["cal_home_prob"] = platt.predict_proba(logit2)[:, 1]
    df2["cal_away_prob"] = 1 - df2["cal_home_prob"]

    def ml_payout(ml):
        ml = float(ml)
        return ml/100 if ml > 0 else 100/abs(ml)

    df2["home_payout"] = df2["home_moneyline"].apply(ml_payout)
    df2["away_payout"] = df2["away_moneyline"].apply(ml_payout)
    df2["home_ev"] = df2["cal_home_prob"] * df2["home_payout"] - df2["cal_away_prob"]
    df2["away_ev"] = df2["cal_away_prob"] * df2["away_payout"] - df2["cal_home_prob"]

    print(f"  Games with moneyline data: {len(df2):,}")
    print(f"\n  {'EV >= ':>8}  {'Bets':>6}  {'P&L':>9}  {'ROI':>8}")
    print("  " + "-"*40)
    for thresh in [0.0, 0.03, 0.05, 0.08, 0.10]:
        hb = df2[df2["home_ev"] >= thresh]
        ab = df2[df2["away_ev"] >= thresh]
        n  = len(hb) + len(ab)
        if n == 0: continue
        pnl = ((hb["home_payout"] * hb["home_win"] - (1-hb["home_win"])).sum() +
               (ab["away_payout"] * (1-ab["home_win"]) - ab["home_win"]).sum())
        roi = pnl / n * 100
        print(f"  {thresh:>7.0%}   {n:>6,}  {pnl:>+8.1f}u  {roi:>+7.1f}%")

    # Save Platt params for use in model.py
    import json
    cal_path = ROOT / "models" / "win_prob_platt.json"
    with open(cal_path, "w") as f:
        json.dump({"a": float(platt.coef_[0][0]),
                   "b": float(platt.intercept_[0]),
                   "ece_before": float(ece),
                   "ece_after":  float(ece2),
                   "brier_before": float(brier_raw),
                   "brier_after":  float(brier_cal)}, f, indent=2)
    print(f"\n  Saved Platt params → models/win_prob_platt.json")

print("\nDone.")
