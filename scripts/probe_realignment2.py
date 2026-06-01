"""
Focused analysis: Power 4 conference realignment and spread pricing inefficiency.
Only looks at moves between major conferences (Big Ten, SEC, Big 12, ACC, Pac-12).
"""
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Build team-season conference lookup (Power 4 only) ────────────────────────
POWER4 = {"Big Ten", "SEC", "Big 12", "ACC", "Pac-12"}

df = pd.read_csv(ROOT / "data" / "processed" / "master_games.csv")
df["season"] = pd.to_numeric(df["season"], errors="coerce")

home = df[["season","home_team","home_conference"]].rename(
    columns={"home_team":"team","home_conference":"conference"})
away = df[["season","away_team","away_conference"]].rename(
    columns={"away_team":"team","away_conference":"conference"})

team_conf = (pd.concat([home, away])
             .dropna(subset=["conference"])
             .drop_duplicates(subset=["season","team"])
             .sort_values(["team","season"])
             .reset_index(drop=True))

# Only track Power 4 conference memberships
team_conf["is_p4"] = team_conf["conference"].isin(POWER4)

# Detect P4 conference changes
team_conf["prev_conf"] = team_conf.groupby("team")["conference"].shift(1)
team_conf["prev_p4"]   = team_conf.groupby("team")["is_p4"].shift(1)
team_conf["new_p4_conf"] = (
    team_conf["is_p4"] &
    (team_conf["conference"] != team_conf["prev_conf"]) &
    team_conf["prev_conf"].notna()
)

# Years in current P4 conference
def conf_years(grp):
    result, count = [], 0
    for changed in grp["new_p4_conf"].values:
        if changed:
            count = 1
        elif grp["is_p4"].values[len(result)]:
            count += 1
        else:
            count = 0
        result.append(count)
    return result

team_conf["p4_conf_years"] = (
    team_conf.groupby("team", group_keys=False)
             .apply(lambda g: pd.Series(conf_years(g), index=g.index))
)

# ── Show meaningful P4 realignment moves ──────────────────────────────────────
print("=== Power 4 conference changes (2019-2026) ===\n")
p4_changes = team_conf[
    team_conf["new_p4_conf"] & (team_conf["season"] >= 2019)
][["season","team","prev_conf","conference"]].sort_values(["season","team"])
for _, r in p4_changes.iterrows():
    print(f"  {int(r['season'])}  {r['team']:<25}  {r['prev_conf']} → {r['conference']}")

# ── Does the market misprice new-P4 entrants? ─────────────────────────────────
print("\n=== ATS performance: new P4 entrants vs. established teams ===")
print("(Using walk-forward OOS predictions)\n")

wf = pd.read_csv(ROOT / "outputs" / "predictions" / "walk_forward_results.csv")
wf = wf.dropna(subset=["covered_spread","spread_edge"]).copy()
wf["covered_spread"] = wf["covered_spread"].astype(int)
wf["season"] = pd.to_numeric(wf["season"], errors="coerce")

# Build lookup: (season, team) -> p4_conf_years
lookup = team_conf.set_index(["season","team"])["p4_conf_years"]

def get_conf_years(season, team):
    try: return lookup.loc[(season, team)]
    except: return None

wf["home_p4_years"] = wf.apply(lambda r: get_conf_years(r["season"], r["home_team"]), axis=1)
wf["away_p4_years"] = wf.apply(lambda r: get_conf_years(r["season"], r["away_team"]), axis=1)

# New entrant = in their first 2 years in a P4 conference
# Need moneyline coverage check
wf["home_is_new"] = wf["home_p4_years"].between(1, 2, inclusive="both")
wf["away_is_new"] = wf["away_p4_years"].between(1, 2, inclusive="both")

# ATS results: positive edge = model likes home, covered_spread=1 means home covered
def ats_stats(subset, label):
    n = len(subset)
    if n < 20:
        return
    # ATS: home covered when covered_spread == 1
    home_ats = subset["covered_spread"].mean()
    # Model edge direction accuracy
    edge_dir = ((subset["spread_edge"] > 0) == (subset["covered_spread"] == 1)).mean()
    print(f"  {label:<45} n={n:4d}  Home ATS: {home_ats:.1%}  Model dir acc: {edge_dir:.1%}")

print("All P4 games (baseline):")
p4_games = wf[wf["home_p4_years"].notna() | wf["away_p4_years"].notna()]
ats_stats(p4_games, "All P4 games")

print("\nNew-conference entrants (year 1-2 in P4):")
ats_stats(wf[wf["home_is_new"]], "New-P4 HOME team")
ats_stats(wf[wf["away_is_new"]], "New-P4 AWAY team (home perspective)")
ats_stats(wf[wf["home_is_new"] & ~wf["away_is_new"]], "New-P4 home vs. established away")
ats_stats(wf[~wf["home_is_new"] & wf["away_is_new"]], "Established home vs. new-P4 away")

print("\nBy specific year in conference:")
for yr in [1, 2]:
    ats_stats(wf[wf["home_p4_years"] == yr], f"Home team in year {yr} of P4 conf")
    ats_stats(wf[wf["away_p4_years"] == yr], f"Away team in year {yr} of P4 conf")

# ── Surprise vs Vegas for new entrants ────────────────────────────────────────
print("\n=== Model edge (pred_spread - vegas_margin) for new vs. established ===")
wf["spread"] = pd.to_numeric(wf["spread"], errors="coerce")
wf["vegas_margin"] = -wf["spread"]

print(f"\n  {'Group':<45} {'n':>5} {'Avg edge':>10} {'Edge std':>10}")
print("  " + "-" * 75)
groups = [
    (wf[wf["home_p4_years"] >= 3],               "Established home (3+ yrs in P4)"),
    (wf[wf["home_p4_years"] == 1],               "New home (year 1 in P4)"),
    (wf[wf["home_p4_years"] == 2],               "New home (year 2 in P4)"),
    (wf[wf["away_p4_years"] == 1],               "New away (year 1 in P4)"),
    (wf[wf["away_p4_years"] == 2],               "New away (year 2 in P4)"),
]
for sub, label in groups:
    if len(sub) < 10: continue
    avg_edge = sub["spread_edge"].mean()
    std_edge = sub["spread_edge"].std()
    print(f"  {label:<45} {len(sub):>5} {avg_edge:>+9.2f}  {std_edge:>9.2f}")

print("\n=== 2026 Power 4 new entrants (potential pricing gaps) ===")
new_2026 = team_conf[
    (team_conf["season"] == 2025) &
    (team_conf["p4_conf_years"].between(1, 2)) &
    (team_conf["is_p4"])
][["team","conference","p4_conf_years"]].sort_values("p4_conf_years")
print(new_2026.to_string(index=False))
