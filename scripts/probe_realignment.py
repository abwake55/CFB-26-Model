"""
Map conference changes across seasons using master_games.csv.
Identifies teams that changed conferences and quantifies how many
games involve new-conference entrants (a potential market inefficiency).

Run: /opt/homebrew/bin/python3 scripts/probe_realignment.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "master_games.csv")
df["season"] = pd.to_numeric(df["season"], errors="coerce")

# ── Build team-season conference lookup ────────────────────────────────────────
# Each game gives us (season, team, conference) for both home and away
home = df[["season", "home_team", "home_conference"]].rename(
    columns={"home_team": "team", "home_conference": "conference"})
away = df[["season", "away_team", "away_conference"]].rename(
    columns={"away_team": "team", "away_conference": "conference"})
team_conf = pd.concat([home, away]).dropna(subset=["conference"])
team_conf = (team_conf.drop_duplicates(subset=["season", "team"])
                      .sort_values(["team", "season"])
                      .reset_index(drop=True))

# ── Detect conference changes ──────────────────────────────────────────────────
team_conf["prev_conf"] = team_conf.groupby("team")["conference"].shift(1)
team_conf["changed"]   = (team_conf["conference"] != team_conf["prev_conf"]) & team_conf["prev_conf"].notna()

# Years in current conference (consecutive streak) — vectorised version
team_conf = team_conf.sort_values(["team", "season"]).reset_index(drop=True)
team_conf["conf_years"] = (
    team_conf.groupby("team")["changed"]
             .transform(lambda x: x.cumsum())
             .rsub(team_conf.groupby("team").cumcount())
             .add(1)
)
# Simpler: just count consecutive same-conference seasons
def streak(changed_series):
    result, count = [], 1
    for i, changed in enumerate(changed_series):
        if i == 0 or not changed:
            result.append(count)
            count += 1
        else:
            count = 1
            result.append(count)
    return result

streaks = []
for _, grp in team_conf.groupby("team"):
    s = streak(grp["changed"].values)
    streaks.extend(s)
team_conf["conf_years"] = streaks

# ── Show all conference changes 2019-2026 ─────────────────────────────────────
print("=== Conference changes detected (2019–2026) ===\n")
changes = team_conf[(team_conf["changed"]) & (team_conf["season"] >= 2019)]
changes = changes.sort_values(["season", "team"])
for _, row in changes.iterrows():
    print(f"  {int(row['season'])}  {row['team']:<25}  {row['prev_conf']} → {row['conference']}")

# ── How many games involve a new-conference team? ─────────────────────────────
print("\n=== Games involving teams in year 1 or 2 of a new conference ===\n")
new_teams = team_conf[team_conf["conf_years"] <= 2][["season", "team", "conf_years"]]

# Merge into games
df2 = df.merge(new_teams.rename(columns={"team": "home_team", "conf_years": "home_conf_years"}),
               on=["season", "home_team"], how="left")
df2 = df2.merge(new_teams.rename(columns={"team": "away_team", "conf_years": "away_conf_years"}),
               on=["season", "away_team"], how="left")
df2["home_conf_years"] = df2["home_conf_years"].fillna(99)
df2["away_conf_years"] = df2["away_conf_years"].fillna(99)
df2["either_new"] = ((df2["home_conf_years"] <= 2) | (df2["away_conf_years"] <= 2)).astype(int)
df2["both_new"]   = ((df2["home_conf_years"] <= 2) & (df2["away_conf_years"] <= 2)).astype(int)

for szn in sorted(df2["season"].unique()):
    if szn < 2022: continue
    sub = df2[df2["season"] == szn]
    n_total    = len(sub)
    n_either   = sub["either_new"].sum()
    n_both     = sub["both_new"].sum()
    pct        = n_either / n_total * 100 if n_total else 0
    print(f"  {int(szn)}: {n_total:4d} games  |  {n_either:3d} ({pct:.0f}%) involve a new-conf team  |  {n_both} both new")

# ── Does having a new-conference team create a pricing edge? ──────────────────
print("\n=== Point differential surprise for new-conference teams (2022-2025) ===")
print("(Positive = new-conf home team outperformed Vegas expectation)\n")

for szn in [2022, 2023, 2024, 2025]:
    sub = df2[(df2["season"] == szn) & df2["point_diff"].notna() & df2["vegas_home_margin"].notna()
              if "vegas_home_margin" in df2.columns else df2["season"] == szn].copy() \
          if "vegas_home_margin" in df2.columns else pd.DataFrame()
    if sub.empty:
        continue
    sub["surprise"] = sub["point_diff"] - sub.get("vegas_home_margin", 0)
    new_home = sub[sub["home_conf_years"] <= 2]["surprise"]
    est_home = sub[sub["home_conf_years"] > 2]["surprise"]
    new_away = sub[sub["away_conf_years"] <= 2]["surprise"]
    if len(new_home) > 10:
        print(f"  {int(szn)} — New-conf home team vs Vegas:  "
              f"mean surprise = {new_home.mean():+.2f} pts  (n={len(new_home)})")
    if len(new_away) > 10:
        print(f"  {int(szn)} — New-conf away team (home perspective):  "
              f"mean surprise = {new_away.mean():+.2f} pts  (n={len(new_away)})")

print("\n=== Top realignment teams for 2026 ===")
latest = team_conf[team_conf["season"] == 2025][["team", "conference", "conf_years"]]
new_2026 = latest[latest["conf_years"] <= 2].sort_values("conf_years")
print(new_2026.to_string(index=False))
print(f"\nTotal teams in year 1-2 of new conference entering 2026: {len(new_2026)}")
