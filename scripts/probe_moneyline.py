"""Probe moneyline EV in walk-forward results."""
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/alex/Desktop/CFB-Betting-Model/outputs/predictions/walk_forward_results.csv')
print('Columns with moneyline/win:', [c for c in df.columns if 'moneyline' in c.lower() or 'win' in c.lower()])
print('Total rows:', len(df))

for col in ['home_moneyline', 'away_moneyline', 'pred_home_win_p']:
    if col in df.columns:
        cov = df[col].notna().mean()
        print(f'{col}: {cov:.0%} coverage')

if 'home_moneyline' not in df.columns:
    print('No moneyline data in walk-forward results.')
    exit()

sample = df[df['home_moneyline'].notna()][
    ['season','home_team','away_team','home_moneyline','away_moneyline','pred_home_win_p','home_win']
].head(8)
print()
print(sample.to_string(index=False))

def ml_to_prob(ml):
    if pd.isna(ml): return None
    ml = float(ml)
    return abs(ml)/(abs(ml)+100) if ml < 0 else 100/(ml+100)

def ml_payout(ml):
    ml = float(ml)
    return ml/100 if ml > 0 else 100/abs(ml)

df2 = df.dropna(subset=['home_moneyline','pred_home_win_p','home_win']).copy()
df2['home_payout'] = df2['home_moneyline'].apply(ml_payout)
df2['away_payout'] = df2['away_moneyline'].apply(ml_payout)
df2['model_home_prob'] = df2['pred_home_win_p']
df2['model_away_prob'] = 1 - df2['pred_home_win_p']
df2['home_ev'] = df2['model_home_prob'] * df2['home_payout'] - df2['model_away_prob']
df2['away_ev'] = df2['model_away_prob'] * df2['away_payout'] - df2['model_home_prob']

print(f'\nMoneyline EV preview ({len(df2)} games with full data):')
print(f"{'EV Threshold':>14} {'Bets':>6} {'Home W':>7} {'Away W':>7} {'P&L':>9} {'ROI':>8}")
print('-' * 60)

for thresh in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    home_bets = df2[df2['home_ev'] >= thresh]
    away_bets = df2[df2['away_ev'] >= thresh]
    n = len(home_bets) + len(away_bets)
    if n == 0: continue

    home_pnl = (home_bets['home_payout'] * home_bets['home_win'] -
                (1 - home_bets['home_win'])).sum()
    away_pnl = (away_bets['away_payout'] * (1 - away_bets['home_win']) -
                away_bets['home_win']).sum()
    total_pnl = home_pnl + away_pnl
    roi = total_pnl / n * 100

    hw = int(home_bets['home_win'].sum())
    aw = int((1 - away_bets['home_win']).sum())

    print(f'  EV >= {thresh:.0%}    {n:>6,}  {hw:>6}  {aw:>6}  {total_pnl:>+8.1f}u  {roi:>+7.1f}%')

# Season breakdown at 5% EV threshold
print('\nSeason breakdown (EV >= 5%):')
thresh = 0.05
for szn in sorted(df2['season'].unique()):
    sub = df2[df2['season'] == szn]
    hb = sub[sub['home_ev'] >= thresh]
    ab = sub[sub['away_ev'] >= thresh]
    n = len(hb) + len(ab)
    if n == 0: continue
    pnl = ((hb['home_payout'] * hb['home_win'] - (1-hb['home_win'])).sum() +
           (ab['away_payout'] * (1-ab['home_win']) - ab['home_win']).sum())
    roi = pnl / n * 100
    print(f'  {int(szn)}: {n:4d} bets  {pnl:+.1f}u  ROI {roi:+.1f}%')
