"""All-market research board, evaluated at executable line/price pairs.

These are candidates, not validated bets. Never infer CORE status from EV.
Prior-season holdouts calibrate uncertainty; current-season ledger results are
reported separately and must not silently become in-sample calibration data.
"""
import math
import numpy as np
import pandas as pd
from decision_quality import POLICY, ev, finite, fresh, payout, probabilities, utc

VERSION = 'all_markets_v1'
MIN_SAMPLES = 300


def calibration_history(history, season):
    required = {'season', 'evaluation_version', 'game_id'}
    if not required.issubset(history.columns):
        return pd.DataFrame()
    h = history[(pd.to_numeric(history.season, errors='coerce') < int(season)) &
                history.evaluation_version.eq('season_holdout_v2')].copy()
    # One outcome must not count multiple times toward confidence.
    return h.drop_duplicates(['season', 'game_id'])


def win_lower_bound(win, samples):
    """95% Wilson lower bound: sampling uncertainty, NOT model-error coverage."""
    z = 1.96
    return (win + z*z/(2*samples) - z*math.sqrt(
        win*(1-win)/samples + z*z/(4*samples*samples))) / (1 + z*z/samples)


def estimate(row, history, market, side, line):
    if market == 'totals':
        return probabilities(history.total_points, history.pred_total,
                             row['pred_total'], line, side)
    if market == 'spreads':
        # The offered handicap belongs to the selected team. For away +7,
        # home margin must be < 7; for home -7, home margin must be > 7.
        threshold = -float(line) if side == 'home' else float(line)
        return probabilities(history.point_diff, history.pred_spread,
                             row['pred_spread'], threshold, side)
    if market != 'h2h':
        raise ValueError('Unknown market')
    p = row.get('pred_win_p', row.get('pred_home_win_p'))
    if not finite(p) or not 0 < float(p) < 1:
        raise ValueError('Missing win probability')
    raw = pd.to_numeric(history.pred_home_win_p, errors='coerce')
    actual = pd.to_numeric(history.home_win, errors='coerce')
    # Fixed probability neighborhood chosen before evaluation. No adaptive
    # window chosen to manufacture positive returns or sufficient samples.
    mask = raw.sub(float(p)).abs().le(.10) & actual.isin([0, 1])
    actual = actual[mask]
    if len(actual) == 0:
        raise ValueError('No comparable moneyline calibration games')
    win = float(actual.mean()) if side == 'home' else float(1-actual.mean())
    return {'win': win, 'loss': 1-win, 'push': 0., 'samples': len(actual)}


def evaluate_markets(row, history, now=None):
    now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    reasons = []
    result = {'version': VERSION, 'decision_at': now.isoformat(),
              'candidates': [], 'reasons': reasons, 'units': 0}
    kickoff = utc(row.get('start_date'))
    if kickoff is None or kickoff <= now:
        reasons.append('missing_or_started_kickoff')
        return result
    h = calibration_history(history, row.get('season', now.year))
    if h.empty:
        reasons.append('no_verified_prior_season_holdouts')
        return result
    quotes = row.get('book_quotes', [])
    if not isinstance(quotes, list):
        quotes = []
    context = []
    if row.get('availability_status') != 'reviewed' or not fresh(
            row.get('availability_observed_at'), now, 24):
        context.append('availability_review_required')
    # A live fetch of odds does not make the fitted model fresh.
    if not fresh(row.get('model_data_asof'), now, 8*24):
        context.append('model_data_freshness_unverified')
    for q in quotes:
        market = q.get('market')
        side = str(q.get('selection', q.get('side', ''))).lower()
        if market == 'totals' and side not in ('under', 'over'):
            continue
        if market in ('spreads', 'h2h') and side not in ('home', 'away'):
            # Only exact CFBD names accepted for older snapshots. New ingestion
            # preserves explicit home/away identity; no ambiguous fuzzy bets.
            names = {str(row.get('home_team')).lower(): 'home',
                     str(row.get('away_team')).lower(): 'away'}
            side = names.get(side)
        if market not in ('totals', 'spreads', 'h2h') or side is None:
            continue
        if not q.get('book') or not all(fresh(q.get(k), now, 1)
                                          for k in ('updated_at', 'observed_at')):
            continue
        quote_kickoff = utc(q.get('kickoff'))
        if quote_kickoff is not None and abs((quote_kickoff-kickoff).total_seconds()) > 3600:
            continue
        try:
            p = estimate(row, h, market, side, q.get('line'))
            expected = ev(p, q['odds'])
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if p['samples'] < MIN_SAMPLES:
            continue
        # Treat uncertainty mass conservatively as losses, not pushes.
        lower = win_lower_bound(p['win'], p['samples'])
        conservative_ev = lower*payout(q['odds']) - (1-lower-p['push'])
        issues = list(context)
        if market == 'totals' and not (row.get('weather_source') and
                fresh(row.get('weather_observed_at'), now, 24) and
                (row.get('is_dome') in (True, 1, '1') or
                 (finite(row.get('wind_speed')) and float(row['wind_speed']) >= 0))):
            issues.append('weather_review_required')
        tier = 'PASS'
        if expected >= .02:
            tier = 'RESEARCH_A' if conservative_ev > 0 else 'RESEARCH_B'
        status = 'REVIEW' if issues else 'PAPER'
        q = {**q, 'selection': side}
        result['candidates'].append({'market': market, 'selection': side,
            'quote': q, 'probabilities': p, 'ev': expected,
            'sampling_lower_ev': conservative_ev, 'tier': tier,
            'status': status, 'reasons': issues + ['prospective_validation_required'],
            'units': 0})
    # Best price-specific EV per market/side. Multiple books aren't extra picks.
    best = {}
    for c in result['candidates']:
        key = (c['market'], c['selection'])
        if key not in best or c['ev'] > best[key]['ev']:
            best[key] = c
    result['candidates'] = sorted(best.values(), key=lambda c: c['ev'], reverse=True)
    if not best:
        reasons.append('no_fresh_priced_quote_with_sufficient_calibration')
    return result


def board_rows(frame):
    """Display all edges with blockers visible; never present them as CORE."""
    rows = []
    for _, game in frame.iterrows():
        decision = game.get('market_decisions', {})
        if not isinstance(decision, dict):
            continue
        for c in decision.get('candidates', []):
            if c['tier'] == 'PASS':
                continue
            q = c['quote']
            side = game.get(c['selection']+'_team', c['selection'].title())
            rows.append({'Game': f"{game['away_team']} at {game['home_team']}",
                'Market': c['market'], 'Pick': side, 'Line': q.get('line'),
                'Odds': q['odds'], 'Book': q['book'], 'Tier': c['tier'],
                'Status': c['status'], 'Estimated EV %': round(100*c['ev'], 2),
                'Sampling lower EV %': round(100*c['sampling_lower_ev'], 2),
                'Win %': round(100*c['probabilities']['win'], 1),
                'Push %': round(100*c['probabilities']['push'], 1),
                'Calibration games': c['probabilities']['samples'],
                'Quote updated': q['updated_at'],
                'Needs attention': ', '.join(c['reasons']).replace('_', ' ')})
    return sorted(rows, key=lambda r: r['Estimated EV %'], reverse=True)
