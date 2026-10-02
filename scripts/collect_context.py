#!/usr/bin/env python3
"""Archive forecasts, or ingest sourced availability/analyst reports as JSON."""
import argparse
from pathlib import Path
import json
import sys
import pandas as pd
import requests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pregame_context import ingest
from decision_quality import utc
from weather import TEAM_VENUES


def collect_weather(games, now=None):
    now = utc(now) if now is not None else pd.Timestamp.now(tz='UTC')
    count = 0
    for _, g in games.iterrows():
        start = utc(g.get('start_date'))
        if start is None or start <= now or start > now + pd.Timedelta(days=7):
            continue
        venue = TEAM_VENUES.get(g['home_team'])
        # Do not invent a venue for neutral games.
        if venue is None or g.get('neutral_site') in (True, 1, '1'):
            continue
        lat, lon, dome = venue
        try:
            payload = {'is_dome': int(dome)}
            if dome:
                payload.update(wind_speed=0, temp_avg=68, precipitation=0)
                source = 'https://github.com/abwake55/CFB-26-Model/blob/main/src/weather.py'
            else:
                source = 'https://api.open-meteo.com/v1/forecast'
                r = requests.get(source, params={'latitude':lat,'longitude':lon,
                    'hourly':'wind_speed_10m', 'daily':'temperature_2m_max,temperature_2m_min,precipitation_sum',
                    'wind_speed_unit':'mph','temperature_unit':'fahrenheit','precipitation_unit':'inch',
                    'timezone':'UTC','forecast_days':8}, timeout=20)
                r.raise_for_status(); data=r.json()
                h=data['hourly']; times=pd.to_datetime(h['time'],utc=True)
                indices=[i for i,t in enumerate(times) if start.floor('h') <= t < start.floor('h')+pd.Timedelta(hours=4)]
                values=[h['wind_speed_10m'][i] for i in indices]
                if len(values)!=4 or any(v is None for v in values):
                    continue
                d=data['daily']; j=d['time'].index(start.date().isoformat())
                payload.update(wind_speed=max(values),temp_avg=(d['temperature_2m_max'][j]+d['temperature_2m_min'][j])/2,
                               precipitation=d['precipitation_sum'][j])
            ingest({'game_id':int(g.game_id),'kind':'weather','source_url':source,
                    'published_at':now.isoformat(),'payload':payload},now=now)
            count+=1
        except (requests.RequestException, ValueError, KeyError, TypeError):
            print(f'Weather unavailable for game {g.game_id}; review required')
    return count


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--report'); p.add_argument('--weather',action='store_true'); args=p.parse_args()
    if args.report:
        print(ingest(json.loads(Path(args.report).read_text())))
    elif args.weather:
        print('Forecasts archived:',collect_weather(pd.read_csv(ROOT/'data/processed/master_games.csv')))
    else:
        p.error('Choose --report FILE or --weather')
