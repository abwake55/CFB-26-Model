#!/usr/bin/env python3
"""Incrementally download CFBD drives. No secret or request URL is logged."""
import argparse
import json
import os
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]


def refresh(seasons):
    key=os.environ.get('CFB_API_KEY')
    if not key:
        raise RuntimeError('CFB_API_KEY is required to collect possession data')
    directory=ROOT/'data/raw/drives'; directory.mkdir(parents=True,exist_ok=True)
    for year in seasons:
        for kind in ('regular','postseason'):
            path=directory/f'{year}_{kind}.json'
            # Historical seasons frozen locally; current season refreshes.
            from datetime import datetime, timezone
            if path.exists() and year<datetime.now(timezone.utc).year:
                continue
            r=requests.get('https://api.collegefootballdata.com/drives',params={'year':year,'seasonType':kind},
                           headers={'Authorization':f'Bearer {key}'},timeout=90)
            r.raise_for_status(); rows=r.json()
            if not isinstance(rows,list):
                raise ValueError('Drive endpoint did not return a list')
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(rows));temp.replace(path)
            print(f'{year} {kind}: {len(rows)} drives')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=2018);p.add_argument('--end',type=int,default=2026)
    a=p.parse_args();refresh(range(a.start,a.end+1))
