import os
import json
from dotenv import load_dotenv
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import requests

# 1. Load the key from your .env file
load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError("Missing API key! Check your .env file.")

# 2. Create a headers dictionary to pass to every request
HEADERS = {
    "X-API-Key": API_KEY
}

OPENAQ_BASE = "https://api.openaq.org/v3"
PARAMETERS = ['pm25', 'no2', 'o3']
RAW_DIR = Path('data/raw')

def fetch_locations(city, limit=100) -> list[dict]:
    resp = requests.get(
        f'{OPENAQ_BASE}/locations',
        params={'city': city, 'limit': limit},
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get('results', [])

def fetch_measurements(location_id, parameter, limit=1000) -> list[dict]:
    resp = requests.get(
        f'{OPENAQ_BASE}/measurements',
        params={
            'location_id':location_id,
            'parameter':parameter,
            'limit':limit,
            'sort':'desc',
        },
        headers=HEADERS,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json().get('results', [])

def flatten_mesurements(raw: list[dict], city) ->pd.DataFrame:
    rows = []
    for r in raw:
        rows.append(
            {
                "city": city,
                "location_id": r.get("locationId"),
                "location_name": r.get("location"),
                "parameter": r.get("parameter"),
                "value": r.get("value"),
                "unit": r.get("unit"),
                "timestamp": r.get("date", {}).get("utc"),
                "latitude": r.get("coordinates", {}).get("latitude"),
                "longitude": r.get("coordinates", {}).get("longitude"),
            }
        )
    return pd.DataFrame(rows)

def main(city: str = "Bengaluru", limit: int = 5000):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching OpenAQ data for: {city}")

    locations = fetch_locations(city, limit=50)
    print(f"    Found {len(locations)} sensor locations")

    all_frames = []
    for loc in locations[:20]:
        loc_id = loc['id']
        for param in PARAMETERS:
            try:
                raw = fetch_measurements(loc_id, param, limit=limit//20)
                if raw:
                    df = flatten_mesurements(raw, city)
                    all_frames.append(df)
                time.sleep(0.2)
            except requests.HTTPError as e:
                print(f"   Skipping loc={loc_id} param={param}: {e}")

    if not all_frames:
        raise RuntimeError("No data fetched - check city name or API availability")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.dropna(subset=['value', 'timestamp'])

    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    out_path = RAW_DIR / f"openaq_{city.lower()}_{ts}.csv"
    combined.to_csv(out_path, index=False)
    print(f"  Saved {len(combined)} rows → {out_path}")

    combined.to_csv(RAW_DIR / 'latest.csv', index=False)
    print(f"  Also wrote data/raw/latest.csv ({len(combined)} rows)")

    return combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="Bengaluru")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    main(args.city, args.limit)