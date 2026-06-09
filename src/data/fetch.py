"""
src/data/fetch.py
-----------------
Pull recent PM2.5, NO2, and O3 readings from the OpenAQ v3 public API.
Saves a flat CSV to data/raw/latest.csv.

v3 API changes vs v2:
  - No city filter — use coordinates + radius instead
  - Measurements are sensor-based: /v3/sensors/{sensor_id}/measurements
  - Each location carries a "sensors" array (no need to iterate parameters separately)
  - API key required — set OPENAQ_API_KEY in your .env file

Run directly:
    python -m src.data.fetch --lat 12.9716 --lon 77.5946 --radius 25000
    python -m src.data.fetch  # uses Bengaluru defaults

Or as a DVC stage (see dvc.yaml).
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAQ_BASE = "https://api.openaq.org/v3"
PARAMETERS = {"pm25", "no2", "o3"}   # WHO core pollutants we care about
RAW_DIR = Path("data/raw")

# Bengaluru city centre defaults
DEFAULT_LAT = 12.9716
DEFAULT_LON = 77.5946
DEFAULT_RADIUS = 25000   # metres — covers the city


def _headers() -> dict:
    """Build request headers. Raises clearly if API key is missing."""
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAQ_API_KEY not set. "
            "Get a free key at https://explore.openaq.org and add it to your .env file."
        )
    return {"X-API-Key": api_key}


def fetch_locations(lat: float, lon: float, radius: int, limit: int = 100) -> list[dict]:
    """
    Return sensor locations within `radius` metres of (lat, lon).

    v3 uses coordinate-based filtering instead of city names.
    Each result includes a 'sensors' array listing every sensor at that location.
    """
    resp = requests.get(
        f"{OPENAQ_BASE}/locations",
        headers=_headers(),
        params={
            "coordinates": f"{lat},{lon}",
            "radius": radius,
            "limit": limit,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def fetch_sensor_measurements(sensor_id: int, limit: int = 500) -> list[dict]:
    """
    Return recent measurements for a single sensor.

    v3 endpoint: GET /v3/sensors/{sensor_id}/measurements
    Response rows look like:
        {
          "value": 23.4,
          "parameter": {"name": "pm25", "units": "µg/m³"},
          "period": {"datetimeTo": {"utc": "2024-06-09T10:00:00Z"}},
          "coordinates": {"latitude": 12.97, "longitude": 77.59}
        }
    """
    resp = requests.get(
        f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
        headers=_headers(),
        params={"limit": limit},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def flatten_measurements(
    raw: list[dict],
    location_id: int,
    location_name: str,
    city_label: str,
) -> pd.DataFrame:
    """
    Convert raw v3 sensor measurement results into a flat DataFrame.

    Key v3 differences handled here:
      - parameter is a dict: r["parameter"]["name"], not a plain string
      - timestamp lives at r["period"]["datetimeTo"]["utc"]
      - coordinates may be missing for some sensors (default to None)
    """
    rows = []
    for r in raw:
        param_info = r.get("parameter", {})
        period = r.get("period", {})
        coords = r.get("coordinates") or {}

        rows.append({
            "city": city_label,
            "location_id": location_id,
            "location_name": location_name,
            "parameter": param_info.get("name"),           # "pm25" / "no2" / "o3"
            "value": r.get("value"),
            "unit": param_info.get("units"),
            "timestamp": (
                period.get("datetimeTo", {}).get("utc")    # preferred
                or period.get("dateFrom", {}).get("utc")   # fallback
            ),
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
        })
    return pd.DataFrame(rows)


def main(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    radius: int = DEFAULT_RADIUS,
    city_label: str = "Bengaluru",
    max_locations: int = 20,
    measurements_per_sensor: int = 200,
):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching OpenAQ v3 data near ({lat}, {lon}), radius={radius}m")

    locations = fetch_locations(lat, lon, radius, limit=max_locations)
    print(f"  Found {len(locations)} locations")

    if not locations:
        raise RuntimeError(
            "No locations returned. Check your coordinates, radius, or API key."
        )

    all_frames = []

    for loc in locations:
        loc_id = loc["id"]
        loc_name = loc.get("name", str(loc_id))
        sensors = loc.get("sensors", [])

        # Filter to only the pollutants we need
        relevant = [
            s for s in sensors
            if s.get("parameter", {}).get("name") in PARAMETERS
        ]

        if not relevant:
            continue

        print(f"  Location {loc_id} ({loc_name}): {len(relevant)} relevant sensors")

        for sensor in relevant:
            sensor_id = sensor["id"]
            param_name = sensor["parameter"]["name"]
            try:
                raw = fetch_sensor_measurements(sensor_id, limit=measurements_per_sensor)
                if raw:
                    df = flatten_measurements(raw, loc_id, loc_name, city_label)
                    all_frames.append(df)
                    print(f"    sensor {sensor_id} ({param_name}): {len(raw)} rows")
                time.sleep(0.25)   # respect rate limits
            except requests.HTTPError as e:
                print(f"    Skipping sensor {sensor_id} ({param_name}): {e}")

    if not all_frames:
        raise RuntimeError(
            "No measurements fetched. "
            "The sensors in this area may not have recent data, "
            "or your API key may lack permissions."
        )

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.dropna(subset=["value", "timestamp"])
    # Drop any rows where parameter somehow slipped through the filter
    combined = combined[combined["parameter"].isin(PARAMETERS)]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    timestamped_path = RAW_DIR / f"openaq_{city_label.lower()}_{ts}.csv"
    combined.to_csv(timestamped_path, index=False)
    print(f"\n  Saved {len(combined)} rows → {timestamped_path}")

    # Stable filename DVC tracks
    combined.to_csv(RAW_DIR / "latest.csv", index=False)
    print(f"  Also wrote data/raw/latest.csv")

    # Quick sanity check
    print(f"\n  Parameter breakdown:")
    print(combined["parameter"].value_counts().to_string())
    print(f"\n  Date range: {combined['timestamp'].min()} → {combined['timestamp'].max()}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OpenAQ v3 air quality data")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Centre latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Centre longitude")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS, help="Search radius in metres")
    parser.add_argument("--city", default="Bengaluru", help="Label for the city column")
    parser.add_argument("--max-locations", type=int, default=20)
    parser.add_argument("--measurements-per-sensor", type=int, default=200)
    args = parser.parse_args()

    main(
        lat=args.lat,
        lon=args.lon,
        radius=args.radius,
        city_label=args.city,
        max_locations=args.max_locations,
        measurements_per_sensor=args.measurements_per_sensor,
    )