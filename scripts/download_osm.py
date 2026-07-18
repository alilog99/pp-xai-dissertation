"""OSM Overpass download for high-rise buildings (building:levels >= 10)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

CITY_QUERIES = {
    "london": 'area["name"="Greater London"]["admin_level"="5"]->.searchArea;',
    "manchester": 'area["name"="Greater Manchester"]->.searchArea;',
    "birmingham": 'area["name"="Birmingham"]["admin_level"="8"]->.searchArea;',
}


def build_query(area_prelude: str, min_levels: int = 10) -> str:
    return f"""
[out:json][timeout:180];
{area_prelude}
(
  way["building"]["building:levels"](if: number(t["building:levels"]) >= {min_levels})(area.searchArea);
  relation["building"]["building:levels"](if: number(t["building:levels"]) >= {min_levels})(area.searchArea);
);
out center tags;
"""


def download_city(city: str, out_dir: Path, min_levels: int = 10) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"osm_highrise_{city}.geojson"
    if out_path.exists() and out_path.stat().st_size > 100:
        print(f"  Skipping {city} (exists): {out_path}")
        return out_path

    prelude = CITY_QUERIES[city]
    query = build_query(prelude, min_levels=min_levels)
    headers = {
        "User-Agent": "PP-XAI-Dissertation/1.0 (University of Hull MSc; academic research)",
        "Accept": "application/json",
    }
    endpoints = [
        OVERPASS_URL,
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]
    print(f"  Querying Overpass for {city}...")
    last_err: Exception | None = None
    data = None
    for url in endpoints:
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"    endpoint failed ({url}): {exc}")
            time.sleep(2)
    if data is None:
        raise RuntimeError(f"All Overpass endpoints failed for {city}: {last_err}")

    features = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        levels = tags.get("building:levels")
        center = el.get("center") or {}
        lon = center.get("lon") or el.get("lon")
        lat = center.get("lat") or el.get("lat")
        if lon is None or lat is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "osm_id": el.get("id"),
                    "osm_type": el.get("type"),
                    "building_levels": levels,
                    "name": tags.get("name"),
                    "building": tags.get("building"),
                    "city": city,
                    "postcode": tags.get("addr:postcode"),
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(geojson))
    print(f"  Saved {len(features)} features → {out_path}")
    return out_path


def download_all(out_dir: str | Path | None = None) -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    target = Path(out_dir or root / "data" / "raw" / "osm")
    paths = []
    for city in CITY_QUERIES:
        try:
            paths.append(download_city(city, target))
            time.sleep(5)  # be polite to Overpass
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: OSM download failed for {city}: {exc}")
    return paths


if __name__ == "__main__":
    download_all()
