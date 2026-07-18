#!/usr/bin/env python3
"""Optional: enrich EPC rows with OSM building:levels via postcode match."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_osm_postcodes(osm_dir: Path) -> pd.DataFrame:
    rows = []
    for path in osm_dir.glob("osm_highrise_*.geojson"):
        data = json.loads(path.read_text())
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            pc = props.get("postcode")
            if not pc:
                continue
            rows.append(
                {
                    "postcode": str(pc).upper().replace(" ", ""),
                    "osm_building_levels": pd.to_numeric(props.get("building_levels"), errors="coerce"),
                    "osm_city": props.get("city"),
                    "osm_name": props.get("name"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    osm_dir = ROOT / "data" / "raw" / "osm"
    epc_path = ROOT / "data" / "processed" / "highrise_combined.csv"
    if not epc_path.exists():
        raise SystemExit("Run load_and_verify_data.py first")
    osm = load_osm_postcodes(osm_dir)
    print(f"OSM features with postcode: {len(osm)}")
    if osm.empty:
        print(
            "NOTE: Most OSM features lack addr:postcode. "
            "For a stronger join, download OS Open UPRN from "
            "https://osdatahub.os.uk/ and match on UPRN/coordinates."
        )
        summary = {
            "london_buildings": _count(osm_dir / "osm_highrise_london.geojson"),
            "manchester_buildings": _count(osm_dir / "osm_highrise_manchester.geojson"),
            "birmingham_buildings": _count(osm_dir / "osm_highrise_birmingham.geojson"),
            "postcode_matches_available": 0,
        }
        out = ROOT / "results" / "tables" / "osm_summary.json"
        out.write_text(json.dumps(summary, indent=2))
        print("Wrote", out)
        return

    epc = pd.read_csv(epc_path)
    epc["_pc"] = epc["postcode"].astype(str).str.upper().str.replace(" ", "", regex=False)
    osm = osm.dropna(subset=["osm_building_levels"]).drop_duplicates("_pc" if "_pc" in osm.columns else "postcode")
    osm["_pc"] = osm["postcode"]
    merged = epc.merge(osm[["_pc", "osm_building_levels", "osm_city"]], on="_pc", how="left")
    n = merged["osm_building_levels"].notna().sum()
    print(f"Postcode join rate: {n}/{len(merged)} ({100*n/len(merged):.1f}%)")
    merged.drop(columns=["_pc"], inplace=True)
    merged.to_csv(ROOT / "data" / "processed" / "highrise_combined_osm.csv", index=False)


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(json.loads(path.read_text()).get("features", []))


if __name__ == "__main__":
    main()
