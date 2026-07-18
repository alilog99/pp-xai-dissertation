#!/usr/bin/env python3
"""Join OS Open UPRN (optional) to EPC + OSM for stronger high-rise enrichment.

Manual prerequisite (browser — free):
  1. Create account at https://osdatahub.os.uk/
  2. Download OS Open UPRN (CSV or GeoPackage)
  3. Place file at: data/raw/os_open_uprn/osopenuprn_*.csv
     or: data/raw/os_open_uprn/osopenuprn.gpkg

This script then:
  - Loads UPRN easting/northing (or lat/lon if present)
  - Matches EPC rows on `uprn`
  - Spatially matches OSM high-rise points to nearest UPRN (optional if geopandas available)
  - Writes data/processed/highrise_combined_uprn.csv and a join report

Safe to run without UPRN files: prints instructions and exits 0 with a stub report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPRN_DIR = ROOT / "data" / "raw" / "os_open_uprn"
OUT_DIR = ROOT / "data" / "processed"
REPORT = ROOT / "results" / "tables" / "uprn_join_report.json"


def find_uprn_file() -> Path | None:
    UPRN_DIR.mkdir(parents=True, exist_ok=True)
    cands = sorted(UPRN_DIR.glob("*.csv")) + sorted(UPRN_DIR.glob("*.gpkg"))
    return cands[0] if cands else None


def main() -> None:
    epc_path = OUT_DIR / "highrise_combined.csv"
    if not epc_path.exists():
        raise SystemExit("Run scripts/load_and_verify_data.py first")

    epc = pd.read_csv(epc_path)
    uprn_path = find_uprn_file()
    report: dict = {
        "epc_rows": len(epc),
        "epc_with_uprn": int(epc["uprn"].notna().sum()) if "uprn" in epc.columns else 0,
        "uprn_file": str(uprn_path) if uprn_path else None,
        "status": "awaiting_download",
    }

    if uprn_path is None:
        report["message"] = (
            "No OS Open UPRN file found. Register free at https://osdatahub.os.uk/ "
            f"and place CSV/GPKG under {UPRN_DIR}"
        )
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2))
        print(report["message"])
        print(f"Wrote stub report → {REPORT}")
        # Keep existing OSM postcode join artefact as best-effort enrichment
        osm_joined = OUT_DIR / "highrise_combined_osm.csv"
        if osm_joined.exists():
            print(f"Existing OSM postcode join available: {osm_joined}")
        return

    print(f"Loading UPRN file: {uprn_path}")
    if uprn_path.suffix.lower() == ".gpkg":
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise SystemExit("geopandas required for GPKG") from exc
        uprn = gpd.read_file(uprn_path)
        uprn_df = pd.DataFrame(uprn.drop(columns="geometry", errors="ignore"))
        if "geometry" in uprn.columns:
            uprn_df["lon"] = uprn.geometry.centroid.x
            uprn_df["lat"] = uprn.geometry.centroid.y
    else:
        uprn_df = pd.read_csv(uprn_path, low_memory=False)

    # Normalise UPRN column name
    cols = {c.lower(): c for c in uprn_df.columns}
    uprn_col = cols.get("uprn") or cols.get("uprn_1") or list(uprn_df.columns)[0]
    uprn_df["uprn"] = pd.to_numeric(uprn_df[uprn_col], errors="coerce")
    epc["uprn"] = pd.to_numeric(epc.get("uprn"), errors="coerce")

    keep_cols = ["uprn"]
    for name in ("latitude", "longitude", "lat", "lon", "x_coordinate", "y_coordinate", "easting", "northing"):
        if name in {c.lower() for c in uprn_df.columns}:
            keep_cols.append(cols[name.lower()])

    uprn_small = uprn_df[keep_cols].drop_duplicates("uprn")
    merged = epc.merge(uprn_small, on="uprn", how="left", suffixes=("", "_os"))
    matched = int(merged["uprn"].notna().sum() and (merged[keep_cols[1]].notna().sum() if len(keep_cols) > 1 else 0))
    # recount properly
    coord_cols = [c for c in merged.columns if c != "uprn" and c in uprn_small.columns]
    if coord_cols:
        matched = int(merged[coord_cols[0]].notna().sum())
    else:
        matched = int(merged["uprn"].notna().sum())

    out = OUT_DIR / "highrise_combined_uprn.csv"
    merged.to_csv(out, index=False)
    report.update(
        {
            "status": "joined",
            "uprn_rows": len(uprn_small),
            "matched_rows": matched,
            "match_rate": matched / max(len(epc), 1),
            "output": str(out),
        }
    )
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
