"""Chunked EPC data loading, high-rise filtering, and city partitioning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.column_definitions import (
    ALL_CLIENT_AUTHORITIES,
    DOMESTIC_MIN_STOREY_COUNT,
    DOMESTIC_TARGET,
    DOMESTIC_TARGET_MAX,
    DOMESTIC_USECOLS,
    EXCLUDED_PROPERTY_TYPE_KEYWORDS,
    FEDERATED_CLIENTS,
    NONDOMESTIC_MIN_FLOOR_AREA,
    NONDOMESTIC_TARGET,
    NONDOMESTIC_TARGET_MAX,
    NONDOMESTIC_USECOLS,
    RAW_DOMESTIC_DIR,
    RAW_NONDOMESTIC_DIR,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assign_city(local_authority_label: str | float | None) -> str | None:
    """Map a local authority label to london / manchester / birmingham."""
    if local_authority_label is None or (isinstance(local_authority_label, float) and np.isnan(local_authority_label)):
        return None
    label = str(local_authority_label).lower().strip()
    for city, authorities in ALL_CLIENT_AUTHORITIES.items():
        for auth in authorities:
            if auth in label or label in auth:
                return city
    # Broader region heuristics
    if "london" in label:
        return "london"
    if "manchester" in label or "greater manchester" in label:
        return "manchester"
    if "birmingham" in label or "west midlands" in label:
        return "birmingham"
    return None


def _is_excluded_property(property_type: str | float | None) -> bool:
    if property_type is None or (isinstance(property_type, float) and np.isnan(property_type)):
        return False
    pt = str(property_type).lower()
    return any(kw in pt for kw in EXCLUDED_PROPERTY_TYPE_KEYWORDS)


def _available_cols(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    return [c for c in wanted if c in header.columns]


class EPCDataLoader:
    """Load and filter UK EPC certificates for high-rise modelling."""

    def __init__(
        self,
        domestic_dir: str | Path | None = None,
        nondomestic_dir: str | Path | None = None,
        chunksize: int = 200_000,
    ) -> None:
        root = _project_root()
        self.domestic_dir = Path(domestic_dir or root / RAW_DOMESTIC_DIR)
        self.nondomestic_dir = Path(nondomestic_dir or root / RAW_NONDOMESTIC_DIR)
        self.chunksize = chunksize
        self.stats: dict = {}

    def _certificate_files(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        files = sorted(directory.glob("certificates-*.csv"))
        return files

    def load_domestic_filtered(self, years: Iterable[int] | None = None) -> pd.DataFrame:
        """Chunk-load domestic certificates; keep mid/high-rise flats in 3 cities."""
        files = self._certificate_files(self.domestic_dir)
        if years is not None:
            year_set = {str(y) for y in years}
            files = [f for f in files if any(y in f.name for y in year_set)]

        frames: list[pd.DataFrame] = []
        raw_rows = 0
        kept_rows = 0

        for path in files:
            usecols = _available_cols(path, DOMESTIC_USECOLS)
            for chunk in pd.read_csv(
                path,
                usecols=usecols,
                chunksize=self.chunksize,
                low_memory=False,
                dtype=str,
            ):
                raw_rows += len(chunk)
                filtered = self._filter_domestic_chunk(chunk)
                kept_rows += len(filtered)
                if not filtered.empty:
                    frames.append(filtered)

        self.stats["domestic_raw"] = raw_rows
        self.stats["domestic_filtered"] = kept_rows
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _filter_domestic_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()
        df["property_type"] = df.get("property_type", "").astype(str)
        mask_flat = df["property_type"].str.lower().eq("flat")

        storey = pd.to_numeric(df.get("flat_storey_count"), errors="coerce")
        mask_storey = storey >= DOMESTIC_MIN_STOREY_COUNT

        df["source_city"] = df.get("local_authority_label", pd.Series(dtype=str)).map(assign_city)
        mask_city = df["source_city"].notna()

        target = pd.to_numeric(df.get(DOMESTIC_TARGET), errors="coerce")
        mask_target = target.notna() & (target > 0) & (target < DOMESTIC_TARGET_MAX)

        area = pd.to_numeric(df.get("total_floor_area"), errors="coerce")
        mask_area = area.notna() & (area > 0)

        keep = mask_flat & mask_storey & mask_city & mask_target & mask_area
        out = df.loc[keep].copy()
        if out.empty:
            return out

        out["building_typology"] = "residential"
        out["floor_area"] = pd.to_numeric(out["total_floor_area"], errors="coerce")
        out["storey_count"] = pd.to_numeric(out["flat_storey_count"], errors="coerce")
        out["energy_consumption"] = pd.to_numeric(out[DOMESTIC_TARGET], errors="coerce")
        out["co2_per_area"] = pd.to_numeric(out.get("co2_emiss_curr_per_floor_area"), errors="coerce")
        out["asset_or_env_score"] = pd.to_numeric(out.get("environment_impact_current"), errors="coerce")
        out["main_fuel"] = out.get("main_fuel")
        out["heating_type"] = out.get("mainheat_description")
        out["wall_efficiency"] = out.get("walls_energy_eff")
        out["roof_efficiency"] = out.get("roof_energy_eff")
        out["glazing_type"] = out.get("glazed_type")
        out["has_aircon"] = "unknown"
        out["epc_band"] = out.get("current_energy_rating")
        return out

    def load_nondomestic_filtered(self, years: Iterable[int] | None = None) -> pd.DataFrame:
        """Chunk-load non-domestic certificates; keep large commercial buildings in 3 cities."""
        files = self._certificate_files(self.nondomestic_dir)
        if years is not None:
            year_set = {str(y) for y in years}
            files = [f for f in files if any(y in f.name for y in year_set)]

        frames: list[pd.DataFrame] = []
        raw_rows = 0
        kept_rows = 0

        for path in files:
            usecols = _available_cols(path, NONDOMESTIC_USECOLS)
            for chunk in pd.read_csv(
                path,
                usecols=usecols,
                chunksize=self.chunksize,
                low_memory=False,
                dtype=str,
            ):
                raw_rows += len(chunk)
                filtered = self._filter_nondomestic_chunk(chunk)
                kept_rows += len(filtered)
                if not filtered.empty:
                    frames.append(filtered)

        self.stats["nondomestic_raw"] = raw_rows
        self.stats["nondomestic_filtered"] = kept_rows
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _filter_nondomestic_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()
        area = pd.to_numeric(df.get("floor_area"), errors="coerce")
        mask_area = area >= NONDOMESTIC_MIN_FLOOR_AREA

        pt = df.get("property_type", pd.Series(dtype=str))
        mask_exclude = ~pt.map(_is_excluded_property)

        df["source_city"] = df.get("local_authority_label", pd.Series(dtype=str)).map(assign_city)
        mask_city = df["source_city"].notna()

        target = pd.to_numeric(df.get(NONDOMESTIC_TARGET), errors="coerce")
        mask_target = target.notna() & (target > 0) & (target < NONDOMESTIC_TARGET_MAX)

        keep = mask_area & mask_exclude & mask_city & mask_target
        out = df.loc[keep].copy()
        if out.empty:
            return out

        out["building_typology"] = "commercial"
        out["floor_area"] = pd.to_numeric(out["floor_area"], errors="coerce")
        out["storey_count"] = pd.to_numeric(out.get("building_level"), errors="coerce")
        out["energy_consumption"] = pd.to_numeric(out[NONDOMESTIC_TARGET], errors="coerce")
        out["co2_per_area"] = pd.to_numeric(out.get("building_emissions"), errors="coerce")
        out["asset_or_env_score"] = pd.to_numeric(out.get("asset_rating"), errors="coerce")
        out["main_fuel"] = out.get("main_heating_fuel")
        out["heating_type"] = out.get("building_environment")
        out["wall_efficiency"] = np.nan
        out["roof_efficiency"] = np.nan
        out["glazing_type"] = np.nan
        aircon = out.get("aircon_present", pd.Series(dtype=str)).astype(str).str.lower()
        out["has_aircon"] = aircon.map(lambda x: "yes" if x in {"y", "yes", "true", "1"} else ("no" if x in {"n", "no", "false", "0"} else "unknown"))
        out["epc_band"] = out.get("asset_rating_band")
        out["total_floor_area"] = out["floor_area"]
        out["flat_storey_count"] = out["storey_count"]
        return out

    def standardise(self, df_dom: pd.DataFrame, df_nondom: pd.DataFrame) -> pd.DataFrame:
        """Unify domestic + non-domestic into a single modelling frame."""
        cols = [
            "certificate_number",
            "postcode",
            "local_authority_label",
            "source_city",
            "building_typology",
            "property_type",
            "floor_area",
            "storey_count",
            "energy_consumption",
            "co2_per_area",
            "asset_or_env_score",
            "main_fuel",
            "heating_type",
            "wall_efficiency",
            "roof_efficiency",
            "glazing_type",
            "has_aircon",
            "epc_band",
            "inspection_date",
            "uprn",
        ]
        frames = []
        for df in (df_dom, df_nondom):
            if df is None or df.empty:
                continue
            present = [c for c in cols if c in df.columns]
            frames.append(df[present].copy())
        if not frames:
            return pd.DataFrame(columns=cols)
        combined = pd.concat(frames, ignore_index=True)
        for c in cols:
            if c not in combined.columns:
                combined[c] = np.nan
        return combined[cols]

    def load_and_prepare(
        self,
        years: Iterable[int] | None = None,
        use_recent_years_only: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline.

        By default uses 2018–2026 to keep runtime manageable while covering
        recent high-rise stock. Pass years=None and use_recent_years_only=False
        for all certificate years.
        """
        if years is None and use_recent_years_only:
            years = list(range(2018, 2027))

        print("Loading domestic certificates (chunked)...")
        df_dom = self.load_domestic_filtered(years=years)
        print(f"  Domestic filtered: {len(df_dom):,} (from {self.stats.get('domestic_raw', 0):,} raw rows scanned)")

        print("Loading non-domestic certificates (chunked)...")
        df_nondom = self.load_nondomestic_filtered(years=years)
        print(f"  Non-domestic filtered: {len(df_nondom):,} (from {self.stats.get('nondomestic_raw', 0):,} raw rows scanned)")

        combined = self.standardise(df_dom, df_nondom)
        print(f"  Combined: {len(combined):,}")
        return combined, df_dom, df_nondom

    def generate_synthetic(self, n: int = 10_000) -> pd.DataFrame:
        """Fallback synthetic EPC-like data for pipeline testing."""
        rng = np.random.default_rng(42)
        cities = list(FEDERATED_CLIENTS.keys())
        typology = rng.choice(["residential", "commercial"], size=n, p=[0.5, 0.5])
        floor_area = np.where(
            typology == "residential",
            rng.lognormal(mean=5.0, sigma=0.4, size=n),
            rng.lognormal(mean=9.0, sigma=0.5, size=n),
        )
        energy = np.where(
            typology == "residential",
            rng.normal(180, 80, size=n),
            rng.normal(320, 150, size=n),
        )
        energy = np.clip(energy, 20, 1500)
        df = pd.DataFrame(
            {
                "certificate_number": [f"SYN-{i:06d}" for i in range(n)],
                "postcode": rng.choice(["E1 6AN", "M1 1AE", "B1 1AA"], size=n),
                "local_authority_label": rng.choice(
                    ["Tower Hamlets", "Manchester", "Birmingham"], size=n
                ),
                "source_city": rng.choice(cities, size=n),
                "building_typology": typology,
                "property_type": np.where(typology == "residential", "Flat", "Offices and Workshop Businesses"),
                "floor_area": floor_area,
                "storey_count": np.where(typology == "residential", rng.integers(5, 12, size=n), rng.integers(3, 20, size=n)),
                "energy_consumption": energy,
                "co2_per_area": rng.normal(40, 15, size=n),
                "asset_or_env_score": rng.normal(70, 20, size=n),
                "main_fuel": rng.choice(["mains gas", "electricity", "oil"], size=n, p=[0.7, 0.2, 0.1]),
                "heating_type": rng.choice(["boiler", "heat pump", "electric"], size=n),
                "wall_efficiency": rng.choice(["Very Good", "Good", "Average", "Poor"], size=n),
                "roof_efficiency": rng.choice(["Very Good", "Good", "Average", "Poor"], size=n),
                "glazing_type": rng.choice(["double", "single", "triple"], size=n),
                "has_aircon": rng.choice(["yes", "no", "unknown"], size=n),
                "epc_band": rng.choice(list("ABCDEFG"), size=n),
                "inspection_date": "2024-01-01",
                "uprn": [str(1000000000 + i) for i in range(n)],
                "is_synthetic": True,
            }
        )
        return df

    def verification_report(self, combined: pd.DataFrame) -> str:
        lines = [
            "=" * 54,
            " EPC DATA LOADING REPORT",
            "=" * 54,
            f" Domestic raw scanned:      {self.stats.get('domestic_raw', 0):,}",
            f" Domestic filtered:         {self.stats.get('domestic_filtered', 0):,}",
            f" Non-domestic raw scanned:  {self.stats.get('nondomestic_raw', 0):,}",
            f" Non-domestic filtered:     {self.stats.get('nondomestic_filtered', 0):,}",
            f" Combined records:          {len(combined):,}",
        ]
        if combined.empty:
            lines.append(" WARNING: empty combined dataset")
            lines.append("=" * 54)
            return "\n".join(lines)

        for typ, n in combined["building_typology"].value_counts().items():
            pct = 100 * n / len(combined)
            lines.append(f"   {typ}: {n:,} ({pct:.1f}%)")

        tgt = combined["energy_consumption"]
        lines.extend(
            [
                f" Target mean:               {tgt.mean():.1f}",
                f" Target std:                {tgt.std():.1f}",
                f" Target min/max:            {tgt.min():.0f} / {tgt.max():.0f}",
                " Federated partitions:",
            ]
        )
        means = []
        for city in FEDERATED_CLIENTS:
            n = int((combined["source_city"] == city).sum())
            m = combined.loc[combined["source_city"] == city, "energy_consumption"].mean()
            means.append(m)
            lines.append(f"   {city}: {n:,} records (mean energy {m:.1f})")
        if means and min(means) > 0:
            ratio = max(means) / min(means)
            lines.append(f" Non-IID ratio:             {ratio:.2f} (>1.2 = heterogeneous)")
        lines.append("=" * 54)
        return "\n".join(lines)
