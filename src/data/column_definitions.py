"""Column definitions and constants for UK EPC high-rise modelling.

Matches bulk CSV schema from get-energy-performance-data.communities.gov.uk
(underscore column names, not hyphenated API names).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DOMESTIC_DIR = "raw-data/domestic-csv"
RAW_NONDOMESTIC_DIR = "raw-data/non-domestic-csv"
PROCESSED_DIR = "data/processed"
FEDERATED_DIR = "data/federated"
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
DOMESTIC_TARGET = "energy_consumption_current"
NONDOMESTIC_TARGET = "primary_energy_value"
UNIFIED_TARGET = "energy_consumption"

# ---------------------------------------------------------------------------
# High-rise filters (revised for bulk export reality)
# ---------------------------------------------------------------------------
DOMESTIC_MIN_STOREY_COUNT = 5  # docs said >=10 but bulk data max is ~8
NONDOMESTIC_MIN_FLOOR_AREA = 5000.0
DOMESTIC_TARGET_MAX = 1000.0
NONDOMESTIC_TARGET_MAX = 2000.0

EXCLUDED_PROPERTY_TYPE_KEYWORDS = [
    "school",
    "hospital",
    "sports",
    "worship",
    "religious",
    "further education",
    "prison",
    "emergency",
    "swimming",
    "community centre",
    "crown and parliament",
    "primary health",
    "education",
    "care home",
]

# ---------------------------------------------------------------------------
# Federated clients — local authority labels
# ---------------------------------------------------------------------------
FEDERATED_CLIENTS: dict[str, list[str]] = {
    "london": [
        "City of London",
        "Westminster",
        "Camden",
        "Islington",
        "Hackney",
        "Tower Hamlets",
        "Southwark",
        "Lambeth",
        "Wandsworth",
        "Hammersmith and Fulham",
        "Kensington and Chelsea",
        "Greenwich",
        "Lewisham",
        "Newham",
        "Waltham Forest",
        "Haringey",
        "Barnet",
        "Brent",
        "Ealing",
        "Hounslow",
        "Richmond upon Thames",
        "Kingston upon Thames",
        "Merton",
        "Sutton",
        "Croydon",
        "Bromley",
        "Bexley",
        "Havering",
        "Barking and Dagenham",
        "Redbridge",
        "Enfield",
        "Harrow",
        "Hillingdon",
    ],
    "manchester": [
        "Manchester",
        "Salford",
        "Trafford",
        "Stockport",
        "Tameside",
        "Oldham",
        "Rochdale",
        "Bolton",
        "Bury",
        "Wigan",
    ],
    "birmingham": [
        "Birmingham",
        "Wolverhampton",
        "Coventry",
        "Solihull",
        "Sandwell",
        "Walsall",
        "Dudley",
    ],
}

# Flatten for quick membership tests (substring match on label)
ALL_CLIENT_AUTHORITIES = {
    city: [a.lower() for a in authorities]
    for city, authorities in FEDERATED_CLIENTS.items()
}

# ---------------------------------------------------------------------------
# Domestic columns to read (chunked load)
# ---------------------------------------------------------------------------
DOMESTIC_USECOLS = [
    "certificate_number",
    "postcode",
    "local_authority",
    "local_authority_label",
    "property_type",
    "built_form",
    "inspection_date",
    "current_energy_rating",
    "energy_consumption_current",
    "co2_emissions_current",
    "co2_emiss_curr_per_floor_area",
    "total_floor_area",
    "flat_storey_count",
    "floor_level",
    "construction_age_band",
    "main_fuel",
    "walls_description",
    "walls_energy_eff",
    "roof_description",
    "roof_energy_eff",
    "windows_description",
    "glazed_type",
    "multi_glaze_proportion",
    "mainheat_description",
    "mainheat_energy_eff",
    "hot_water_energy_eff",
    "lighting_energy_eff",
    "number_habitable_rooms",
    "number_heated_rooms",
    "mains_gas_flag",
    "floor_height",
    "environment_impact_current",
    "uprn",
    "region",
]

# ---------------------------------------------------------------------------
# Non-domestic columns to read
# ---------------------------------------------------------------------------
NONDOMESTIC_USECOLS = [
    "certificate_number",
    "postcode",
    "local_authority",
    "local_authority_label",
    "property_type",
    "inspection_date",
    "asset_rating",
    "asset_rating_band",
    "floor_area",
    "primary_energy_value",
    "building_emissions",
    "standard_emissions",
    "target_emissions",
    "main_heating_fuel",
    "building_environment",
    "aircon_present",
    "aircon_kw_rating",
    "renewable_sources",
    "special_energy_uses",
    "building_level",
    "uprn",
]

# ---------------------------------------------------------------------------
# Unified modelling features
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "floor_area",
    "storey_count",
]

CATEGORICAL_FEATURES = [
    "building_typology",
    "property_type",
    "main_fuel",
    "heating_type",
    "wall_efficiency",
    "roof_efficiency",
    "glazing_type",
    "has_aircon",
    "source_city",
]

FINAL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
