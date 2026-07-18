# Data licence and access — UK EPC

## Licence

UK Energy Performance Certificate (EPC) open data used in this project are provided under the **Open Government Licence v3.0 (OGL)**.

- OGL overview: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/  
- EPC data portal: https://get-energy-performance-data.communities.gov.uk/  

You must include attribution when using the data, for example:

> Contains public sector information licensed under the Open Government Licence v3.0.  
> Energy Performance of Buildings Data: England and Wales.

## What is in this GitHub repository

| Included | Not included (by design) |
|---|---|
| Code, scripts, dissertation drafts | Raw EPC bulk CSVs (`raw-data/`, tens of GB) |
| Aggregated metrics / figures under `results/` (where not gitignored) | `.env` secrets / API keys |
| Instructions to obtain data | University ethics policy PDF (link only — see [ETHICS.md](ETHICS.md)) |

Raw certificate files are **gitignored**. Replicate locally by downloading from the official portal and placing files under:

```text
raw-data/
├── domestic-csv/
└── non-domestic-csv/
```

See the root [README.md](../README.md) for filters and pipeline commands.

## Supplementary data

- **OpenStreetMap** high-rise geometries: ODbL (Open Data Commons Open Database License) — attribute OSM contributors.  
- **OS Open UPRN** (optional): follow Ordnance Survey open data licence terms if used.

## Responsible use

Do not attempt to re-identify individuals beyond what the open register already publishes. Do not use model outputs as a substitute for official EPC assessments.
