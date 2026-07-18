# Appendix A — Reproducibility and Environment Notes

## A.1 Repository layout

```
Building-Energy-Prediction/
├── raw-data/                 # EPC certificates (not in git)
├── data/processed/           # filtered CSVs, npy arrays, preprocessor
├── data/federated/           # client_*.csv and X_*/y_* arrays
├── data/raw/osm/             # OSM high-rise GeoJSON
├── src/data|models|federated|xai|evaluation|webapp
├── scripts/                  # run_all, load, preprocess, baselines, FL, XAI, eval
├── results/figures|tables|models
├── dissertation/
└── dev-docs/
```

## A.2 Data volumes

Domestic certificate CSVs total on the order of tens of gigabytes; non-domestic about two gigabytes. Recommendations files are larger still and unused. Filtering must be chunked; never `pd.read_csv` an entire domestic year file without usecols and chunksize.

## A.3 Command book

```bash
# Activate
cd /Users/ali/Development/UNIVERSITYOFHULL/FINAL-PROJECT/WKPS/Building-Energy-Prediction
unset DYLD_LIBRARY_PATH
arch -arm64 /bin/zsh -c 'source scripts/env.sh && /bin/bash scripts/run_all.sh'

# App
arch -arm64 /bin/zsh -c 'source scripts/env.sh && streamlit run src/webapp/app.py'
```

## A.4 Known pitfalls

1. `/usr/local/bin/bash` is x86 Homebrew — do not `arch -arm64 bash`.
2. Setting `DYLD_LIBRARY_PATH` to `.deps/libomp` breaks NumPy.
3. LightGBM may hang without `OMP_NUM_THREADS=1` / `n_jobs=1`.
4. Conda `(base)` + Rosetta can force x86 Python against arm64 wheels.

## A.5 OGL attribution example

Contains public sector information licensed under the Open Government Licence v3.0. Energy Performance of Buildings Data: England and Wales.

## A.6 Extended discussion of federated averaging

FedAvg initialises a global model \(w_0\). At round \(t\), each client \(k\) copies \(w_t\), runs local SGD for \(E\) epochs on its data \(D_k\), and returns \(w_{t+1}^{k}\). The server sets

\[ w_{t+1} = \sum_k \frac{n_k}{n} w_{t+1}^{k} \]

with \(n_k=|D_k|\) and \(n=\sum n_k\). In this project, \(E=4\), rounds \(=8\), and clients are London, Manchester, and Birmingham matrices after shared preprocessing. The weighting scheme automatically emphasises London’s larger sample.

## A.7 Extended discussion of SHAP stability

Let \(\phi^{c}_j\) and \(\phi^{f}_j\) be mean absolute SHAP values for feature \(j\) under centralised and federated models. Spearman’s \(\rho\) is the Pearson correlation of the ranks of \(\phi^{c}\) and \(\phi^{f}\). High \(\rho\) indicates that features important under pooled training remain important under federated training, supporting transferable explanation dashboards.

## A.8 Threat model (brief)

This MSc does not claim protection against membership inference on updates, model inversion, or malicious server attacks. The privacy claim is organisational data residency during training (raw rows stay on clients in the simulator), not cryptographic privacy.

## A.9 Mapping objectives to artefacts

| Objective | Artefact |
|-----------|----------|
| 1 Baselines | `results/tables/baseline_metrics.json`, model joblibs |
| 2 FL | `federated_metrics.json`, `federated_mlp.pt` |
| 3 XAI | `shap_importance_*.png/csv`, lime plots |
| 4 Comparison | `model_comparison.csv`, `statistical_tests.json`, `xai_stability.json` |
| 5 Prototype | `src/webapp/app.py` |

## A.10 Narrative for viva (2-minute version)

We predict high-rise building energy from UK EPCs. Bulk data do not support a naïve ten-storey domestic filter, so we revised filters and focused on large commercial buildings plus a small residential proxy. Central gradient boosting reaches R² about 0.56; federated MLP about 0.52. SHAP rankings agree at Spearman 0.96. That means we can keep data local and still explain the model coherently. The Streamlit app shows this live.
