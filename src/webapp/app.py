"""Streamlit prototype — EPC energy prediction with explanations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@st.cache_resource
def load_artefacts():
    processed = ROOT / "data" / "processed"
    models_dir = ROOT / "results" / "models"
    pre = joblib.load(processed / "preprocessor.joblib")
    feature_names = json.loads((processed / "feature_names.json").read_text())
    metrics = {}
    mp = ROOT / "results" / "tables" / "baseline_metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    best = min(metrics.items(), key=lambda kv: kv[1]["RMSE"])[0] if metrics else "random_forest"
    model_path = models_dir / f"{best}.joblib"
    if not model_path.exists():
        cands = [p for p in models_dir.glob("*.joblib") if "federated" not in p.name and "parameters" not in p.name]
        model_path = cands[0]
        best = model_path.stem
    model = joblib.load(model_path)
    importance_path = ROOT / "results" / "figures" / f"shap_importance_central_{best}.csv"
    # SHAP CSVs are saved under figures/ by run_xai — also check tables
    alt = list((ROOT / "results" / "figures").glob("shap_importance_central_*.csv"))
    importance = pd.read_csv(alt[0]) if alt else pd.DataFrame()
    return pre, model, best, feature_names, metrics, importance


def main() -> None:
    st.set_page_config(page_title="PP-XAI Building Energy", layout="wide")
    st.title("PP-XAI — High-Rise Building Energy Prediction")
    st.caption("Privacy-preserving explainable AI prototype (MSc dissertation demo)")

    try:
        pre, model, best, feature_names, metrics, importance = load_artefacts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load models. Run the training pipeline first.\n\n{exc}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Building inputs")
        typology = st.selectbox("Building typology", ["commercial", "residential"])
        city = st.selectbox("City client", ["london", "manchester", "birmingham"])
        floor_area = st.number_input("Floor area (m²)", min_value=50.0, max_value=100000.0, value=6000.0)
        storey_count = st.number_input("Storey count", min_value=1.0, max_value=80.0, value=10.0)
        main_fuel = st.selectbox("Main fuel", ["Natural Gas", "Grid Electricity", "Oil", "mains gas", "electricity"])
        has_aircon = st.selectbox("Air conditioning", ["yes", "no", "unknown"])
        property_type = st.text_input("Property type", "Offices and Workshop Businesses" if typology == "commercial" else "Flat")

    with col2:
        st.subheader("Model")
        st.write(f"Active model: **{best}**")
        if metrics:
            st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
        shap_fig = list((ROOT / "results" / "figures").glob(f"shap_importance_central_{best}.png"))
        if shap_fig:
            st.image(str(shap_fig[0]), caption="Global SHAP importance (centralised)")

    if st.button("Predict energy consumption", type="primary"):
        from src.data.column_definitions import FINAL_FEATURE_COLUMNS

        feat = pd.DataFrame(
            [
                {
                    "floor_area": floor_area,
                    "storey_count": storey_count,
                    "building_typology": typology,
                    "property_type": property_type,
                    "main_fuel": main_fuel,
                    "heating_type": "unknown",
                    "wall_efficiency": "Average",
                    "roof_efficiency": "Average",
                    "glazing_type": "double",
                    "has_aircon": has_aircon,
                    "source_city": city,
                }
            ]
        )
        for c in FINAL_FEATURE_COLUMNS:
            if c not in feat.columns:
                feat[c] = "unknown"
        Xt = pre.transform(feat[FINAL_FEATURE_COLUMNS])
        pred = float(model.predict(Xt)[0])
        st.success(f"Predicted energy intensity: **{pred:.1f}** (kWh/m²/year or primary energy units)")
        st.info(
            "Explanations: see SHAP global plot (right). Local LIME plots are in results/figures/ "
            "after running scripts/run_xai.py."
        )
        if not importance.empty:
            st.subheader("Top features (SHAP)")
            st.dataframe(importance.head(10), use_container_width=True)

    st.divider()
    st.markdown(
        "Data: UK EPC Open Government Licence v3.0. "
        "Federated clients simulate London / Manchester / Birmingham data holders."
    )


if __name__ == "__main__":
    main()
