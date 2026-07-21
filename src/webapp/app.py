"""Streamlit prototype — EPC energy prediction with SHAP/LIME explanations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.column_definitions import (  # noqa: E402
    EPC_BANDS_KWH,
    FINAL_FEATURE_COLUMNS,
    energy_to_epc_band,
)


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
        cands = [
            p
            for p in models_dir.glob("*.joblib")
            if "federated" not in p.name and "parameters" not in p.name
        ]
        model_path = cands[0]
        best = model_path.stem
    model = joblib.load(model_path)
    alt = list((ROOT / "results" / "figures").glob("shap_importance_central_*.csv"))
    importance = pd.read_csv(alt[0]) if alt else pd.DataFrame()

    lime_path = ROOT / "results" / "tables" / "lime_weights.json"
    lime_data = json.loads(lime_path.read_text()) if lime_path.exists() else {"explanations": []}
    return pre, model, best, feature_names, metrics, importance, lime_data


def _short_feature(name: str, max_len: int = 42) -> str:
    text = (
        str(name)
        .replace("cat__", "")
        .replace("num__", "")
        .replace(" <= 0.00", " =0")
        .replace(" > 0.00", " >0")
    )
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def render_lime_bar(weights: list[dict], title: str) -> None:
    """Horizontal bar chart of LIME feature weights (positive vs negative)."""
    if not weights:
        st.warning("No LIME weights available for this selection.")
        return
    df = pd.DataFrame(weights).copy()
    df["label"] = df["feature"].map(_short_feature)
    df = df.sort_values("weight")
    colours = ["#2B6CB0" if w >= 0 else "#C53030" for w in df["weight"]]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(df))))
    ax.barh(df["label"], df["weight"], color=colours)
    ax.axvline(0, color="#4A5568", lw=0.8)
    ax.set_xlabel("LIME weight (contribution to prediction)")
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="PP-XAI Building Energy", layout="wide")
    st.title("PP-XAI — High-Rise Building Energy Prediction")
    st.caption("Privacy-preserving explainable AI prototype (MSc dissertation demo)")

    try:
        pre, model, best, feature_names, metrics, importance, lime_data = load_artefacts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load models. Run the training pipeline first.\n\n{exc}")
        st.stop()

    with st.expander("EU AI Act transparency note (Article 13)", expanded=False):
        st.markdown(
            """
This prototype is a **research demonstrator**, not a certified EPC tool or an automated
decision system for tenancy, lending, or enforcement.

In line with **EU AI Act Article 13** transparency expectations for AI systems that may
inform human decisions, the interface:

- discloses the model type used for prediction;
- shows **global SHAP** feature importance;
- shows **local LIME** attributions for example instances;
- maps predicted intensity to an **illustrative EPC band (A–G)**.

Human oversight is required before any operational use. Predictions are correlational
estimates from open EPC features and must not replace statutory assessment procedures.
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Building inputs")
        typology = st.selectbox("Building typology", ["commercial", "residential"])
        city = st.selectbox("City client", ["london", "manchester", "birmingham"])
        floor_area = st.number_input("Floor area (m²)", min_value=50.0, max_value=100000.0, value=6000.0)
        storey_count = st.number_input("Storey count", min_value=1.0, max_value=80.0, value=10.0)
        main_fuel = st.selectbox(
            "Main fuel",
            ["Natural Gas", "Grid Electricity", "Oil", "mains gas", "electricity"],
        )
        has_aircon = st.selectbox("Air conditioning", ["yes", "no", "unknown"])
        property_type = st.text_input(
            "Property type",
            "Offices and Workshop Businesses" if typology == "commercial" else "Flat",
        )

    with col2:
        st.subheader("Model")
        st.write(f"Active model: **{best}**")
        if metrics:
            st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
        shap_fig = list((ROOT / "results" / "figures").glob(f"shap_importance_central_{best}.png"))
        if not shap_fig:
            shap_fig = list((ROOT / "results" / "figures").glob("shap_importance_central_*.png"))
        if shap_fig:
            st.image(str(shap_fig[0]), caption="Global SHAP importance (centralised)")

    if st.button("Predict energy consumption", type="primary"):
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
        band = energy_to_epc_band(pred)

        st.success(
            f"Predicted energy intensity: **{pred:.1f}** kWh/m²/year  ·  "
            f"Illustrative EPC band: **{band}**"
        )
        band_rows = [
            {"Band": b, "Min kWh/m²/year": lo, "Max kWh/m²/year": ("∞" if hi == float("inf") else hi)}
            for b, (lo, hi) in EPC_BANDS_KWH.items()
        ]
        with st.expander("EPC band thresholds used in this demo"):
            st.caption(
                "Domestic-style illustrative bands for communication. Official non-domestic "
                "asset ratings use a different scale; do not treat this as a statutory EPC."
            )
            st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)

        if not importance.empty:
            st.subheader("Top features (global SHAP)")
            st.dataframe(importance.head(10), use_container_width=True)

        # LIME local explanation panel
        st.subheader("Local LIME explanation")
        explanations = lime_data.get("explanations") or []
        if not explanations:
            st.warning("No `results/tables/lime_weights.json` found. Run `scripts/run_xai.py` then `scripts/aggregate_lime_weights.py`.")
        else:
            labels = [
                f"{e.get('model', 'model')} · instance {e.get('instance')}"
                for e in explanations
            ]
            choice = st.selectbox("Saved LIME instance", labels, index=0)
            selected = explanations[labels.index(choice)]
            render_lime_bar(
                selected.get("weights", []),
                title=f"LIME weights — {selected.get('model')} (instance {selected.get('instance')})",
            )
            # Also show matching on-disk plot if present
            fig_name = selected.get("file", "").replace(".csv", ".png")
            fig_path = ROOT / "results" / "figures" / fig_name
            if fig_path.exists():
                st.image(str(fig_path), caption=f"Saved LIME plot: {fig_name}")

        st.info(
            "This system provides explanations in support of EU AI Act Article 13 "
            "transparency expectations (research prototype — human oversight required)."
        )

    st.divider()
    st.markdown(
        "Data: UK EPC Open Government Licence v3.0. "
        "Federated clients simulate London / Manchester / Birmingham data holders. "
        "PP-XAI MSc demonstrator — not an official EPC assessment."
    )


if __name__ == "__main__":
    main()
