# Assignment 1 — Presentation Pack (PP-XAI)

Student ID **202440724** · MSc Applied Artificial Intelligence · University of Hull  

This folder is the deliverable pack for **Assignment 1: Presentation and Video (20%)**.

## Contents

| File | Purpose |
|------|---------|
| [`202440724_PPXAI_Presentation.pptx`](202440724_PPXAI_Presentation.pptx) | Live deck (14 slides, widescreen 16:9) |
| [`speaker_notes_15min.md`](speaker_notes_15min.md) | Timed script for the ≤15 min live presentation |
| [`video_script_10min.md`](video_script_10min.md) | Timed script for the ≤10 min video (presenter visible) |
| [`assets/figures/`](assets/figures/) | Curated result / architecture figures |
| [`assets/diagrams/`](assets/diagrams/) | New slide diagrams (options, triangle, headlines) |
| [`scripts/generate_diagrams.py`](scripts/generate_diagrams.py) | Regenerate diagrams |
| [`scripts/build_pptx.py`](scripts/build_pptx.py) | Rebuild the PowerPoint |

## Brief alignment

Mandatory sections covered in the deck:

1. **Introduction** — problem / context (slides 1–3)  
2. **Options analysis** — considered vs chosen (slide 4 + diagram)  
3. **Development and results** — pipeline, FL, metrics, XAI (slides 5–11)  
4. **Discussion and conclusions** — critical analysis + future work (slides 13–14)  
5. **Technical demonstration** — Streamlit cue (slide 12)

Rubric targets: **Presentation content 10%** + **Technical demonstration 10%**.

## Anonymity

Canvas guidance: identify work by **student ID only**. The PPTX filename and slide footers use `202440724` (no personal name).

## Live demo (Technical Demonstration)

On Apple Silicon, avoid `source scripts/env.sh` from a conda `(base)` / Rosetta shell — it can force x86_64 Python against arm64 packages. Use the wrapper instead:

```bash
cd /path/to/Building-Energy-Prediction
conda deactivate          # if prompt shows (base)
/bin/bash scripts/run_streamlit.sh
```

App URL: http://localhost:8501

One-liner alternative:

```bash
arch -arm64 /bin/zsh -c 'source scripts/env.sh && streamlit run src/webapp/app.py'
```

Prepare 2–3 different building inputs so the demo looks interactive / non-scripted.

## Timing & overlength

| Deliverable | Limit |
|-------------|-------|
| Live presentation | Max **15** minutes (+ Q&A) |
| Video | Strongly **≤10** minutes |

Penalties (from brief): ≤10% over = none; 10–20% over = 10% of marks; >50% over = unmarked.

## Rebuild assets

```bash
source venv/bin/activate
pip install python-pptx   # if needed
python docs/presentation/scripts/generate_diagrams.py
python docs/presentation/scripts/build_pptx.py
```

## Key numbers (do not invent others)

- Filtered corpus: **5,663** records (≈99.4% commercial)  
- Best central GB: RMSE **113.04**, R² **0.559**  
- FedAvg MLP (8 rounds): RMSE **118.21**, R² **0.518**  
- SHAP Spearman ρ (GB vs Fed): **0.956**  
- Cross-client top-5 Jaccard: **0.78**
