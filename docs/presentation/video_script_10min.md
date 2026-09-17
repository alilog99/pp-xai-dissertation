# Video Script — ≤10 Minutes

**Deck:** [`202440724_PPXAI_Presentation.pptx`](202440724_PPXAI_Presentation.pptx) (14 slides)  
**Audience:** Intelligent viewers **not** familiar with federated learning or XAI.  
**Format:** Presenter clearly **visible** (picture-in-picture or face camera) throughout.  
**Spoken intro:** Use your real name (as in the Week 1 sample video).  
**Submission files / end card:** student ID **202440724** only (Canvas anonymity for filenames).  
**Goal:** Conversational demo video — who you are, why the project, what you built, **live software walkthrough**, short results, limits.  
**Style tip (Week 1 sample):** Hook early · speak naturally · visuals over code · demonstrate running software · avoid jargon · tell a story.

**Recommended length:** 8–9 minutes (safe margin under 10).

---

## Slide map for the video (do not use all 14)

| Time | On screen | Deck slide | Action |
|-----:|-----------|------------|--------|
| 0:00–0:45 | PowerPoint | **1 — Title** | Hello + name + project |
| 0:45–1:45 | PowerPoint | **2 — The Problem** | Why EPC + privacy; point at triangle |
| 1:45–2:35 | PowerPoint | **6 — PP-XAI Pipeline** | Brief left→right walk |
| ~2:35–2:40 | PowerPoint (optional) | **12 — Live Demonstration** | 5s cue: “switching to the app” |
| 2:40–7:00 | **Browser** | *(leave PPT)* | Streamlit full screen |
| 7:00–8:20 | PowerPoint | **9 — Federated Results** | Three headline numbers |
| 8:20–9:15 | PowerPoint | **13 — Discussion & Limitations** | Caveats + further work |
| 9:15–9:50 | PowerPoint | **14 — Conclusions** | Thank you + student ID |

### Skip in the video (use only in the live 15-min talk)

| Skip | Slide | Why |
|------|-------|-----|
| — | **3** Aim & RQs | Too dense for video |
| — | **4** Options analysis | Live presentation requirement |
| — | **5** Data & high-rise | Mentioned verbally on slide 6 |
| — | **7** FL architecture | Covered lightly in speech |
| — | **8** Centralised results | Folded into slide 9 headlines |
| — | **10** Critical stats | One sentence on slide 9 is enough |
| — | **11** XAI charts | Shown live in Streamlit instead |

Live talk script for **all** slides: [`speaker_notes_15min.md`](speaker_notes_15min.md).

### How to operate while recording

1. Open the PPTX on **Slide 1**. Start Streamlit first (`scripts/run_streamlit.sh` → http://localhost:8501).  
2. Advance only: **1 → 2 → 6 → (optional 12)**.  
3. Alt-Tab / Mission Control to the browser for the demo.  
4. After demo, jump to **Slide 9** (not next slide — skip 7–8–10–11–12 if needed).  
5. Then **13 → 14**.  
6. Keep your face visible the whole time (PiP over slides and app).

---

## Shot plan (summary)

| Time | Segment | On screen | Camera |
|-----:|---------|-----------|--------|
| 0:00–0:45 | Hello + project title | **Slide 1** | Face visible |
| 0:45–1:45 | Why this project | **Slide 2** | Face + slides |
| 1:45–2:40 | What I built (brief) | **Slide 6** (+ optional **12**) | Face + slides |
| 2:40–7:00 | **Software demo (main act)** | Streamlit | Face PiP |
| 7:00–8:20 | Results story | **Slide 9** | Face + slides |
| 8:20–9:15 | Limits / further work | **Slide 13** | Face + slides |
| 9:15–9:50 | Close | **Slide 14** | Face visible |

**Style note:** most of the video should *show the running product* — not a long lecture with a short demo clip.

---

## Full spoken script (with slide cues)

### [0:00–0:45] Hello + introduce the project

**On screen: Slide 1 — Title**  
*(Privacy-Preserving Explainable AI for Building Energy Performance Prediction · Student ID 202440724)*

“Hello. My name is **Syed Ali Raza**, and I’m going to talk to you today about my final year project, which is **PP-XAI** — Privacy-Preserving Explainable AI for Building Energy Performance Prediction.

In short: it’s a system that predicts how energy-intensive large buildings are, explains *why* it made that prediction, and does the training without pooling everyone’s raw data into one place.”

*(Advance to Slide 2.)*

---

### [0:45–1:45] Why you need this (motivation)

**On screen: Slide 2 — The Problem**  
*(Bullets on the left · PP-XAI Design Triangle on the right — Privacy / Accuracy / Explainability)*

“First of all, let’s discuss why this matters — it may not be obvious if you’re not in the energy or AI space.

In the UK we have **Energy Performance Certificates** — open records that describe how buildings use energy. They’re very useful for machine learning. But for sensitive assets — high-rise homes and large commercial sites — councils and landlords often **cannot** share microdata with each other.

A brand-new centralised model assumes you can put everything in one spreadsheet. In practice, you often can’t. So the obvious approach is to train models **where the data already lives**, share only **model updates**, and still give people an explanation they can trust.

*(Gesture at the triangle.)* That’s the design idea on the right: **privacy**, **accuracy**, and **explainability** together — not accuracy alone.”

*(Advance to Slide 6 — skip 3, 4, 5.)*

---

### [1:45–2:40] What was built (brief)

**On screen: Slide 6 — PP-XAI Pipeline**  
*(End-to-end strip: ingest → filter → preprocess → centralised + federated → SHAP/LIME → Streamlit)*

“Now, the project itself uses UK EPC open data for London, Manchester, and Birmingham — about **five and a half thousand** buildings after honest high-rise filters.

*(Point left to right on the pipeline.)* I’ve built centralised baselines for comparison, and a neural network trained with **federated averaging** across three city clients, so raw CSV rows never leave their partition. On top of that I added **SHAP** and **LIME** for explanations, and wrapped everything in a **Streamlit** web app so it looks and feels usable — not just a notebook.

Okay. So that’s enough of the background. Let’s move on to the demonstration of the software itself.”

**Optional 5 seconds — Slide 12 — Live Demonstration**  
*(Only if you want a visual cue before Alt-Tabbing.)*  
“Here’s the live demonstrator — switching to the running app now.”

*(Leave PowerPoint. Switch to browser. Face stays visible as PiP.)*

---

### [2:40–7:00] LIVE DEMO — click-by-click (main act)

**On screen: Streamlit at http://localhost:8501**  
*(Not a PowerPoint slide)*

**Before record:** app already open · browser zoom ~110% · notifications off.

“So, I’ve got a copy of the PP-XAI demonstrator running here.”

**1. Orient the UI (~30s)**  
“You’ll see the user interface is split in a familiar way. On the left we have the building inputs. On the right we have the active model and a live view of **global SHAP** importance — which features matter most across the stock.

Further down we also have **local LIME** explanations for individual cases. Those two — the prediction and the explanation — are the most important characteristics of the system, so they’re represented clearly.”

**2. Building A — large London commercial (~60–75s)**  
“Let’s configure a realistic large commercial building.

I’ll leave typology as **commercial**.  
City client: **London**.  
Floor area: about **six thousand** square metres.  
Storeys: **ten**.  
Main fuel: **Natural Gas**.  
Air conditioning: **yes**.  
Property type: **Offices and Workshop Businesses**.

And if everything looks correct, we can trigger the prediction. I click **Predict energy consumption**.”

*(Pause for the metric to appear.)*

“There we are. The model returns a predicted energy intensity — here, about **[read the number aloud]** kilowatt-hours per square metre per year.

That’s the kind of figure a planner could use to prioritise which assets to inspect first — without needing every organisation’s raw EPC spreadsheet in one place.”

**3. Point to explanations (~45–60s)**  
“Under the prediction, we get transparency.

This **global SHAP** view shows which drivers matter most in general — things like storey count and property type often rank high.

And here is the **local LIME** panel — for a saved instance, the bars show which features push this particular prediction **up** or **down**.

That matters for trust: a stakeholder should see *why*, not only a single score. It also aligns with the spirit of transparency expectations such as EU AI Act Article 13.”

**4. Building B — change one thing and re-predict (~60–75s)**  
“We can adjust the scenario at any time. I’ll change the city to **Manchester**, switch main fuel to **Grid Electricity**, and maybe reduce floor area a little — so you can see it’s interactive, not a fixed screenshot.

Predict again.”

*(Pause. Read the new number.)*

“You can see the predicted intensity has changed — and the explanation context updates with the new inputs.

So the demonstrator isn’t just a static chart: a user can try different cities and fabric assumptions and immediately see the effect.”

**5. Optional Building C if time (~30s)**  
“We can start that over with another building type if we wish. One more quick check — residential typology, Birmingham, smaller floor area.

Predict… and again we get a number plus the explanation panels. Same workflow for a different building type.”

**6. Close the demo (~20–30s)**  
“So that’s essentially what the demonstrator does. It runs from inputs through to prediction and explanation, and you can re-run it as needed.

We haven’t currently hard-coded every typical error case into this prototype, but the scope is there to do so, and that’s discussed in the project report as further work.

Okay. So that was the demonstration.”

*(Return to PowerPoint → jump to **Slide 9**. Do not walk slides 7–8–10–11.)*

---

### [7:00–8:20] Results (three numbers only)

**On screen: Slide 9 — Federated Results (RQ1)**  
*(Convergence chart + headline result cards: GB R² 0.559 · Fed R² 0.518 · ρ 0.956)*

“Very briefly — three headline results from the experiments behind this app.

**First — accuracy.** Best centralised model: R-squared about **zero point five six**. Federated neural model: about **zero point five two** — nearly matching a neural net trained on pooled data.

**Second — practical gap.** Versus the best tree model the difference is statistically detectable, but the **effect size is tiny** and confidence intervals overlap. Under a privacy constraint, that is a small price.

**Third — explanations.** SHAP rankings correlate at about **zero point nine six** between federated and best centralised models. Keeping data local did **not** scramble what the model says matters.”

*(Advance to Slide 13 — skip 10–12.)*

---

### [8:20–9:15] Critical honesty / further work

**On screen: Slide 13 — Discussion & Limitations**

“Two important caveats — and then I’m done.

The filtered dataset is overwhelmingly **large commercial** buildings. True residential towers are hard to identify in bulk EPC storey fields — I document that rather than hide it.

And this federated setup is a **research simulator**. It preserves data residency, but it is not yet hardened with differential privacy or secure aggregation — that is future work.

I also removed features that effectively leaked the answer — early trials with emissions fields produced unrealistically high scores. Reporting the honest, lower scores was the right scientific choice.”

*(Advance to Slide 14.)*

---

### [9:15–9:50] Close

**On screen: Slide 14 — Conclusions, Contributions & Future Work**

“So, in short: privacy-preserving training can approach centralised neural accuracy, keep explanations stable, and ship as a working demonstrator — which you’ve just seen running.

That’s PP-XAI — that’s what the project does. Thank you for watching.

If you have any questions, well — that’s going to be rather tricky because this is a video. Student ID **202440724**.”

---

## Recording checklist

- [ ] Face clearly visible for the whole video  
- [ ] Open with **“Hello. My name is Syed Ali Raza…”** (sample style)  
- [ ] PPT path rehearsed: **1 → 2 → 6 → (12) → browser → 9 → 13 → 14**  
- [ ] Mic test; quiet room; 1080p if possible  
- [ ] Streamlit running locally before you press record (`scripts/run_streamlit.sh`)  
- [ ] Practise Building A → explain panels → Building B once before recording  
- [ ] Filename uses student ID only, e.g. `202440724_PPXAI_video.mp4`  
- [ ] Final length **under 10:00** (demo should be the longest segment)  
- [ ] No need for Hollywood polish — clarity beats effects  

## Editing tips

- Cut dead air **outside** the demo; keep natural clicking pauses **inside** the demo  
- Soft zoom when the prediction number updates and when LIME bars appear  
- Burn-in captions for the three headline numbers if useful  
- End card: project title + student ID (no name on the card)  
- If over 10:00, trim the pre-demo slides first — **do not** cut the demo short
