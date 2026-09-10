# Video Script — ≤10 Minutes

**Audience:** Intelligent viewers **not** familiar with federated learning or XAI.  
**Format:** Presenter clearly **visible** (picture-in-picture or face camera) throughout.  
**Anonymity:** Use **student ID 202440724** only in titles/filenames — not your name.  
**Goal:** Video version of the abstract — problem, what you built, demo, headline results, limitations.  
**Style tip (from brief):** Hook early · visuals over code · demonstrate running software · avoid jargon · tell a story.

**Recommended length:** 8–9 minutes (safe margin under 10).

---

## Shot plan

| Time | Segment | On screen | Camera |
|-----:|---------|-----------|--------|
| 0:00–0:40 | Hook + who/what | Title slide | Face visible |
| 0:40–1:40 | Problem story | Problem slide / triangle | Face + slides |
| 1:40–2:40 | What was built (brief) | Pipeline slide only | Face + slides |
| 2:40–7:00 | **Software demo (main act)** | Streamlit app full screen | Face PiP |
| 7:00–8:20 | Results story | Headline cards + 1 chart | Face + slides |
| 8:20–9:20 | Critical take + limits | Discussion bullets | Face + slides |
| 9:20–9:50 | Close + contributions | Final slide | Face visible |

**Style note (Week 1 sample):** most of the video should *show the running product*, with click-by-click narration — not a long lecture with a short demo clip.

---

## Full spoken script

### [0:00–0:40] Hook
“Imagine you manage energy certificates for towers and large buildings across several cities. Machine learning could help prioritise retrofit — but you are not allowed to upload everyone’s raw data into one central lake.  

This project — **PP-XAI** — asks: can we still get accurate predictions **and** clear explanations, while keeping the data local?”

### [0:40–1:40] Problem (non-specialist)
“In the UK, **Energy Performance Certificates** describe how buildings use energy. They are open data, and they are valuable for prediction models.  

The catch: for sensitive assets — high-rise homes and large commercial sites — organisations often cannot pool microdata. Standard AI training assumes everything sits in one spreadsheet.  

So I built a system that trains models **where the data already lives**, shares only **model updates**, and then explains **why** a prediction was made.”

### [1:40–2:40] What was developed (keep very light)
“Briefly — what I built. I filtered UK EPC records for London, Manchester, and Birmingham — about **five and a half thousand** buildings after honest high-rise rules.  

I trained centralised baselines for comparison, and a neural network with **federated averaging** across the three cities so raw rows never leave their partition.  

I then applied **SHAP** and **LIME** for explanations, and wrapped the model in a **Streamlit** web app. That’s what I’ll demonstrate now.”

*(Skip deep options talk in the video — save that for the live 15-minute presentation.)*

### [2:40–7:00] LIVE DEMO — click-by-click (main act)

**Before record:** Streamlit already open at http://localhost:8501 · browser zoom ~110% · notifications off.

“Okay — enough slides. Here’s the running software.”

*(Switch to full-screen browser. Face stays visible as PiP.)*

**1. Orient the UI (~30s)**  
“This is the PP-XAI demonstrator. On the left you set the building. On the right you see the active model and a global SHAP importance chart — which features matter most across the stock.  

Down the page we also have local **LIME** explanations for individual cases.”

**2. Building A — large London commercial (~60–75s)**  
“Let’s configure a realistic large commercial building.  

I’ll leave typology as **commercial**.  
City client: **London**.  
Floor area: about **six thousand** square metres.  
Storeys: **ten**.  
Main fuel: **Natural Gas**.  
Air conditioning: **yes**.  
Property type: **Offices and Workshop Businesses**.  

Now I click **Predict energy consumption**.”

*(Pause for the metric to appear.)*

“The model returns a predicted energy intensity — here, about **[read the number aloud]** kilowatt-hours per square metre per year.  

That’s the kind of figure a planner could use to prioritise which assets to inspect first — without needing every organisation’s raw EPC spreadsheet in one place.”

**3. Point to explanations (~45–60s)**  
“Under the prediction, we get transparency.  

This **global SHAP** view shows which drivers matter most in general — things like storey count and property type often rank high.  

And here is the **local LIME** panel — for a saved instance, the bars show which features push this particular prediction **up** or **down**.  

That matters for trust: a stakeholder should see *why*, not only a single score. It also aligns with the spirit of transparency expectations such as EU AI Act Article 13.”

**4. Building B — change one thing and re-predict (~60–75s)**  
“Now I’ll change the scenario so you can see it’s interactive — not a fixed screenshot.  

I’ll switch city to **Manchester**, change main fuel to **Grid Electricity**, and maybe reduce floor area a little.  

Predict again.”

*(Pause. Read the new number.)*

“You can see the predicted intensity has changed — and the explanation context updates with the new inputs.  

So the demonstrator isn’t just a static chart: a user can try different cities and fabric assumptions and immediately see the effect.”

**5. Optional Building C if time (~30s)**  
“One more quick check — residential typology, still in Birmingham, smaller floor area.  

Predict… and again we get a number plus the explanation panels. Same workflow for a different building type.”

**6. Close the demo (~20s)**  
“So that’s the product behaviour end-to-end: inputs, prediction, global importance, local explanation, and an interactive re-run.  

I haven’t hard-coded every edge case into this prototype — those limits are discussed as further work in the report. For now, the important point is: the pipeline runs, and a non-specialist can operate it.”

*(Return to slides.)*

### [7:00–8:20] Results (three numbers only)
“Three headline results from the experiments behind this app.  

**First — accuracy.** Best centralised model: R-squared about **zero point five six**. Federated neural model: about **zero point five two** — nearly matching a neural net trained on pooled data.  

**Second — practical gap.** Versus the best tree model the difference is statistically detectable, but the **effect size is tiny** and confidence intervals overlap. Under a privacy constraint, that is a small price.  

**Third — explanations.** SHAP rankings correlate at about **zero point nine six** between federated and best centralised models. Keeping data local did **not** scramble what the model says matters.”

Show `results_headline.png` once — not a table dump.

### [8:20–9:20] Critical honesty
“Two important caveats.  

The filtered dataset is overwhelmingly **large commercial** buildings. True residential towers are hard to identify in bulk EPC storey fields — I document that rather than hide it.  

And this federated setup is a **research simulator**. It preserves data residency, but it is not yet hardened with differential privacy or secure aggregation — that is future work.  

I also removed features that effectively leaked the answer — early trials with emissions fields produced unrealistically high scores. Reporting the honest, lower scores was the right scientific choice.”

### [9:20–9:50] Close
“In short: privacy-preserving training can approach centralised neural accuracy, keep explanations stable, and ship as a working demonstrator — which you’ve just seen running.  

That is PP-XAI. Thank you for watching — student ID **202440724**.”

---

## Recording checklist

- [ ] Face clearly visible for the whole video  
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
- End card: project title + student ID (no name)  
- If over 10:00, trim the pre-demo slides first — **do not** cut the demo short
