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
| 0:00–0:45 | Hook + who/what | Title slide | Face visible |
| 0:45–2:00 | Problem story | Problem slide / triangle | Face + slides |
| 2:00–3:30 | What was built | Pipeline + options (brief) | Face + slides |
| 3:30–5:30 | **Software demo** | Streamlit app full screen | Face PiP |
| 5:30–7:30 | Results story | Headline cards + 1–2 charts | Face + slides |
| 7:30–9:00 | Critical take + limits | Discussion bullets | Face + slides |
| 9:00–9:45 | Close + contributions | Final slide | Face visible |

---

## Full spoken script

### [0:00–0:45] Hook
“Imagine you manage energy certificates for towers and large buildings across several cities. Machine learning could help prioritise retrofit — but you are not allowed to upload everyone’s raw data into one central lake.  

This project — **PP-XAI** — asks: can we still get accurate predictions **and** clear explanations, while keeping the data local?”

### [0:45–2:00] Problem (non-specialist)
“In the UK, **Energy Performance Certificates** describe how buildings use energy. They are open data, and they are valuable for prediction models.  

The catch: for sensitive assets — high-rise homes and large commercial sites — organisations often cannot pool microdata. Standard AI training assumes everything sits in one spreadsheet.  

So I built a system that trains models **where the data already lives**, shares only **model updates**, and then explains **why** a prediction was made.”

### [2:00–3:30] What was developed (keep light)
“I filtered UK EPC records for London, Manchester, and Birmingham — about **five and a half thousand** buildings after honest high-rise rules.  

I trained strong **centralised** baselines — including gradient boosting — for comparison.  

In parallel, I trained a neural network with **federated averaging** across the three cities, so raw CSV rows never leave their city partition.  

Then I applied **SHAP** for global feature importance and **LIME** for individual building explanations — and I checked whether those explanations stayed consistent between the federated and centralised models.  

Finally, I wrapped the best model in a **Streamlit** web app so a planner could try a prediction interactively.”

*(Optional one sentence on options: “I deliberately compared approaches — for example neural nets for federated training, because tree models don’t average cleanly across clients.”)*

### [3:30–5:30] LIVE DEMO (visuals over talk)
“Here’s the running software.”

1. Show home / input panel.  
2. Enter a plausible building (city, area, storeys, fuel/type).  
3. Click predict — read the energy intensity aloud.  
4. Show the explanation panel: “These are the features pushing the prediction up or down.”  
5. Change one input (e.g. fuel or property type) and predict again: “See how the output and drivers respond.”  

Keep narration short. Let the UI do the work.

### [5:30–7:30] Results (three numbers only)
“Three headline results.  

**First — accuracy.** The best centralised model reached R-squared about **zero point five six**. The federated neural model reached about **zero point five two** — almost the same as a neural net trained on pooled data.  

**Second — statistics.** The gap versus the best tree model is statistically detectable, but the **effect size is tiny**, and confidence intervals overlap. Under a privacy constraint, that is a small practical price.  

**Third — explanations.** Feature rankings from SHAP correlate at about **zero point nine six** between the federated model and the best centralised model. So keeping data local did **not** scramble the story of what matters.”

Show `results_headline.png` and optionally one convergence or SHAP figure — not a table dump.

### [7:30–9:00] Critical honesty
“Two important caveats.  

The filtered dataset is overwhelmingly **large commercial** buildings. True residential towers are hard to identify in bulk EPC storey fields — I document that limitation rather than hide it.  

And this federated setup is a **research simulator**. It preserves data residency, but it is not yet hardened with differential privacy or secure aggregation — that is future work.  

I also removed features that effectively leaked the answer — early trials with emissions fields produced unrealistically high scores. Reporting the honest, lower scores was the right scientific choice.”

### [9:00–9:45] Close
“In short: privacy-preserving training can approach centralised neural accuracy, keep explanations stable, and ship as a working demonstrator.  

That is PP-XAI. Thank you for watching — student ID **202440724**.”

---

## Recording checklist

- [ ] Face clearly visible for the whole video  
- [ ] Mic test; quiet room; 1080p if possible  
- [ ] Streamlit running locally before you press record  
- [ ] Filename uses student ID only, e.g. `202440724_PPXAI_video.mp4`  
- [ ] Final length **under 10:00** (check Canvas overlength rules)  
- [ ] No need for Hollywood polish — clarity beats effects  

## Editing tips

- Cut dead air between segments  
- Soft zoom on the demo when prediction updates  
- Burn-in captions for the three headline numbers if useful  
- End card: project title + student ID (no name)
