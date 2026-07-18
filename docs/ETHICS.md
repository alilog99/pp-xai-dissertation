# Ethics statement — PP-XAI

**Project:** Privacy-Preserving Explainable AI for Building Energy Performance Prediction  
**Student:** Syed Ali Raza (202440724)  
**Institution:** University of Hull — MSc Applied Artificial Intelligence

This statement summarises how the project aligns with University of Hull research ethics expectations for dissemination. It is **not** a substitute for the University’s official policy.

## Official University policy (do not copy into this repo)

The controlling document is the University of Hull **Research Ethics Policy**, published on the University website:

- Policy PDF (official): https://www.hull.ac.uk/asset-library/docs/research-ethics-policy.pdf  
- Research integrity & governance: https://www.hull.ac.uk/research/research-with-us/integrity-and-governance  
- Research policies index: https://www.hull.ac.uk/policies-and-information/research  

Downloaded copies are **uncontrolled**. This repository **does not** redistribute the University’s policy PDF (copyright / republication of institutional policy documents). Researchers should consult the official URL above.

## Nature of this research

| Aspect | Status in PP-XAI |
|---|---|
| Human participants / interviews / surveys | **None** |
| Animals / human tissue | **None** |
| Primary identifiable personal data collection | **None** |
| Data source | Secondary **open** UK EPC certificate bulk data (public register extracts) |
| Licence | Open Government Licence v3.0 — see [DATA_LICENCE.md](DATA_LICENCE.md) |

The work is computational modelling on publicly released administrative/energy-certificate data, with a software prototype for demonstration.

## Ethical principles applied

Consistent with Hull Research Ethics Policy themes (beneficence / non-maleficence, integrity, responsible dissemination — including §28 Publication of research findings):

1. **Accurate reporting** — Metrics, limitations (e.g. mild non-IID, domestic high-rise data sparsity, revised high-rise filters), and methods are documented in `dissertation/` and `results/`.
2. **Electronic dissemination** — The same standards of honesty apply to this GitHub repository as to a printed dissertation (policy §28.5).
3. **Misuse awareness** — Predictions must not be presented as certified EPCs, tenancy decisions, or regulatory determinations without independent professional oversight (policy §28.4).
4. **Privacy narrative honesty** — Federated Learning keeps client partitions local *in the simulator*; it does **not** by itself provide differential privacy. Future hardening (e.g. DP-SGD) is noted as future work.
5. **No secrets in public artefacts** — API keys, `.env`, and raw multi-gigabyte CSVs are gitignored and not published.
6. **Attribution** — EPC data providers are acknowledged under OGL v3.0.

## Ethics review posture

Secondary analysis of **open government** EPC data without interaction with human subjects is treated as low-risk desk research. Formal faculty ethics applications, where required by programme rules, should follow Faculty Ethics Officer guidance. This repository’s public materials are intended to support transparent examination and reproducibility, not to replace institutional approval processes.

## Prototype disclaimer

The Streamlit app and trained models are a **research demonstrator**. They are not an official Energy Performance Certificate tool and must not be used as the sole basis for legal, financial, or housing decisions.

## Contact

For University ethics process questions: Faculty Ethics Officer / Research Governance (`researchgovernance@hull.ac.uk` as listed on University pages).  
For this dissertation codebase: see repository README and supervisor contact via University of Hull programme channels.
