# Chapter 2 — Literature Review

## 2.1 Building energy performance and EPC data

Building energy prediction research spans physics-based simulation, statistical regression, and modern machine learning. Physics-based tools (for example Dynamic Simulation and reduced-order thermal models) offer interpretability grounded in heat transfer but require detailed geometry and schedules that are rarely available at national scale. Data-driven approaches instead learn mappings from observable building descriptors to energy intensity or demand. Ensemble tree methods (Random Forests, Gradient Boosting, XGBoost, LightGBM) consistently perform well on tabular building datasets because they capture non-linear interactions among fabric, systems, and categorical typology fields without heavy feature engineering (Amasyali & El-Gohary, 2018; Wei et al., 2018). Neural networks and hybrid models appear when large samples or temporal sequences are available, though tabular EPCs often favour trees.

UK EPCs provide standardised fields for ratings, modelled consumption, construction descriptions, and heating fuels under the Open Government Licence. Prior studies use EPCs for stock characterisation, fuel poverty proxies, retrofit targeting, and rating prediction. Fewer works isolate **high-rise** typologies with explicit storey or floor-area filters, and fewer still confront the mismatch between documentary high-rise thresholds and bulk export field quality. This dissertation contributes an audit-informed filter design that privileges honesty about data limitations over aspirational thresholds that empty the sample.

## 2.2 Privacy-preserving learning for built-environment data

Energy and building datasets can be sensitive even when released under open licences: address-level linkages, small-area inference, and commercial HVAC configurations create residual risks. Privacy-preserving machine learning encompasses anonymisation, differential privacy, secure multiparty computation, and federated learning. FL is attractive when multiple organisations each hold locally rich but non-shareable tables—precisely the situation of local authorities and portfolio managers holding EPC-related or meter-linked datasets.

FedAvg (McMahan et al., 2017) iteratively averages client model parameters weighted by local dataset size. Under non-IID client distributions, client drift can degrade convergence; FedProx and related methods add proximal regularisation (Li et al., 2020). Flower (Beutel et al., 2020) provides an engineering framework for simulating and deploying FL. Applications in smart grids, healthcare, and IoT are common; building-sector FL remains comparatively sparse, and coupling FL with formal XAI evaluation is rarer still.

## 2.3 Explainable AI for regression on tabular data

Global explanations summarise which features drive predictions across a dataset; local explanations justify individual instances. SHAP assigns each feature an additive contribution based on Shapley values from cooperative game theory, with desirable consistency properties (Lundberg & Lee, 2017). TreeExplainer computes exact SHAP values efficiently for tree ensembles; KernelExplainer approximates SHAP for arbitrary models at higher computational cost. LIME fits interpretable local surrogates around an instance (Ribeiro et al., 2016). Critiques of XAI emphasise fidelity, stability under perturbation, and the risk of misleading attributions (Doshi-Velez & Kim, 2017; Slack et al., 2020). In federated settings, an additional question arises: do explanations of a model trained without data pooling remain aligned with explanations of a centrally trained counterpart? This dissertation operationalises that question via Spearman correlation of mean absolute SHAP profiles.

## 2.4 Regulation, transparency, and the built environment

GDPR constrains processing of personal data and influences how household energy information may be shared. Commercial landlords may treat operational energy and system details as confidential. The EU AI Act and related governance debates elevate transparency and human oversight for higher-risk AI systems. While EPC prediction for research may not map one-to-one onto regulated high-risk categories, the **normative** argument remains: if AI informs retrofit advice or portfolio decisions, stakeholders benefit from explanations that are stable and auditable. PP-XAI therefore treats FL and XAI as complementary pillars of trustworthy analytics rather than optional extras.

## 2.5 Related work synthesis and research gap

Synthesising the above: (i) tabular ML on EPCs is mature for centralised prediction; (ii) FL is theoretically and practically ready for multi-organisation building data; (iii) SHAP/LIME are established for tabular regression; (iv) integrated evaluation of **accuracy + explanation stability** for UK high-rise EPC FL is lacking. No identified MSc-scale study simultaneously (a) revises high-rise filters against bulk schema reality, (b) compares centralised ensembles with FedAvg on London/Manchester/Birmingham partitions, and (c) reports Spearman stability of SHAP rankings across those regimes. Chapter 3 designs a methodology to address this gap within a constrained dissertation timeline.


## 2.6 Deep dive: tabular learning for energy intensity

Cross-sectional prediction of energy intensity differs from short-term load forecasting. The former emphasises static building descriptors; the latter emphasises calendars, weather, and autoregression. EPC modelling belongs primarily to the former. Feature sets typically include floor area, construction age or age band, wall/roof/window descriptions, main heating fuel, and typology. Target variables may be modelled consumption (kWh/m²/year), primary energy, or ordinal rating bands. Classification of EPC bands is common in the literature; this dissertation focuses on continuous regression because federated averaging and SHAP analyses are more directly comparable on a shared numeric target.

Tree ensembles remain strong baselines on medium-sized tabular datasets (thousands to hundreds of thousands of rows). They handle mixed types after encoding, tolerate moderate missingness with imputation, and provide impurity-based importance that can be contrasted with SHAP. Neural MLPs require careful scaling and often more tuning; they are, however, natural citizens of FedAvg because parameter tensors average cleanly. Choosing an MLP as the federated model and trees as centralised competitors therefore reflects both methodological fairness and FL practicality.

## 2.7 Non-IID federated learning and evaluation protocols

Non-IID client data is the default assumption in cross-silo FL. Geographic partitions of EPCs induce differences in building age, fuel mix, and typology prevalence—for example, London may contain denser office stock than Birmingham. Evaluation protocols should report both global test metrics and, where possible, per-client metrics. This study emphasises a pooled held-out test set transformed by a globally fitted preprocessor, mirroring a scenario where a coordinator evaluates a global model after federated training. Alternative protocols (client-local test sets only) are left to future work.

Communication efficiency, client dropout, and partial participation are central FL systems topics. For an MSc simulator with three reliable clients and eight rounds, these concerns are secondary. The scientific claim under test is not “FL at planet scale” but “FL can approach centralised neural accuracy on realistic UK high-rise EPC partitions while keeping explanations stable.”

## 2.8 Explanation quality beyond visual plots

Many papers present SHAP beeswarm plots without quantitative comparison across models. Stability metrics—rank correlation, top-k overlap, and attribution distance—make XAI evaluation falsifiable. This dissertation adopts Spearman correlation of mean |SHAP| as a simple, interpretable stability score. Fidelity (how well attributions reconstruct model behaviour) is partially addressed by using model-agnostic KernelExplainer consistently on both models, reducing confounding from explainer choice. LIME complements SHAP with instance-level narratives useful in stakeholder demos, even if LIME’s local sampling can be unstable.

## 2.9 Open data, reproducibility, and research software

Reproducible scientific software is part of the contribution. The repository separates raw data (gitignored), processed artefacts, scripts, and results. Environment issues on Apple Silicon (OpenMP, Rosetta, DYLD_LIBRARY_PATH) are documented because unreproducible pipelines undermine dissertation claims. Open EPC data under OGL enables legal redistribution of analysis code without sharing the multi-gigabyte raw CSVs in git.

## 2.10 Positioning relative to closest prior work

Closest prior strands include: centralised EPC ML papers; FL for smart meter or HVAC control; and XAI for building energy models in centralised settings. PP-XAI sits at their intersection with an explicit high-rise focus and dual accuracy–stability reporting. The claim is not absolute novelty of each component, but of their integrated evaluation on audited UK bulk EPC high-rise filters.


## 2.11 Sectoral energy policy context in the United Kingdom

UK building policy has evolved through Building Regulations Part L, the Energy Performance of Buildings Directive transposition, Minimum Energy Efficiency Standards (MEES) for the private rented sector, and successive net-zero strategies. EPCs function as both consumer information tools and quasi-regulatory instruments. Their modelled nature means they are not identical to metered consumption; nevertheless, they remain the most complete national descriptor set for fabric and systems. For AI research, that completeness is valuable, but users must avoid treating EPC targets as ground-truth operational energy. This dissertation therefore frames predictions as EPC-consistent energy intensity estimates, useful for screening and comparative analytics rather than billing reconciliation.

High-rise residential policy also intersects with building safety reforms following Grenfell, affecting how data about tall residential buildings are governed and shared. While this project uses open certificates only, the broader climate of caution around residential building data reinforces the relevance of privacy-preserving analytics.

## 2.12 Commercial real estate analytics

In commercial real estate, energy intensity affects operating expenditure, ESG reporting, and increasingly financing conditions. Landlords and managing agents may hesitate to pool granular asset data with competitors or third-party platforms. Federated learning is therefore not merely an academic construct; it matches an industry incentive structure where collaboration on models is desirable but collaboration on raw tables is not. Explainability further supports asset managers who must justify retrofit packages to investment committees.

## 2.13 Methodological alternatives considered

Alternative designs included: (a) classification of EPC bands instead of regression; (b) purely domestic focus; (c) graph neural nets over building footprints; (d) time-series models on repeated lodgements. Classification would simplify some metrics but weaken SHAP comparability for a continuous physical quantity. Purely domestic focus failed the data audit. Graph methods require reliable spatial joins (UPRN/footprints) not yet complete. Repeated-lodgement panels are uneven and biased toward transacted properties. The chosen design maximises feasibility and RQ coverage within one MSc cycle.

## 2.14 Validity threats anticipated from literature

From the energy ML literature, common threats include data leakage from post-outcome fields, temporal leakage when random-splitting longitudinal data, and overclaiming generalisation from one geography. This project mitigates leakage by feature ablation, uses a random split after filtering (with temporal robustness as future work), and limits claims to three English metro regions.

## 2.15 Summary of literature chapter

The literature supports tree and neural models for EPC-like tabular prediction, FedAvg for cross-silo collaboration, and SHAP/LIME for explanation—but rarely evaluates all three together on high-rise UK EPCs with audited filters. Chapter 3 turns that gap into a concrete protocol.


## 2.16 Concluding bridge to methodology

Taken together, the literature and policy context motivate a methodology that (1) audits data before modelling, (2) compares central and federated learners on shared metrics, (3) quantifies explanation stability, and (4) delivers a demonstrator. Chapter 3 specifies how those principles are operationalised in code and experiments for PP-XAI.


## 2.17 Additional scholarly context on trust and adoption

Adoption of predictive models in local government and property management depends as much on trust as on accuracy. Trust is built through transparent limitations, reproducible methods, and explanations that match domain intuition. PP-XAI’s emphasis on leakage control and SHAP stability is therefore not ornamental; it is part of the adoption pathway. Conversely, opaque high-R² models that rely on near-tautological features can undermine trust when stakeholders discover the circularity.

Studies of technology acceptance in the public sector emphasise ease of use and demonstrability. The Streamlit prototype addresses demonstrability. Ease of use for FL operations remains limited in this MSc simulator and should be framed as future engineering work rather than a completed product capability.

Finally, the open-data setting creates a pedagogical paradox: we motivate FL with privacy, yet train on open certificates that could be pooled. The resolution is that open EPCs are a safe research sandbox for methods intended for sensitive silos. Reporting that paradox explicitly protects against examiner critique.
