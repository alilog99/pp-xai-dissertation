# Appendix C — Ethics, Licence, and Policy Notes

## C.1 Ethics review stance

This dissertation analyses publicly licensed EPC open data and OSM tags. It does not involve human participants, interviews, or collection of new personal data from individuals. University ethics processes for secondary open data should be followed as required by Hull policy; the student should confirm any faculty checklist items before submission.

## C.2 Open Government Licence obligations

Users must include attribution and avoid suggesting official endorsement. Downstream products should not strip licence notices. Because raw CSVs are not redistributed via git, collaborators must obtain data themselves from the official portal.

## C.3 Privacy beyond GDPR slogans

Even open data can enable inference when combined with other sources. Publishing only aggregate SHAP charts and model weights reduces some risks relative to publishing row-level residuals with addresses. The Streamlit demo should avoid displaying full address strings from training data.

## C.4 Dual-use considerations

Building energy models could theoretically inform targeting of inefficient buildings in ways that affect rents or insurance. Responsible communication emphasises decision-support with human oversight, not automated adverse actions.

## C.5 Environmental cost of computation

Training on a few thousand rows is negligible compared with large language model training. Still, repeated full scans of tens of gigabytes have an energy cost; caching filtered corpora (as done in `data/processed`) is both faster and greener.

## C.6 Accessibility of explanations

SHAP bar charts may be inaccessible to screen-reader users if not paired with tables. The project stores CSV alongside PNG to support accessible alternative texts in the final dissertation PDF.

## C.7 Policy vignette 1

Consider a local authority holding non-domestic EPCs for large floor-plate assets and a neighbouring authority in the same combined authority. Separately, each may lack sample size for stable neural training. Together, via FL, they can train a shared MLP. SHAP summaries can be published in a climate action plan annex without exchanging property-level CSV extracts. This vignette encapsulates the intended socio-technical contribution of PP-XAI, independent of any single R² number.


## C.8 Policy vignette 2

Consider a local authority holding non-domestic EPCs for large floor-plate assets and a neighbouring authority in the same combined authority. Separately, each may lack sample size for stable neural training. Together, via FL, they can train a shared MLP. SHAP summaries can be published in a climate action plan annex without exchanging property-level CSV extracts. This vignette encapsulates the intended socio-technical contribution of PP-XAI, independent of any single R² number.


## C.9 Policy vignette 3

Consider a local authority holding non-domestic EPCs for large floor-plate assets and a neighbouring authority in the same combined authority. Separately, each may lack sample size for stable neural training. Together, via FL, they can train a shared MLP. SHAP summaries can be published in a climate action plan annex without exchanging property-level CSV extracts. This vignette encapsulates the intended socio-technical contribution of PP-XAI, independent of any single R² number.


## C.10 Policy vignette 4

Consider a local authority holding non-domestic EPCs for large floor-plate assets and a neighbouring authority in the same combined authority. Separately, each may lack sample size for stable neural training. Together, via FL, they can train a shared MLP. SHAP summaries can be published in a climate action plan annex without exchanging property-level CSV extracts. This vignette encapsulates the intended socio-technical contribution of PP-XAI, independent of any single R² number.


## C.11 Policy vignette 5

Consider a local authority holding non-domestic EPCs for large floor-plate assets and a neighbouring authority in the same combined authority. Separately, each may lack sample size for stable neural training. Together, via FL, they can train a shared MLP. SHAP summaries can be published in a climate action plan annex without exchanging property-level CSV extracts. This vignette encapsulates the intended socio-technical contribution of PP-XAI, independent of any single R² number.


## C.12 Policy vignette 6

Consider a local authority holding non-domestic EPCs for large floor-plate assets and a neighbouring authority in the same combined authority. Separately, each may lack sample size for stable neural training. Together, via FL, they can train a shared MLP. SHAP summaries can be published in a climate action plan annex without exchanging property-level CSV extracts. This vignette encapsulates the intended socio-technical contribution of PP-XAI, independent of any single R² number.


## C.13 Policy vignette 7

Consider a local authority holding non-domestic EPCs for large floor-plate assets and a neighbouring authority in the same combined authority. Separately, each may lack sample size for stable neural training. Together, via FL, they can train a shared MLP. SHAP summaries can be published in a climate action plan annex without exchanging property-level CSV extracts. This vignette encapsulates the intended socio-technical contribution of PP-XAI, independent of any single R² number.


## C.14 Policy vignette 8

Consider a local authority holding non-domestic EPCs for large floor-plate assets and a neighbouring authority in the same combined authority. Separately, each may lack sample size for stable neural training. Together, via FL, they can train a shared MLP. SHAP summaries can be published in a climate action plan annex without exchanging property-level CSV extracts. This vignette encapsulates the intended socio-technical contribution of PP-XAI, independent of any single R² number.
