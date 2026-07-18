# Abstract

Predicting building energy performance from Energy Performance Certificate (EPC) data can support retrofit prioritisation and portfolio analytics, but centralising microdata raises privacy and governance barriers—especially for high-rise residential and large commercial assets. This dissertation presents PP-XAI, a privacy-preserving explainable AI framework that combines Federated Averaging with SHAP and LIME explanations for high-rise energy prediction on UK EPC certificates. After auditing bulk export schemas, high-rise filters were revised: domestic flats with storey count ≥ 5 in London, Manchester, and Birmingham clients, and non-domestic buildings with floor area ≥ 5,000 m² excluding institutional types. Target-leaking emissions features were excluded. On 5,663 filtered records, centralised gradient boosting achieved R² ≈ 0.56; a federated MLP reached R² ≈ 0.52, closely matching a centralised MLP. Spearman correlation of mean absolute SHAP values between centralised and federated models was ≈ 0.96. A Streamlit prototype demonstrates interactive prediction with global explanations. The results indicate that federated learning can approach centralised neural accuracy while preserving explanation stability, with honest limitations regarding domestic tower coverage in bulk EPC storey fields.

**Keywords:** Federated Learning; Explainable AI; SHAP; LIME; Energy Performance Certificates; High-rise buildings; Privacy-preserving machine learning

# References (starter list — expand before submission)

Amasyali, K. and El-Gohary, N.M. (2018) ‘A review of data-driven building energy consumption prediction studies’, *Renewable and Sustainable Energy Reviews*, 81, pp. 1192–1205.

Beutel, D.J. et al. (2020) ‘Flower: A Friendly Federated Learning Research Framework’, *arXiv preprint* arXiv:2007.14390.

Doshi-Velez, F. and Kim, B. (2017) ‘Towards a rigorous science of interpretable machine learning’, *arXiv preprint* arXiv:1702.08608.

Li, T. et al. (2020) ‘Federated optimization in heterogeneous networks’, *Proceedings of Machine Learning and Systems*, 2, pp. 429–450.

Lundberg, S.M. and Lee, S.-I. (2017) ‘A unified approach to interpreting model predictions’, *Advances in Neural Information Processing Systems*, 30.

McMahan, H.B. et al. (2017) ‘Communication-efficient learning of deep networks from decentralized data’, *AISTATS*.

Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) ‘“Why should I trust you?” Explaining the predictions of any classifier’, *KDD*.

Slack, D. et al. (2020) ‘Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods’, *AIES*.

Wei, Y. et al. (2018) ‘A review of data-driven approaches for prediction and classification of building energy consumption’, *Renewable and Sustainable Energy Reviews*, 82, pp. 1027–1047.

UK Government (n.d.) *Energy Performance of Buildings Data: England and Wales*. Open Government Licence v3.0. Available at: https://get-energy-performance-data.communities.gov.uk/


## Acknowledgements (draft)

I thank my supervisor Mona for guidance, and the University of Hull MSc AI programme. EPC open data providers are acknowledged under OGL v3.0. Any errors remain my own.

## Glossary

- **EPC:** Energy Performance Certificate
- **FL:** Federated Learning
- **FedAvg:** Federated Averaging
- **SHAP:** SHapley Additive exPlanations
- **LIME:** Local Interpretable Model-agnostic Explanations
- **OGL:** Open Government Licence
- **UPRN:** Unique Property Reference Number
- **PP-XAI:** Privacy-Preserving Explainable AI (this project)
