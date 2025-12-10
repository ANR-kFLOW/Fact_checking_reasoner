# Causal Reasoner for Fact-Checking

A causal explanation-based verdict prediction system that relies on semantically-precise event relations (cause, prevent, intend, enable), derived from the [FARO ontology](https://anr-kflow.github.io/faro/). The approach only applies to claims that include at least one causal relation between events.

The full description of this work is contained in the [EXTENDED VERSION of the accepted KNLP/SAC'26 paper]('KNLP_2026_Causal_Reasoning_for_Fact_Checking - extended version.pdf').

For citing this work, please use ([bibtex](rebboud2026causalfactcheck.bib)):

> Youssra Rebboud, Pasquale Lisena, and Raphael Troncy. 2026. Integrating Causal Reasoning into Automated Fact-Checking. In The 41st ACM/SIGAPP Symposium on Applied Computing (SAC ’26), March 23–27, 2026, Thessaloniki, Greece. ACM, New York, NY, USA, 3 pages. https://doi.org/10.1145/3748522.3779831

### Main Scripts

- **`reasoner.py`** - Applies the reasoner to analyze the relationships between claims and evidence based on similarity, polarity, and causality.
- **`evaluate_metrics.py`** - Evaluates the performance of the predictions using appropriate metrics.

### Causality Extraction Methods

##### [Causality_extraction_across_Claim_and_Evidence](./Causality_extraction_across_Claim_and_Evidence/)`

This folder contains scripts for inferring causality between claims and evidence using Large Language Models (LLMs) or by training on common sense data.

###### `LLMs based causality extraction`

**Run inference:**

  ```bash
  python LLM_inference.py
  ```

###### `Common Sense-based Causality Extraction`

This folder contains Pretrained model-based approach to infer causality between claim and evidence events using common sense data.

**Train the model on common sense data:**

  ```bash
  python train.py
  ```
  
##### [causality_extraction_within_claim_and_evidence](./causality_extraction_within_claim_and_evidence/)

The dataset used for training in this project is released as part of the work by [Rebboud et al. (2023)](https://hal.science/hal-04121015).

**Train the model on the aforementioned dataset:**

  ```bash
  python train.py
  ```

##### [data](./data/)

Data folder contains the use cases filtered for testing our reasoner together with common sense training data.

## Usage

1. Choose the Causality extraction method across claim and evidence method.
2. Choose the Causality extraction method within claim and evidence method.
3. Run reasoning on a dataset
4. Evaluate the predictions.
