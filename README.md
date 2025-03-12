# Fact_checking_reasoner


### Main Scripts

- **`reasoner.py`** - Applies the reasoner to analyze the relationships between claims and evidence based on similarity, polarity, and causality.
- **`evaluate_metrics.py`** - Evaluates the performance of the predictions using appropriate metrics.

### Causlality Extraction Methods

##### [Causality_extraction_across_Claim_and_Evidence](./Causality_extraction_across_Claim_and_Evidence/)`

This folder contains scripts for inferring causality between claims and evidence using Large Language Models (LLMs) or by training on common sense data.
###### `inference_LLMs/`

- **Run inference:**
  ```bash
  python LLM_inference.py
  ```

###### [Common Sense-based Causality Extraction](Fact_checking_reasoner/Causality_extraction_across_Claim_and_Evidence/Common_Sense-based_Causality_Extraction/)

This folder contains Pretrained model-based approach to infer causality between claim and evidence events using common sense data.

- **Train the model on common sense data:**
  ```bash
  python train.py
  ```
  
##### [causality_extraction_within_claim_and_evidence](./causality_extraction_within_claim_and_evidence/)
     -The dataset used for training in this project is released as part of the work by [Rebboud et al. (2023)](https://hal.science/hal-04121015).
   - **Train the model on the aformentioned dataset:**
  ```bash
  python train.py
  ```

### [data](./data/)
Data folder contains teh use cases filtered for testing our reasoner togther with common sense trainig data 


## Usage

1.**Choose the causlality extraction method across claim and evidence  method:**
2.**Choose the causlality extraction method within claim and evidence method:**   
3. **Run reasoning on a dataset** 
4. **Evaluate the predictions** 






