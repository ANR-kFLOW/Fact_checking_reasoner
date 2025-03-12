# Fact_checking_reasoner


### Main Scripts

- **`reasoner.py`** - Applies the reasoner to analyze the relationships between claims and evidence based on similarity, polarity, and causality.
- **`evaluate_metrics.py`** - Evaluates the performance of the predictions using appropriate metrics.

### Inference Methods

#### `inference_LLMs/`

This folder contains scripts for inferring causality between claims and evidence using Large Language Models (LLMs).

- **Run inference:**
  ```bash
  python inference_LLMs/LLM_inference.py
  ```

#### `inference_model_based_on_sub_obj/`

This folder contains a model-based approach to infer causality between claim and evidence events.

- **Train the model on common sense data:**
  ```bash
  python inference_model_based_on_sub_obj/train.py
  ```
- **Run inference using the trained model:**
  ```bash
  python inference_model_based_on_sub_obj/inference.py
  ```



## Usage

1.**Choose an inference method:**
   - Use LLM-based inference (`LLM_inference.py`).
   - Use a model trained on common sense data (`train.py`, `inference.py`).
2. **Run reasoning on a dataset** using `reasoner.py`.
4. **Evaluate the predictions** with `evaluate_metrics.py`.






