# MSc Data Science Thesis — Income Prediction Pipeline

This repo contains the full pipeline I used for my thesis: preprocessing → splitting → model training → baseline evaluation → robustness tests → interpretability + subgroup analyses.
The code is organised by step number so the workflow is easy to follow and re-run.

## Project structure (what each folder does)

### `00_data/`
Data-related folder.
- This repo does **not** commit the raw dataset itself.
- If you see a `README.md` inside `00_data/`, it documents where the dataset came from and/or how it should be placed.

### `01_preprocessing/`
All preprocessing logic (raw → model-ready).
- `clean_data.py`  
  Cleans the raw data (basic cleaning, missing values handling, removing/renaming columns where needed).
- `encode_features.py`  
  Encodes categorical variables / prepares features for modelling.
- `feature_engineering.py`  
  Creates engineered features used in the experiments (and keeps feature logic consistent across models).

### `02_splitting/`
Train/validation/test split logic.
- `split_data.py`  
  Creates the splits used throughout the thesis. This is the “single source of truth” for which samples belong to train/test.

### `03_model_training/`
Model training + tuning scripts.
- `hyperparameter_search.py`  
  Runs hyperparameter tuning (where applicable).
- `train_rf.py`  
  Trains the Random Forest model.
- `train_gbdt.py`  
  Trains a Gradient Boosted Decision Trees model (implementation depends on the libraries used in the thesis).
- `train_xgboost.py`  
  Trains the XGBoost model.
- `__init__.py`  
  Makes the folder importable as a module.

### `04_baseline_performance/`
Baseline performance evaluation (clean / default setting).
- `evaluate_baseline.py`  
  Computes baseline metrics on the test set and exports results.
- `baseline_metrics.ipynb`  
  Notebook version for inspecting/plotting baseline metrics and generating thesis-ready outputs.

### `05_robustness/`
Robustness and stress testing.
- `add_noise.py`  
  Applies controlled perturbations/noise to the features (as used in the robustness experiments).
- `distribution_shift.py`  
  Simulates distribution shift / dataset shift experiments.
- `robustness_evaluation.ipynb`  
  Runs the robustness evaluation end-to-end and exports figures/tables.

### `06_interpretability/`
Interpretability + subgroup performance.
- `subgroup_analysis.py`  
  Creates subgroup breakdowns and subgroup-level metrics (fairness-style analysis).
- `subgroup_performance.ipynb`  
  Notebook for subgroup performance plots/tables (thesis outputs).
- `shap_analysis.ipynb`  
  Global interpretability using SHAP (feature importance patterns, etc.).
- `lime_analysis.ipynb`  
  Local interpretability using LIME (instance-level explanations; appendix-style outputs).

### `figures/`
Exported figures used in the thesis (PNG/PDF).

### `results/`
Exported tables/metrics (CSV/JSON). These are the files I reference when writing up results.

### `src/`
Shared helper code (utilities used across steps), if needed.

## How to run the project

### 1 Environment setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
### 2 Data setup
This repo does **not** include the raw dataset. Follow the instructions in:

- `00_data/README.md` (if present), and/or
- the preprocessing scripts in `01_preprocessing/`.

### 3 How to run (recommended order)
If you want to reproduce the full pipeline, run in this order:

1. **Preprocess**
   - `01_preprocessing/clean_data.py`
   - `01_preprocessing/encode_features.py`
   - `01_preprocessing/feature_engineering.py`

2. **Split**
   - `02_splitting/split_data.py`

3. **Train**
   - `03_model_training/train_rf.py`
   - `03_model_training/train_gbdt.py`
   - `03_model_training/train_xgboost.py`
   - (optional) `03_model_training/hyperparameter_search.py`

4. **Baseline evaluation**
   - `04_baseline_performance/evaluate_baseline.py`
   - or `04_baseline_performance/baseline_metrics.ipynb` for figures/tables

5. **Robustness**
   - `05_robustness/robustness_evaluation.ipynb`
   - (uses `add_noise.py` and/or `distribution_shift.py` internally)

6. **Interpretability + subgroup**
   - `06_interpretability/subgroup_performance.ipynb`
   - `06_interpretability/shap_analysis.ipynb`
   - `06_interpretability/lime_analysis.ipynb`

Outputs will be written to:
- `results/` (tables/metrics)
- `figures/` (plots)


## Reproducibility (how results are kept consistent)
I aimed to make the experiments reproducible in a practical “thesis workflow” way:
- **Fixed splits:** the same train/test split is reused across experiments (created in `02_splitting/split_data.py`).
- **Fixed seeds:** wherever randomness is used (splitting, training, sampling), the code uses a fixed random state/seed.
- **Saved outputs:** evaluation notebooks read from consistent inputs and export tables/figures to `results/` and `figures/`.
- **Environment pinned:** dependencies are captured in `requirements.txt`.

### Note on small variation (expected):
Some methods (especially **LIME**) involve internal sampling. Even with seeds, you can sometimes see small run-to-run differences in individual explanations. In the thesis, I focus on consistent patterns and (where relevant) aggregate results rather than relying on a single instance.

## Configuration (`.env`)
- `.env.example` shows the expected environment variables (paths, config, etc.).
- Copy it to `.env` and fill in values if you use environment-based paths.

```bash
cp .env.example .env
```
