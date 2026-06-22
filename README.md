# AI-Driven Next-Day Sales Forecasting for Retail Inventory Planning

This repository contains the full project submission for CSCI323 – Modern Artificial Intelligence at the University of Wollongong Dubai (UOWD).  
The project builds an AI forecasting pipeline to predict next-day sales (`OUT`) for an Oman-based agri, animal feed, and flour reseller, with the goal of supporting inventory and replenishment decisions.

## Repository Structure

```text
.
├── notebook/
│   └── CSCI323_Sales_Forecasting.ipynb
├── data/
│   └── dat.csv
├── pipeline/
│   └── data_extraction_and_preprocessing.py
├── presentation/
│   └── CSCI323_Sales_Forecasting_Canva_Ready.pptx
├── report/
│   └── CSCI323_Sales_Forecasting_Report.pdf
├── demo/
│   └── video_demo.mp4
└── README.md
```

> Adjust file names/paths above to match your actual layout.

---

## 1. Project Overview

**Business context.**  
The business is a reseller of agricultural, animal feed, and flour products operating in Oman, where daily inventory decisions depend on volatile demand influenced by seasonality, Islamic calendar events, weather, and product mix.  
The objective is to forecast next-day units sold (`OUT`) at the item level so that planners can choose appropriate replenishment quantities (`IN`) and reduce both overstock and stockout risk.

**AI framing.**
- Supervised learning, regression task.
- Time-dependent data, so chronological splits and time-series cross-validation are used.
- Comparison of classical ML, tree ensembles, and a 1-D CNN on the same forecasting objective.

---

## 2. Dataset

The main dataset used by the notebook is `dat.csv` (derived from `project_dataset_preprocessed.csv` in the Colab).

- **Rows:** ~46,600 daily records across items.
- **Columns:** 80+ features including operational, calendar, sector, and engineered features.
- **Date range:** From earliest `Date` in 2023 to latest in mid–2025 (check printout in Colab for exact range).
- **Target variable:** `OUT` – daily units sold, modeled as next-day sales in the forecasting setup.

**Core raw fields:**  
`Date`, `Open`, `IN`, `TOTAL`, `OUT`, `BALANCE`, `Product_Sector`

**Item indicators:** many `Item_...` binary columns indicating the specific SKU (e.g., feed type, flour SKU).

### Engineered Features

Feature groups are defined in the notebook under "Feature Groups and Leakage Policy":

- **Flag features:**  
  `discrepancy_flag`, `is_friday`, `is_gap_row`, `is_Friday_or_is_gap`, `is_ramadan_period`, `eidalfitrwindow`, `eidaladhawindow`

- **Proximity features:**  
  `eid_fitr_proximity`, `eid_adha_proximity`

- **Gregorian calendar numeric features:**  
  `year`, `month`, `day`, `day_of_year`, `day_of_week`, plus cyclic encodings `day_of_week_sin/cos`, `month_sin/cos`, `year_sin/cos`

- **Season and climate:**  
  `season_name`, `season_num`, `season_sin/cos`, `climate_curve`, simple weather features such as `real_max_temperature`, `real_rainfall_mm`

- **Hijri calendar:**  
  `HijriYear`, `HijriMonth`, `HijriDay`, `hijri_month_sin/cos`, `hijri_day_sin/cos` to capture Islamic-calendar-driven demand patterns

- **Lag and rolling statistics (per item):**
  - Lags: `out_lag_1/7/14/30`, `balance_lag_1/7/14/30`
  - Rolling means/SD: `in_roll_mean_7/30`, `in_roll_sd_7/30`, `out_roll_mean_7/30`, `out_roll_sd_7/30`, `balance_roll_mean_7/30`, `balance_roll_sd_7/30`

These engineered features are central to the feature justification plots (seasonality, day-of-week, Ramadan effect, autocorrelation, etc.) generated in Sections 9–10 of the notebook.

---

## 3. Data Extraction & Preprocessing Pipeline

The `pipeline/data_extraction_and_preprocessing.py` (name placeholder) encapsulates the steps demonstrated in the Colab under:

- `load_data()` – CSV loading, `Date` parsing, sorting by item and date, basic stats and sanity checks.
- `get_available_cols(df)` – selects numeric and categorical features that actually exist in the current dataset.
- Feature groups (`FLAG_COLS`, `PROXIMITY_COLS`, `NUMERIC_COLS`, `CATEGORICAL_COLS`) – central definition of what the models see.
- **Preprocessing pipeline:**
  - Numeric pipeline: `SimpleImputer(strategy="median")` -> (optional scaling if you add it).
  - Categorical pipeline: `SimpleImputer(strategy="most_frequent")` -> `OneHotEncoder(handle_unknown="ignore")`.
  - Combined using `ColumnTransformer` and wrapped with a final estimator in `Pipeline`.

### Leakage Policy

The leakage policy is explicitly documented at the top of the notebook and implemented in feature selection:

- Same-day `BALANCE` is not used directly as a feature because it contains information from after the target `OUT` is realized.
- Only lagged and rolling variants of `BALANCE` are used, which are based on strictly past data.
- `TOTAL = Open + IN` is considered safe, because `Open` equals previous day's `BALANCE` and `IN` is a decision known before the selling day begins.

This logic is important to call out in both the README and the report because it demonstrates time-aware feature design.

---

## 4. Modeling and Evaluation

All modeling code lives primarily in the notebook `CSCI323_Sales_Forecasting.ipynb` under sections 7–12.

### Models

- **Linear Regression** – baseline.
- **Random Forest Regressor** – tuned ensemble with ~300 trees, limited depth, min samples per leaf, and `n_jobs=-1`.
- **XGBoost Regressor** – gradient boosting with tuned tree count, learning rate, depth, and subsampling (enabled when `xgboost` is installed).
- **1-D CNN** – deep learning sequence model over a 14-day window of numeric features for each item.

All classical models share the same preprocessing pipeline and feature groups, providing apples-to-apples comparison.

### Validation Strategy

Two complementary evaluation setups are implemented:

1. **80/20 chronological split** — Train on earliest 80% of the timeline, test on the most recent 20%. Simulates "train on history, predict the future" realistically.
2. **Time-series cross-validation (`TimeSeriesSplit`)** — Multiple expanding-window folds (`N_CV_SPLITS = 3`). For each model configuration, metrics are computed per fold and then averaged.

Time-series CV results are saved as:
- `model_results_all.csv` – per-fold metrics.
- `model_results_summary.csv` – averaged metrics.

### Key Metrics

The common metric helper `evaluate(y_true, y_pred)` returns MAE, RMSE, and MAPE, with clipping to non-negative predictions and safe handling of zero targets.

**Average CV metrics:**

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Linear Regression | 17.9001 | 40.6667 | 223.5937% |
| Random Forest | 7.9007 | 28.9144 | 123.6138% |
| XGBoost | 7.9635 | 29.0017 | 121.9425% |

Random Forest is selected as the final model because it offers the best overall trade-off between absolute and squared error while keeping the pipeline relatively simple to deploy to production.

The CNN experiment achieves test metrics of MAE ≈ 10.94, RMSE ≈ 42.97, MAPE ≈ 110.56%, which is competitive but not clearly superior to the ensembles on this dataset.

---

## 5. Inference & Business Scenario Notebook Sections

Section 13 of the notebook showcases a single-item forecast and business recommendation.

- Choose an item (`PREDICT_ITEM`, e.g., `"Starter"`), a future date (`PREDICT_DATE`), and a candidate inbound stock `IN` (`PREDICT_IN`).
- Use the trained Random Forest pipeline to compute:
  - Opening stock based on the latest historical `BALANCE`.
  - Recomputed calendar, Hijri, and weather features for that date.
  - Lag and rolling features from the item's recent history.
- Get a predicted next-day `OUT` (non-negative integer), plus a curve of `OUT` vs `IN` to reason about how much inbound stock is justified.

This section is the bridge from technical forecasting to decision support for planners.

---

## 6. Presentation (PPTX)

The `presentation/CSCI323_Sales_Forecasting_Canva_Ready.pptx` file is a Canva-ready slide deck that follows the CSCI323 presentation rubric.

It covers:
- Title, team details, domain context, and business problem.
- State-of-the-art AI in retail/demand forecasting (baseline vs ensemble vs deep models).
- Data and features, including calendar, Hijri, weather, lag, and rolling windows.
- Modeling and validation approach (chronological split + time-series CV).
- Model comparison with the above metrics and key visualizations.
- Business impact, limitations, and future work (deployment roadmap, extra signals).

You can import this PPTX into Canva, apply a theme, and plug in charts (`fig1`–`fig16` etc.) generated by the notebook.

---

## 7. Report

`report/CSCI323_Sales_Forecasting_Report.pdf` (name placeholder) is the written report that complements the notebook and slide deck. It includes:

- **Introduction & domain background** – Retail AI, inventory and demand forecasting context.
- **Literature review** – AI applications in retail, tree-based models for forecasting, deep learning for time series, and industry case studies (15–20 sources, 2022–2026).
- **Methodology** – Data description, preprocessing, feature groups, leakage policy, model selection, hyperparameters, and validation.
- **Results** – Quantitative performance, visualizations, error analysis, and business interpretation.
- **Discussion & limitations** – Data issues, model assumptions, generalization, and robustness.
- **Future work** – Additional data sources, advanced architectures, deployment plans, and monitoring.

---

## 8. Video Demo

The `demo/video_demo.mp4` file contains a short screencast demonstrating:

- The notebook workflow (data load, training, evaluation, and plots).
- The single-item inference tool and `OUT` vs `IN` scenario chart.
- A brief walkthrough of the PowerPoint and how it ties back to the code and results.

**Typical flow for the video:**
- 0–2 min: Project intro and business problem.
- 2–7 min: Notebook tour (data, features, models, validation).
- 7–10 min: Inference scenario and planner recommendation.
- 10–12 min: Short look at the slide deck.

---

## 9. How to Run Locally

### Requirements

Python 3.10+ (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
jupyter
```

### Steps

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook notebook/CSCI323_Sales_Forecasting.ipynb
```

4. Run the pipeline script directly (optional):

```bash
python pipeline/data_extraction_and_preprocessing.py
```

---

## 10. Reproducibility and Randomness

A global `RANDOM_SEED = 42` is used for scikit-learn models and XGBoost to ensure reproducible splits and training.  
TensorFlow models attempt to respect the same seed, but deep learning training may still have some non-determinism depending on hardware and backend.

---

## 11. License

Add your license here (e.g., MIT, Apache-2.0, or university-specific).

---

## 12. Acknowledgements

- **Course:** CSCI323 – Modern Artificial Intelligence, UOWD.
- **Instructors:** Dr. Milan Dordevic, Dr. Abdullah El Nokiti, Ms. Asma Damankesh, Ms. Syama Kurungodathil.
- Any collaborators, data providers, or tooling you wish to acknowledge.
