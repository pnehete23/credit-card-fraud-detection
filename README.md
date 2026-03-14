# Credit Card Fraud Detection — End-to-End ML/DL Pipeline

**Team:** Nadezhda Shiroglazova, Prathamesh Nehete, Sheng Hu  
**Course:** MSDS 422 — Practical Machine Learning  
**Date:** February 2026  
**Methodology:** CRISP-DM  

---

## Problem Statement

Credit card fraud causes billions in annual losses globally. This project builds an automated, real-time fraud detection pipeline on a highly imbalanced dataset (0.172% fraud rate), with emphasis on realistic evaluation protocols and business-oriented recommendations.

> **Research question:** Can we reliably identify fraudulent credit card transactions from anonymized transaction data, and what patterns distinguish fraud from legitimate purchases?

**Answer:** Yes — XGBoost catches 77.6% of all fraud while generating false alerts on only 0.007% of legitimate transactions, delivering an estimated net annual benefit of ~$4.84M on a 10M transaction/year volume.

---

## Dataset

**Kaggle Credit Card Fraud Detection Dataset**
- 284,807 transactions from European cardholders (September 2013, 2 days)
- 492 fraudulent transactions (0.172% — extreme class imbalance)
- 28 PCA-transformed anonymous features (V1–V28) + `Time` + `Amount`
- Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> **Note:** The dataset is loaded automatically via `kagglehub` in the notebook. No manual CSV placement is required.

---

## Models Implemented

| # | Model | Type | Imbalance Strategy | Tuning Method |
|---|-------|------|--------------------|---------------|
| 1 | Logistic Regression | Linear baseline | `class_weight='balanced'` | GridSearchCV (5-fold, 24 combos) |
| 2 | Random Forest | Ensemble (Bagging) | `class_weight='balanced'` | RandomizedSearchCV (5-fold, 20 iter) |
| 3 | XGBoost | Ensemble (Boosting) | `scale_pos_weight ≈ 577` | RandomizedSearchCV (5-fold, 20 iter) |
| 4 | Neural Network (BCE) | Deep Learning | `class_weight={0:1, 1:50}` | Early stopping on val PR-AUC |
| 5 | Neural Network (Focal) | Deep Learning | Focal Loss γ=2, α=0.65 | Early stopping on val PR-AUC |

> **No resampling (SMOTE/undersampling) is used in this notebook.** All imbalance correction is handled through loss-function weighting, which avoids synthetic data artifacts on the 492-sample minority class.

---

## Best Model Results

**XGBoost** — selected by validation PR-AUC (0.8072), confirmed on held-out test set:

| Metric | Validation | Test Set |
|--------|-----------|----------|
| PR-AUC (primary) | 0.8072 | **0.8823** |
| F1-Score | 0.8372 | 0.8539 |
| Recall (fraud caught) | 73.5% | **77.6% (76/98)** |
| Precision | 97.3% | 95.0% |
| ROC-AUC | 0.9737 | 0.9809 |
| MCC | 0.8453 | 0.8581 |
| False alarm rate | — | **0.007%** of legitimate transactions |
| Decision threshold | — | 0.988 |

**Why XGBoost over alternatives:**
- Highest PR-AUC with smallest CV-to-validation gap (0.055)
- Inference < 1 ms per transaction (real-time deployable)
- 9× faster to train than Random Forest (83s vs 727s)
- SHAP-explainable for regulatory compliance

---

## Notebook Structure

The analysis is contained in a single Jupyter notebook (`CreditCard_Fraud_Final_v2.ipynb`) organized as follows:

| Section | Content |
|---------|---------|
| **Executive Summary** | Management-level results, best model, core benefit, recommendations |
| **Problem Statement** | Research question, dataset description, 4 specific objectives |
| **Literature Review** | 6 papers on fraud detection, class imbalance, tree/DL methods |
| **1. Setup & Data Loading** | `kagglehub` download, initial shape/dtype inspection |
| **Tooling & Environment** | Python 3.11, library versions, CPU hardware, reproducibility |
| **2. Initial EDA** | Class distribution, Amount/Time distributions, skewness/kurtosis |
| **3. Expanded EDA** | V-feature class separation, correlation matrix, outlier analysis, hypotheses |
| **4. Data Preparation** | 70/10/20 stratified split, feature engineering, winsorization, scaling |
| **5. Original vs Prepared** | Visual before/after comparison of preprocessing |
| **6. Data Leakage Checklist** | 9-point verification that no leakage occurred |
| **7. Data Ready** | Final dataset shapes and preprocessing summary |
| **8. Baseline Model** | Dummy classifier proving accuracy is meaningless (99.83%, zero fraud caught) |
| **9. Random Forest** | RandomizedSearchCV, 5-fold CV, validation evaluation |
| **10. Logistic Regression** | GridSearchCV, 5-fold CV, regularization sensitivity |
| **11. XGBoost** | RandomizedSearchCV, 5-fold CV, CV overfitting analysis |
| **12. Neural Network** | BCE vs Focal Loss comparison, architecture search, early stopping |
| **13. Model Comparison** | Multi-metric bar charts, PR curves, precision-recall scatter, CV-val gap table |
| **→ CSV Export** | `model_comparison.csv` for Power BI (validation + test + per-fold CV rows) |
| **14. Final Test Evaluation** | Single held-out test set evaluation of XGBoost |
| **15. Findings & Conclusions** | Key findings, business impact ($4.84M/yr), overfitting analysis, deployment strategy |
| **Lessons Learned** | 6 methodological lessons, next steps, third-party dataset recommendations |
| **References** | 10 Chicago-style citations |

---

## Feature Engineering

Six new features are engineered from the two interpretable raw features (`Time`, `Amount`). V1–V28 are left untouched (already PCA-standardized):

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `Hour_Sin` | sin(2π × hour / 24) | Cyclical time encoding — hour 23 ≈ hour 0 |
| `Hour_Cos` | cos(2π × hour / 24) | Cyclical time encoding (orthogonal component) |
| `Is_Night` | 1 if 23:00–06:00 | Fraud concentration in off-peak hours |
| `Log_Amount` | log1p(Amount) | Reduces extreme right skew (skewness 16) |
| `Amount_Bin_Small` | 1 if Amount ≤ €5 | Captures card-testing micro-transactions |
| `Amount_Bin_Large` | 1 if Amount > €500 | Flags unusually large transactions |

**Preprocessing pipeline:**
1. 70/10/20 stratified split (before any transformation)
2. Feature engineering (element-wise, no fitting)
3. Winsorization of `Amount` (0.1th–99.9th percentile caps from training only)
4. `RobustScaler` on `Amount`, `Time`, `Log_Amount` (fit on training only)
5. V1–V28: no scaling (already PCA-standardized and orthogonal)

---

## Evaluation Protocol

- **Primary metric:** PR-AUC (Average Precision) — the 0.172% fraud rate makes accuracy and ROC-AUC misleadingly optimistic
- **Secondary metrics:** F1-Score, Recall, Precision, MCC, ROC-AUC
- **Threshold:** Optimized on validation set (maximize F1); operationally set by cost matrix
- **CV strategy:** StratifiedKFold (5 folds) preserving 0.172% fraud rate in every fold
- **Test set:** Held out throughout; evaluated exactly once on the final selected model

---

## Tooling, Environment, and Hardware

| Component | Detail |
|-----------|--------|
| Language | Python 3.11 |
| Environment | Jupyter Notebook (Anaconda) |
| Data loading | `kagglehub` (auto-downloads dataset) |
| ML library | scikit-learn 1.4 |
| Boosting | XGBoost 2.0 |
| Deep learning | TensorFlow / Keras 2.15 |
| Data | NumPy 1.26, Pandas 2.1 |
| Visualization | Matplotlib 3.8, Seaborn 0.13 |
| Hardware | Local CPU (no GPU required) |
| Random seed | `random_state=42` throughout |

---

## Project Structure

```
├── CreditCard_Fraud_Final_v2.ipynb   # Full end-to-end notebook (EDA → modeling → conclusions)
├── model_comparison.csv              # Power BI export: validation + test + CV per-fold metrics
├── fraud_model_dashboard.html        # Standalone interactive dashboard (open in any browser)
├── README.md                         # This file
└── creditcard.csv                    # Dataset (auto-downloaded via kagglehub, or place here)
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow

# 2. Open the notebook
jupyter notebook CreditCard_Fraud_Final_v2.ipynb

# 3. Run all cells (dataset downloads automatically via kagglehub)
#    Total runtime: ~20–30 min on CPU (RF search is the bottleneck at ~12 min)

# 4. Open fraud_model_dashboard.html in a browser for the interactive dashboard
```

> **Runtime note:** Random Forest RandomizedSearchCV (20 iter × 5-fold) takes ~727s. XGBoost takes ~83s. Neural network training completes in 40–80 epochs with early stopping.

---

## Key Findings

1. **XGBoost is the recommended production model** — best PR-AUC, smallest generalization gap, fastest training, SHAP-explainable
2. **Accuracy is actively dangerous as a KPI** — the dummy baseline scores 99.83% while catching zero fraud
3. **PR-AUC must be the primary metric** — ROC-AUC is misleadingly optimistic at 0.172% fraud rate; our Logistic Regression has ROC-AUC 0.966 yet the worst PR-AUC (0.665)
4. **V14, V17, V12, V10, V3** are the dominant fraud discriminators (class separation > 2σ)
5. **Decision threshold matters** — XGBoost's optimal threshold is 0.988, not the default 0.5
6. **Estimated net annual benefit** — ~$4.84M on 10M transaction/year volume (77.6% recall, 0.007% false alarm rate)

---

## Deployment Strategy (Summary)

- **Serving:** XGBoost as a REST API microservice; < 1 ms inference per transaction
- **Threshold:** 0.988 (validation-optimized); recalibrate quarterly using cost matrix
- **Review tiers:** Level 1 auto-rules → Level 2 analyst (5 min, prob > 0.998) → Level 3 soft-block (2 hr, prob 0.988–0.998)
- **Monitoring:** Monthly KS-test on V-feature distributions; weekly flag rate and confirmation rate tracking
- **Retraining:** Quarterly scheduled; 72-hour emergency retrain on confirmed new fraud pattern

---

## References

1. Alarfaj et al. (2022) — CNN architectures for fraud detection. *IEEE Access* 10: 39700–39715.
2. Dal Pozzolo et al. (2015) — Calibrating probability with undersampling. *IEEE SSCI*, 159–166.
3. Fariha et al. (2025) — Behavioral feature engineering for fraud detection. *IEEE Access* 13: 14872–14890.
4. Karunya et al. (2025) — Probability-based kNN for imbalanced fraud data. *IJIT* 17: 1219–1226.
5. LexisNexis Risk Solutions (2023) — *True Cost of Fraud Study*. Atlanta.
6. Marazqah Btoush et al. (2023) — Systematic review of ML/DL fraud detection. *IEEE Access* 11: 112607–112629.
7. Mienye & Jere (2024) — Deep learning approaches for fraud detection. *MAKE* 6(1): 516–537.
8. Nilson Report (2022) — Global card fraud losses. Issue 1209.
9. Popova & Gardi (2024) — Hybrid sampling strategies for fraud detection. *JFC* 31(3): 890–908.
10. Saito & Rehmsmeier (2015) — Precision-Recall vs ROC for imbalanced datasets. *PLoS ONE* 10(3): e0118432.
11. ULB Machine Learning Group & Worldline (2018) — Credit Card Fraud Detection. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
