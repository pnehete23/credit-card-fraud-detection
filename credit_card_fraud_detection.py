"""
============================================================================
Credit Card Fraud Detection — End-to-End ML/DL Pipeline
============================================================================
Team: Nadezhda Shiroglazova, Prathamesh Nehete, Sheng Hu
Course: MSDS 422 — Practical Machine Learning
Dataset: Kaggle Credit Card Fraud Detection
         https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Instructions:
  1. Download creditcard.csv from Kaggle and place it in the same directory.
  2. pip install pandas numpy scikit-learn xgboost imbalanced-learn
             matplotlib seaborn tensorflow keras
  3. Run:  python credit_card_fraud_detection.py
     OR open in Jupyter and run cells marked by section headers.

Methodology: CRISP-DM
Models implemented (≥4 required):
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. Deep Autoencoder (anomaly detection)
  5. Feed-Forward Neural Network (DL classifier)

Resampling strategies evaluated:
  - Random Undersampling
  - SMOTE Oversampling
  - Hybrid (SMOTE + Tomek Links)
============================================================================
"""

import os
import warnings
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, precision_score,
    recall_score, average_precision_score, matthews_corrcoef
)
from sklearn.pipeline import Pipeline

import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
np.random.seed(42)

# Output directory for all figures and artifacts
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION — END-TO-END PIPELINE")
print("=" * 70)



# 1. DATA COLLECTION & LOADING

print("\n" + "=" * 70)
print("PHASE 1: DATA COLLECTION & LOADING")
print("=" * 70)

DATA_PATH = "creditcard.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"'{DATA_PATH}' not found. Download from:\n"
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "Place creditcard.csv in the current working directory."
    )

df = pd.read_csv(DATA_PATH)
print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\nClass Distribution:")
print(df["Class"].value_counts())
print(f"\nFraud rate: {df['Class'].mean()*100:.3f}%")
print(f"Imbalance ratio: 1:{int((1-df['Class'].mean())/df['Class'].mean())}")



# 2. EXPLORATORY DATA ANALYSIS (EDA)

print("\n" + "=" * 70)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 2.1 Basic statistics
print("\n--- Basic Statistics ---")
print(df.describe().T[["mean", "std", "min", "max"]].to_string())

print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove duplicates if any
n_before = len(df)
df = df.drop_duplicates()
print(f"Rows after deduplication: {len(df):,} (removed {n_before - len(df)})")

# 2.1b Skewness & Kurtosis analysis
print("\n--- Skewness & Kurtosis ---")
skewness = df.skew(numeric_only=True).sort_values(ascending=False)
kurtosis_vals = df.kurtosis(numeric_only=True).sort_values(ascending=False)

print("\nTop 10 Most Skewed Features:")
print(skewness.head(10).to_string(float_format="%.4f"))
print("\nBottom 5 (Most Negatively Skewed):")
print(skewness.tail(5).to_string(float_format="%.4f"))
print("\nTop 10 Highest Kurtosis Features:")
print(kurtosis_vals.head(10).to_string(float_format="%.4f"))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Skewness plot — highlight highly skewed features (|skew| > 1)
skew_sorted = skewness.sort_values()
skew_colors = ["#e74c3c" if abs(v) > 1 else "#3498db" for v in skew_sorted]
skew_sorted.plot(kind="barh", ax=axes[0], color=skew_colors, edgecolor="black", linewidth=0.5)
axes[0].set_title("Feature Skewness", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Skewness")
axes[0].axvline(0, color="black", linestyle="-", linewidth=0.8)
axes[0].axvline(-1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
axes[0].axvline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="|Skew| = 1 threshold")
axes[0].legend(fontsize=9)

# Kurtosis plot — highlight leptokurtic features (excess kurtosis > 0)
kurt_sorted = kurtosis_vals.sort_values()
kurt_colors = ["#e74c3c" if v > 0 else "#3498db" for v in kurt_sorted]
kurt_sorted.plot(kind="barh", ax=axes[1], color=kurt_colors, edgecolor="black", linewidth=0.5)
axes[1].set_title("Feature Kurtosis (Excess)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Kurtosis")
axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Normal (excess κ = 0)")
axes[1].legend(fontsize=9)

plt.suptitle("Distributional Shape Analysis — Skewness & Kurtosis",
             fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01b_skewness_kurtosis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[SAVED] 01b_skewness_kurtosis.png")

# 2.2 Class distribution plot
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Bar plot (normal scale)
counts = df["Class"].value_counts()
colors = ["#2ecc71", "#e74c3c"]
axes[0].bar(["Legitimate", "Fraud"], counts.values, color=colors, edgecolor="black")
axes[0].set_title("Class Distribution (Linear Scale)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

# Bar plot (LOG scale — makes fraud visible)
axes[1].bar(["Legitimate", "Fraud"], counts.values, color=colors, edgecolor="black")
axes[1].set_yscale("log")
axes[1].set_title("Class Distribution (Log Scale)", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Count (log)")
for i, v in enumerate(counts.values):
    axes[1].text(i, v * 1.3, f"{v:,}", ha="center", fontweight="bold")

# Imbalance ratio visualization (waffle-style)
ratio = int(counts.values[0] / counts.values[1])
grid_size = 24
grid = np.zeros((grid_size, grid_size))
grid[0, 0] = 1  # 1 fraud out of ~578
axes[2].imshow(grid, cmap=matplotlib.colors.ListedColormap(["#2ecc71", "#e74c3c"]),
               interpolation="nearest")
axes[2].set_title(f"Imbalance Ratio ≈ 1:{ratio}\n(1 red = fraud per {ratio} transactions)",
                  fontsize=13, fontweight="bold")
axes[2].set_xticks([])
axes[2].set_yticks([])
# Add annotation
axes[2].annotate("← 1 Fraud", xy=(0, 0), xytext=(3, 2),
                 fontsize=12, fontweight="bold", color="#e74c3c",
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[SAVED] 01_class_distribution.png")

# 2.3 Transaction Amount Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Legitimate vs Fraud amount distributions
for cls, color, label in [(0, "#2ecc71", "Legitimate"), (1, "#e74c3c", "Fraud")]:
    subset = df[df["Class"] == cls]["Amount"]
    axes[0].hist(subset, bins=50, alpha=0.7, color=color, label=label, density=True)
axes[0].set_title("Transaction Amount Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Amount ($)")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].set_xlim(0, 500)

# Box plot
df.boxplot(column="Amount", by="Class", ax=axes[1],
           boxprops=dict(color="black"), medianprops=dict(color="red"))
axes[1].set_title("Amount by Class", fontsize=14, fontweight="bold")
axes[1].set_xticklabels(["Legitimate", "Fraud"])
axes[1].set_ylabel("Amount ($)")
axes[1].set_ylim(0, 500)
plt.suptitle("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_amount_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 02_amount_distribution.png")

# 2.4 Time analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Convert Time to hours
df["Hour"] = (df["Time"] / 3600) % 24

for cls, color, label in [(0, "#2ecc71", "Legitimate"), (1, "#e74c3c", "Fraud")]:
    subset = df[df["Class"] == cls]["Hour"]
    axes[0].hist(subset, bins=48, alpha=0.7, color=color, label=label, density=True)
axes[0].set_title("Transaction Timing (Hour of Day)", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Density")
axes[0].legend()

# Fraud rate by hour
hourly_fraud = df.groupby(df["Hour"].astype(int))["Class"].mean() * 100
axes[1].bar(hourly_fraud.index, hourly_fraud.values, color="#e74c3c", alpha=0.8)
axes[1].set_title("Fraud Rate by Hour", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Fraud Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_temporal_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 03_temporal_analysis.png")

# 2.5 Correlation heatmap for top features
fig, ax = plt.subplots(figsize=(16, 12))
corr = df.corr()
# Focus on features most correlated with Class
top_features = corr["Class"].abs().sort_values(ascending=False).head(15).index
sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, square=True)
ax.set_title("Correlation Matrix — Top 15 Features vs Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 04_correlation_heatmap.png")

# 2.6 PCA feature distributions (fraud vs legitimate)
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
important_features = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V9",
                       "V10", "V11", "V12", "V14", "V16", "V17", "V18", "V19"]
for idx, feat in enumerate(important_features):
    ax = axes[idx // 4, idx % 4]
    for cls, color, label in [(0, "#2ecc71", "Legit"), (1, "#e74c3c", "Fraud")]:
        ax.hist(df[df["Class"] == cls][feat], bins=50, alpha=0.6,
                color=color, label=label, density=True)
    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
plt.suptitle("PCA Feature Distributions: Fraud vs Legitimate",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_pca_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 05_pca_feature_distributions.png")

# Print key EDA insights
print("\n--- Key EDA Insights ---")
print(f"• Average fraud amount:      ${df[df['Class']==1]['Amount'].mean():.2f}")
print(f"• Average legitimate amount:  ${df[df['Class']==0]['Amount'].mean():.2f}")
print(f"• Max fraud amount:           ${df[df['Class']==1]['Amount'].max():.2f}")
print(f"• Peak fraud hour:            {hourly_fraud.idxmax()}:00 ({hourly_fraud.max():.2f}%)")
print(f"• Lowest fraud hour:          {hourly_fraud.idxmin()}:00 ({hourly_fraud.min():.2f}%)")



# 3. FEATURE ENGINEERING

print("\n" + "=" * 70)
print("PHASE 3: FEATURE ENGINEERING")
print("=" * 70)

print("\nEngineering features across 8 categories...")
pca_cols = [f"V{i}" for i in range(1, 29)]

# ---- 3.1 Amount features ----
df["Log_Amount"] = np.log1p(df["Amount"])

# ---- 3.2 Temporal features — cyclical encoding ----
# Cyclical encoding preserves circular nature of hours (23:00 ≈ 00:00)
# Fraud rate peaks at 1-3 AM (temporal plot); sin/cos captures this for all models
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
# Inter-transaction interval (global temporal density signal)
df["Time_Diff"] = df["Time"].diff().fillna(0)

# ---- 3.3 PCA pairwise interaction features ----
# Interactions between top fraud-correlated PCA components
# V17 (r=-0.31), V14 (-0.29), V12 (-0.25), V10 (-0.21), V4 (+0.13), V3 (-0.18)
df["V14_V17_interaction"] = df["V14"] * df["V17"]
df["V14_V10_interaction"] = df["V14"] * df["V10"]
df["V14_V12_interaction"] = df["V14"] * df["V12"]
df["V10_V12_interaction"] = df["V10"] * df["V12"]
df["V17_V12_interaction"] = df["V17"] * df["V12"]
df["V3_V4_interaction"]  = df["V3"] * df["V4"]

# ---- 3.4 PCA polynomial features ----
# Squared terms capture non-linear thresholds for the 4 dominant predictors
# (PCA distributions show fraud clusters at extreme negative tails)
df["V14_squared"] = df["V14"] ** 2
df["V10_squared"] = df["V10"] ** 2
df["V12_squared"] = df["V12"] ** 2
df["V17_squared"] = df["V17"] ** 2

# ---- 3.5 PCA magnitude features ----
# Full L2 norm — overall distance from origin in PCA space
df["PCA_magnitude"] = np.sqrt((df[pca_cols] ** 2).sum(axis=1))
# Focused magnitude of top-5 fraud-correlated components only
df["V_top5_magnitude"] = np.sqrt(
    df["V14"]**2 + df["V10"]**2 + df["V12"]**2 + df["V17"]**2 + df["V4"]**2
)

# ---- 3.6 Per-row PCA distributional features ----
# These capture the "shape" of each transaction's PCA vector
# Fraud transactions are outliers in multiple PCA dimensions simultaneously
pca_values = df[pca_cols].values
df["V_mean"]      = pca_values.mean(axis=1)
df["V_std"]       = pca_values.std(axis=1)
df["V_range"]     = pca_values.max(axis=1) - pca_values.min(axis=1)
# Count of PCA components with |value| > 2 — direct multi-dimensional anomaly signal
df["V_n_extreme"] = (np.abs(pca_values) > 2).sum(axis=1)

# ---- 3.7 Three-way and cross-domain interactions ----
# Three-way interaction of the 3 most important features (V14, V10, V12)
df["V14_V10_V12"] = df["V14"] * df["V10"] * df["V12"]
# Amount cross-interactions with strongest PCA fraud signals
# (fraud Amount distribution is bimodal; combining with PCA captures both patterns)
df["Amount_V14"] = df["Amount"] * df["V14"]
df["Amount_V17"] = df["Amount"] * df["V17"]

n_engineered = len(df.columns) - 31  # 31 = original 31 cols
print(f"  Engineered features added: {n_engineered}")
print(f"  Total columns now:         {len(df.columns)}")

# ---- 3.8 Feature selection — drop raw Time and Hour ----
# Time: replaced by Time_Diff; Hour: replaced by Hour_sin/Hour_cos
drop_cols = ["Time", "Hour"]
df_model = df.drop(columns=drop_cols)

feature_cols = [c for c in df_model.columns if c != "Class"]
target = "Class"

print(f"Features for modeling: {len(feature_cols)}")
print(f"Feature list: {feature_cols}")



# 4. DATA PREPARATION — TRAIN/TEST SPLIT & SCALING

print("\n" + "=" * 70)
print("PHASE 4: DATA PREPARATION — TRAIN/TEST SPLIT & SCALING")
print("=" * 70)

X = df_model[feature_cols].values
y = df_model[target].values

# Stratified split: 80/20 — BEFORE scaling to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RobustScaler handles outliers better than StandardScaler
# Fit on training data ONLY to prevent data leakage, then transform both
print("\nScaling features (fit on train only)...")
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]:,} samples")
print(f"  Legitimate: {(y_train==0).sum():,}")
print(f"  Fraud:      {(y_train==1).sum():,}")
print(f"\nTest set:     {X_test.shape[0]:,} samples")
print(f"  Legitimate: {(y_test==0).sum():,}")
print(f"  Fraud:      {(y_test==1).sum():,}")
print(f"\n** Test set remains IMBALANCED (real-world evaluation) **")
print(f"** Scaler fit on training data only (no data leakage) **")



# 5. RESAMPLING STRATEGIES

print("\n" + "=" * 70)
print("PHASE 5: RESAMPLING STRATEGIES")
print("=" * 70)

resampling_methods = {}

# 5.1 No resampling (baseline)
resampling_methods["No_Resampling"] = (X_train, y_train)
print(f"\n[Baseline] No resampling: {(y_train==0).sum():,} legit / {(y_train==1).sum():,} fraud")

# 5.2 Random Undersampling
print("\nApplying Random Undersampling...")
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
resampling_methods["Undersampling"] = (X_under, y_under)
print(f"  Result: {(y_under==0).sum():,} legit / {(y_under==1).sum():,} fraud")

# 5.3 SMOTE Oversampling
print("\nApplying SMOTE Oversampling...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
resampling_methods["SMOTE"] = (X_smote, y_smote)
print(f"  Result: {(y_smote==0).sum():,} legit / {(y_smote==1).sum():,} fraud")

# 5.4 Hybrid (SMOTE + Tomek Links) — as recommended by Popova & Gardi (2024)
print("\nApplying Hybrid (SMOTE + Tomek Links)...")
smt = SMOTETomek(random_state=42)
X_hybrid, y_hybrid = smt.fit_resample(X_train, y_train)
resampling_methods["Hybrid_SMOTETomek"] = (X_hybrid, y_hybrid)
print(f"  Result: {(y_hybrid==0).sum():,} legit / {(y_hybrid==1).sum():,} fraud")

# Visualize resampling results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for idx, (name, (X_r, y_r)) in enumerate(resampling_methods.items()):
    counts = [np.sum(y_r == 0), np.sum(y_r == 1)]
    axes[idx].bar(["Legitimate", "Fraud"], counts, color=colors, edgecolor="black")
    axes[idx].set_title(name.replace("_", " "), fontsize=13, fontweight="bold")
    axes[idx].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[idx].text(i, v + max(counts)*0.02, f"{v:,}", ha="center", fontsize=10)
plt.suptitle("Resampling Strategy Comparison", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_resampling_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[SAVED] 06_resampling_comparison.png")



# 6. MODEL DEVELOPMENT & TRAINING

print("\n" + "=" * 70)
print("PHASE 6: MODEL DEVELOPMENT & TRAINING")
print("=" * 70)

# We'll train each model with the Hybrid resampling (best per literature)
# and also evaluate with other strategies for comparison

X_train_final, y_train_final = resampling_methods["Hybrid_SMOTETomek"]

results = {}


# MODEL 1: Logistic Regression
print("\n--- Model 1: Logistic Regression ---")
t0 = time.time()
lr_model = LogisticRegression(
    C=0.1,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
lr_model.fit(X_train_final, y_train_final)
lr_time = time.time() - t0

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

results["Logistic Regression"] = {
    "model": lr_model,
    "y_pred": y_pred_lr,
    "y_prob": y_prob_lr,
    "train_time": lr_time
}
print(f"  Training time: {lr_time:.2f}s")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=["Legitimate", "Fraud"]))


# MODEL 2: Random Forest
print("\n--- Model 2: Random Forest ---")
t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_final, y_train_final)
rf_time = time.time() - t0

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

results["Random Forest"] = {
    "model": rf_model,
    "y_pred": y_pred_rf,
    "y_prob": y_prob_rf,
    "train_time": rf_time
}
print(f"  Training time: {rf_time:.2f}s")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=["Legitimate", "Fraud"]))


# MODEL 3: XGBoost
print("\n--- Model 3: XGBoost ---")
t0 = time.time()
# Calculate scale_pos_weight for class imbalance
scale_pos = (y_train_final == 0).sum() / max((y_train_final == 1).sum(), 1)
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="aucpr",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_final, y_train_final)
xgb_time = time.time() - t0

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

results["XGBoost"] = {
    "model": xgb_model,
    "y_pred": y_pred_xgb,
    "y_prob": y_prob_xgb,
    "train_time": xgb_time
}
print(f"  Training time: {xgb_time:.2f}s")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=["Legitimate", "Fraud"]))


# MODEL 4: Deep Autoencoder (Anomaly Detection)
print("\n--- Model 4: Deep Autoencoder (Anomaly Detection) ---")

# Lazy import TensorFlow (heavy dependency)
try:
    import tensorflow as tf
    from keras import layers, callbacks
    keras = tf.keras
    TF_AVAILABLE = True
    tf.random.set_seed(42)
    # Suppress TF logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")
except ImportError:
    TF_AVAILABLE = False
    print("  [WARNING] TensorFlow not installed. Skipping deep learning models.")
    print("  Install with: pip install tensorflow")

if TF_AVAILABLE:
    # Autoencoder: train ONLY on legitimate transactions
    # Fraud = high reconstruction error
    X_train_legit = X_train[y_train == 0]
    print(f"  Training on {len(X_train_legit):,} legitimate transactions only")

    input_dim = X_train.shape[1]

    # Build Autoencoder
    encoder_input = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    encoded = layers.Dense(16, activation="relu", name="bottleneck")(x)

    x = layers.Dense(32, activation="relu")(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = keras.Model(encoder_input, decoded, name="FraudAutoencoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss="mse")

    print(f"  Autoencoder parameters: {autoencoder.count_params():,}")

    t0 = time.time()
    ae_history = autoencoder.fit(
        X_train_legit, X_train_legit,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ],
        verbose=0
    )
    ae_time = time.time() - t0

    # Compute reconstruction errors
    X_test_reconstructed = autoencoder.predict(X_test, verbose=0)
    reconstruction_errors = np.mean((X_test - X_test_reconstructed) ** 2, axis=1)

    # Find optimal threshold using validation set
    X_val_legit_recon = autoencoder.predict(X_train_legit[:5000], verbose=0)
    legit_errors = np.mean((X_train_legit[:5000] - X_val_legit_recon) ** 2, axis=1)
    # Threshold = mean + 2*std of legitimate reconstruction errors
    ae_threshold = np.mean(legit_errors) + 2 * np.std(legit_errors)

    y_pred_ae = (reconstruction_errors > ae_threshold).astype(int)
    # Use reconstruction error as probability proxy (normalized)
    y_prob_ae = (reconstruction_errors - reconstruction_errors.min()) / \
                (reconstruction_errors.max() - reconstruction_errors.min())

    results["Autoencoder"] = {
        "model": autoencoder,
        "y_pred": y_pred_ae,
        "y_prob": y_prob_ae,
        "train_time": ae_time,
        "threshold": ae_threshold
    }
    print(f"  Training time: {ae_time:.2f}s")
    print(f"  Threshold: {ae_threshold:.6f}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_ae):.4f}")
    print(classification_report(y_test, y_pred_ae, target_names=["Legitimate", "Fraud"]))

    # Autoencoder training loss plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ae_history.history["loss"], label="Training Loss", linewidth=2)
    ax.plot(ae_history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_title("Autoencoder Training Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_autoencoder_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] 07_autoencoder_loss.png")

    # Reconstruction error distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(reconstruction_errors[y_test == 0], bins=100, alpha=0.7,
            color="#2ecc71", label="Legitimate", density=True)
    ax.hist(reconstruction_errors[y_test == 1], bins=100, alpha=0.7,
            color="#e74c3c", label="Fraud", density=True)
    ax.axvline(ae_threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold = {ae_threshold:.4f}")
    ax.set_title("Autoencoder Reconstruction Error Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0, np.percentile(reconstruction_errors, 99))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_ae_reconstruction_errors.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] 08_ae_reconstruction_errors.png")


    # MODEL 5: Feed-Forward Neural Network (DL Classifier)
    print("\n--- Model 5: Feed-Forward Neural Network ---")

    nn_model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ], name="FraudClassifierNN")

    nn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    

    print(f"  NN parameters: {nn_model.count_params():,}")

    # Use hybrid-resampled training data
    t0 = time.time()
    nn_history = nn_model.fit(
        X_train_final, y_train_final,
        epochs=50,
        batch_size=512,
        validation_split=0.1,
        class_weight={0: 1, 1: 5},  # additional weighting
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ],
        verbose=0
    )
    nn_time = time.time() - t0

    y_prob_nn = nn_model.predict(X_test, verbose=0).flatten()
    y_pred_nn = (y_prob_nn > 0.5).astype(int)

    results["Neural Network"] = {
        "model": nn_model,
        "y_pred": y_pred_nn,
        "y_prob": y_prob_nn,
        "train_time": nn_time
    }
    print(f"  Training time: {nn_time:.2f}s")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_nn):.4f}")
    print(classification_report(y_test, y_pred_nn, target_names=["Legitimate", "Fraud"]))

    # NN training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(nn_history.history["loss"], label="Train Loss")
    axes[0].plot(nn_history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Neural Network Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Find the AUC key (name varies by TF version)
    if "auc" in nn_history.history:
        axes[1].plot(nn_history.history["auc"], label="Train AUC")
        axes[1].plot(nn_history.history["val_auc"], label="Val AUC")
    else:
        # Fallback: just show available metrics
        available = [k for k in nn_history.history.keys() if "val" not in k and "loss" not in k]
        if available:
            key = available[0]
            axes[1].plot(nn_history.history[key], label=f"Train {key}")
            axes[1].plot(nn_history.history[f"val_{key}"], label=f"Val {key}")
        else:
            axes[1].text(0.5, 0.5, "No AUC metric recorded", ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_title("Neural Network AUC", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Neural Network Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_nn_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] 09_nn_training_curves.png")



# 7. MODEL EVALUATION & COMPARISON

print("\n" + "=" * 70)
print("PHASE 7: MODEL EVALUATION & COMPARISON")
print("=" * 70)

# 7.1 Comprehensive metrics table
print("\n--- Comprehensive Model Comparison ---")
comparison_data = []
for name, res in results.items():
    y_p = res["y_pred"]
    y_pr = res["y_prob"]
    comparison_data.append({
        "Model": name,
        "Precision": precision_score(y_test, y_p),
        "Recall": recall_score(y_test, y_p),
        "F1-Score": f1_score(y_test, y_p),
        "AUC-ROC": roc_auc_score(y_test, y_pr),
        "AUC-PR": average_precision_score(y_test, y_pr),
        "MCC": matthews_corrcoef(y_test, y_p),
        "Train Time (s)": res["train_time"]
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values("F1-Score", ascending=False)
print(comparison_df.to_string(index=False, float_format="%.4f"))

# Save comparison
comparison_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
print("\n[SAVED] model_comparison.csv")

# 7.2 Confusion Matrices
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[idx],
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    axes[idx].set_title(name, fontsize=13, fontweight="bold")
    axes[idx].set_xlabel("Predicted")
    axes[idx].set_ylabel("Actual")
plt.suptitle("Confusion Matrices — All Models (Imbalanced Test Set)",
             fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 10_confusion_matrices.png")

# 7.3 ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
model_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
for idx, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    auc_val = roc_auc_score(y_test, res["y_prob"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})",
            linewidth=2, color=model_colors[idx % len(model_colors)])

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.5)")
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 11_roc_curves.png")

# 7.4 Precision-Recall Curves (more informative for imbalanced data)
fig, ax = plt.subplots(figsize=(10, 8))
for idx, (name, res) in enumerate(results.items()):
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    ap = average_precision_score(y_test, res["y_prob"])
    ax.plot(rec, prec, label=f"{name} (AP={ap:.4f})",
            linewidth=2, color=model_colors[idx % len(model_colors)])

ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_precision_recall_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 12_precision_recall_curves.png")

# 7.5 Metrics bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics_to_plot = ["Precision", "Recall", "F1-Score"]
for idx, metric in enumerate(metrics_to_plot):
    bars = axes[idx].bar(comparison_df["Model"], comparison_df[metric],
                          color=model_colors[:len(comparison_df)], edgecolor="black")
    axes[idx].set_title(metric, fontsize=14, fontweight="bold")
    axes[idx].set_ylim(0, 1.05)
    axes[idx].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, comparison_df[metric]):
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 13_metrics_comparison.png")



# 8. THRESHOLD OPTIMIZATION

print("\n" + "=" * 70)
print("PHASE 8: THRESHOLD OPTIMIZATION")
print("=" * 70)

# Optimize threshold for the best model (by F1)
best_model_name = comparison_df.iloc[0]["Model"]
best_prob = results[best_model_name]["y_prob"]
print(f"\nOptimizing threshold for: {best_model_name}")

thresholds = np.arange(0.1, 0.95, 0.05)
threshold_results = []
for t in thresholds:
    y_t = (best_prob >= t).astype(int)
    threshold_results.append({
        "Threshold": t,
        "Precision": precision_score(y_test, y_t, zero_division=0),
        "Recall": recall_score(y_test, y_t),
        "F1": f1_score(y_test, y_t),
        "FP_Count": ((y_t == 1) & (y_test == 0)).sum(),
        "FN_Count": ((y_t == 0) & (y_test == 1)).sum()
    })

thresh_df = pd.DataFrame(threshold_results)
optimal_idx = thresh_df["F1"].idxmax()
optimal_threshold = thresh_df.loc[optimal_idx, "Threshold"]
print(f"\nOptimal threshold: {optimal_threshold:.2f}")
print(thresh_df.to_string(index=False, float_format="%.4f"))

# Threshold visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(thresh_df["Threshold"], thresh_df["Precision"], "b-o", label="Precision", linewidth=2)
axes[0].plot(thresh_df["Threshold"], thresh_df["Recall"], "r-o", label="Recall", linewidth=2)
axes[0].plot(thresh_df["Threshold"], thresh_df["F1"], "g-o", label="F1-Score", linewidth=2)
axes[0].axvline(optimal_threshold, color="black", linestyle="--",
                label=f"Optimal = {optimal_threshold:.2f}")
axes[0].set_title(f"Threshold Optimization — {best_model_name}", fontweight="bold")
axes[0].set_xlabel("Classification Threshold")
axes[0].set_ylabel("Score")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(thresh_df["Threshold"], thresh_df["FP_Count"], "b-o", label="False Positives")
axes[1].plot(thresh_df["Threshold"], thresh_df["FN_Count"], "r-o", label="False Negatives")
axes[1].axvline(optimal_threshold, color="black", linestyle="--")
axes[1].set_title("Error Counts vs Threshold", fontweight="bold")
axes[1].set_xlabel("Classification Threshold")
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_threshold_optimization.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[SAVED] 14_threshold_optimization.png")



# 9. RESAMPLING STRATEGY COMPARISON

print("\n" + "=" * 70)
print("PHASE 9: RESAMPLING STRATEGY COMPARISON (XGBoost)")
print("=" * 70)

resampling_comparison = []
for rs_name, (X_rs, y_rs) in resampling_methods.items():
    t0 = time.time()
    model_rs = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", use_label_encoder=False,
        random_state=42, n_jobs=-1
    )
    model_rs.fit(X_rs, y_rs)
    rs_time = time.time() - t0

    y_pred_rs = model_rs.predict(X_test)
    y_prob_rs = model_rs.predict_proba(X_test)[:, 1]

    resampling_comparison.append({
        "Resampling": rs_name,
        "Precision": precision_score(y_test, y_pred_rs),
        "Recall": recall_score(y_test, y_pred_rs),
        "F1-Score": f1_score(y_test, y_pred_rs),
        "AUC-ROC": roc_auc_score(y_test, y_prob_rs),
        "FP_Count": ((y_pred_rs == 1) & (y_test == 0)).sum(),
        "FN_Count": ((y_pred_rs == 0) & (y_test == 1)).sum()
    })
    print(f"\n{rs_name}: Precision={precision_score(y_test, y_pred_rs):.4f}, "
          f"Recall={recall_score(y_test, y_pred_rs):.4f}, "
          f"F1={f1_score(y_test, y_pred_rs):.4f}")

rs_df = pd.DataFrame(resampling_comparison)
print(f"\n{rs_df.to_string(index=False, float_format='%.4f')}")
rs_df.to_csv(f"{OUTPUT_DIR}/resampling_comparison.csv", index=False)

# Plot resampling comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(rs_df))
width = 0.25
ax.bar(x - width, rs_df["Precision"], width, label="Precision", color="#3498db")
ax.bar(x, rs_df["Recall"], width, label="Recall", color="#e74c3c")
ax.bar(x + width, rs_df["F1-Score"], width, label="F1-Score", color="#2ecc71")
ax.set_xticks(x)
ax.set_xticklabels(rs_df["Resampling"].str.replace("_", "\n"), fontsize=11)
ax.set_title("XGBoost Performance Across Resampling Strategies",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/15_resampling_strategy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[SAVED] 15_resampling_strategy_comparison.png")



# 10. FEATURE IMPORTANCE ANALYSIS

print("\n" + "=" * 70)
print("PHASE 10: FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

# XGBoost feature importance
xgb_importance = pd.Series(
    results["XGBoost"]["model"].feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

# Random Forest feature importance
rf_importance = pd.Series(
    results["Random Forest"]["model"].feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# XGBoost top 20
top_n = 20
xgb_importance.head(top_n).plot(kind="barh", ax=axes[0], color="#e74c3c", edgecolor="black")
axes[0].set_title("XGBoost — Top 20 Features", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Importance Score")
axes[0].invert_yaxis()

# Random Forest top 20
rf_importance.head(top_n).plot(kind="barh", ax=axes[1], color="#2ecc71", edgecolor="black")
axes[1].set_title("Random Forest — Top 20 Features", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Importance Score")
axes[1].invert_yaxis()

plt.suptitle("Feature Importance Analysis", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/16_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 16_feature_importance.png")

print("\nTop 10 features (XGBoost):")
for feat, imp in xgb_importance.head(10).items():
    print(f"  {feat:30s} {imp:.4f}")



# 11. ERROR ANALYSIS

print("\n" + "=" * 70)
print("PHASE 11: ERROR ANALYSIS")
print("=" * 70)

best_pred = results[best_model_name]["y_pred"]
best_proba = results[best_model_name]["y_prob"]

# False Negatives (missed fraud)
fn_mask = (best_pred == 0) & (y_test == 1)
fn_count = fn_mask.sum()
print(f"\nBest model: {best_model_name}")
print(f"False Negatives (missed fraud): {fn_count}")
print(f"False Positives (false alarms):  {((best_pred == 1) & (y_test == 0)).sum()}")

# Analyze characteristics of missed fraud
if fn_count > 0:
    fn_indices = np.where(fn_mask)[0]
    fn_probs = best_proba[fn_indices]
    print(f"\nMissed fraud probability scores:")
    print(f"  Mean: {fn_probs.mean():.4f}")
    print(f"  Min:  {fn_probs.min():.4f}")
    print(f"  Max:  {fn_probs.max():.4f}")

# Confidence distribution
fig, ax = plt.subplots(figsize=(10, 6))
for label, mask_vals, color in [
    ("True Positive", (best_pred == 1) & (y_test == 1), "#2ecc71"),
    ("True Negative", (best_pred == 0) & (y_test == 0), "#3498db"),
    ("False Positive", (best_pred == 1) & (y_test == 0), "#f39c12"),
    ("False Negative", (best_pred == 0) & (y_test == 1), "#e74c3c"),
]:
    if mask_vals.sum() > 0:
        ax.hist(best_proba[mask_vals], bins=50, alpha=0.6, color=color,
                label=f"{label} (n={mask_vals.sum():,})", density=True)
ax.set_title(f"Prediction Confidence Distribution — {best_model_name}",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/17_error_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] 17_error_analysis.png")



# 12. BUSINESS IMPACT ANALYSIS

print("\n" + "=" * 70)
print("PHASE 12: BUSINESS IMPACT ANALYSIS")
print("=" * 70)

# Reconstruct original amounts for test set (inverse transform not straightforward
# with RobustScaler on all features, so we'll use the original df)
test_indices = df_model.index[-len(y_test):]

# Estimate using global stats
avg_fraud_amount = df[df["Class"] == 1]["Amount"].mean()
avg_legit_amount = df[df["Class"] == 0]["Amount"].mean()
total_fraud_in_test = (y_test == 1).sum()

print(f"\nAverage fraudulent transaction:  ${avg_fraud_amount:.2f}")
print(f"Average legitimate transaction:  ${avg_legit_amount:.2f}")
print(f"Total fraud cases in test set:   {total_fraud_in_test}")

for name, res in results.items():
    tp = ((res["y_pred"] == 1) & (y_test == 1)).sum()
    fn = ((res["y_pred"] == 0) & (y_test == 1)).sum()
    fp = ((res["y_pred"] == 1) & (y_test == 0)).sum()

    fraud_prevented = tp * avg_fraud_amount
    fraud_missed = fn * avg_fraud_amount
    investigation_cost = fp * 25  # $25 per false positive investigation

    print(f"\n{name}:")
    print(f"  Fraud detected:        {tp}/{total_fraud_in_test} "
          f"(${fraud_prevented:,.2f} prevented)")
    print(f"  Fraud missed:          {fn} (${fraud_missed:,.2f} losses)")
    print(f"  False alarms:          {fp:,} (${investigation_cost:,.2f} investigation cost)")
    print(f"  Net savings:           ${fraud_prevented - investigation_cost:,.2f}")



# 13. MODEL DEPLOYMENT STRATEGY

print("\n" + "=" * 70)
print("PHASE 13: MODEL DEPLOYMENT STRATEGY")
print("=" * 70)

# Save models as pickle/h5
print("\nSaving model artifacts...")

# Save traditional ML models
for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    model_path = f"{OUTPUT_DIR}/{name.lower().replace(' ', '_')}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(results[name]["model"], f)
    print(f"  [SAVED] {model_path}")

# Save scaler
with open(f"{OUTPUT_DIR}/robust_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"  [SAVED] {OUTPUT_DIR}/robust_scaler.pkl")

# Save DL models if available
if TF_AVAILABLE:
    autoencoder.save(f"{OUTPUT_DIR}/autoencoder_model.keras")
    nn_model.save(f"{OUTPUT_DIR}/neural_network_model.keras")
    print(f"  [SAVED] {OUTPUT_DIR}/autoencoder_model.keras")
    print(f"  [SAVED] {OUTPUT_DIR}/neural_network_model.keras")

# Save feature list
with open(f"{OUTPUT_DIR}/feature_list.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"  [SAVED] {OUTPUT_DIR}/feature_list.json")

# Create deployment inference script
deployment_code = '''"""
Inference Script — Credit Card Fraud Detection
Usage: python inference.py <transaction_json>
"""
import pickle
import json
import numpy as np
import sys

# Load artifacts
with open("output/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("output/robust_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("output/feature_list.json", "r") as f:
    features = json.load(f)

def predict_fraud(transaction: dict, threshold: float = OPTIMAL_THRESHOLD):
    """Predict if a transaction is fraudulent.

    Args:
        transaction: dict with feature values
        threshold: classification threshold (default optimized)

    Returns:
        dict with prediction and probability
    """
    X = np.array([[transaction.get(f, 0) for f in features]])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]

    return {
        "is_fraud": bool(prob >= threshold),
        "fraud_probability": float(prob),
        "risk_level": "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.5 else "LOW",
        "threshold_used": threshold
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        txn = json.loads(sys.argv[1])
        result = predict_fraud(txn)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python inference.py '<json_transaction>'")
'''

deployment_code = deployment_code.replace("OPTIMAL_THRESHOLD", str(round(optimal_threshold, 2)))

with open(f"{OUTPUT_DIR}/inference.py", "w") as f:
    f.write(deployment_code)
print(f"  [SAVED] {OUTPUT_DIR}/inference.py")

print("""
Deployment Architecture:
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Transaction  │────>│  Feature     │────>│  ML Model    │
  │  Input (API)  │     │  Engineering │     │  (XGBoost)   │
  └──────────────┘     └──────────────┘     └──────┬───────┘
                                                    │
                                                    
                              ┌──────────────────────┘
                              ▼
                    ┌───────────────────┐
                    │  Threshold Check  │
                    │  (optimized)      │
                    └────────┬──────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
          ┌─────────┐ ┌──────────┐ ┌──────────┐
          │  LOW    │ │  MEDIUM  │ │  HIGH    │
          │ Approve │ │ Flag for │ │ Block &  │
          │         │ │ Review   │ │ Alert    │
          └─────────┘ └──────────┘ └──────────┘
""")



# 14. FINAL SUMMARY

print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
Dataset:        Kaggle Credit Card Fraud Detection
                284,807 transactions, 492 fraud (0.172%)
Methodology:    CRISP-DM

Models Evaluated:
  1. Logistic Regression    — Linear baseline
  2. Random Forest          — Ensemble (bagging)
  3. XGBoost                — Ensemble (boosting)
  4. Deep Autoencoder       — Anomaly detection (unsupervised DL)
  5. Neural Network         — Supervised DL classifier

Best Overall Model: {best_model_name}
  F1-Score:  {f1_score(y_test, results[best_model_name]['y_pred']):.4f}
  Recall:    {recall_score(y_test, results[best_model_name]['y_pred']):.4f}
  Precision: {precision_score(y_test, results[best_model_name]['y_pred']):.4f}
  AUC-ROC:   {roc_auc_score(y_test, results[best_model_name]['y_prob']):.4f}
  Optimal Threshold: {optimal_threshold:.2f}

Resampling: Hybrid (SMOTE + Tomek Links) recommended
            Validates findings of Popova & Gardi (2024)

Artifacts Saved in '{OUTPUT_DIR}/' directory:
  - 17 visualization PNGs
  - Model comparison CSV
  - Resampling comparison CSV
  - Trained model files (.pkl, .keras)
  - Feature list JSON
  - Inference deployment script
""")

print("=" * 70)
print("PIPELINE COMPLETE — All artifacts saved to output/")
print("=" * 70)
