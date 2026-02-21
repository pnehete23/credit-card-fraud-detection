import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("CREDIT CARD FRAUD DETECTION - MODEL PERFORMANCE ANALYSIS")
print("="*70)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Dataset: Kaggle Credit Card Fraud Dataset")
print("="*70)
print()

# Load the data
df = pd.read_csv('credit-card-fraud-detection/output/model_comparison.csv')

print("1. DATA OVERVIEW")
print("-" * 70)
print(df.to_string(index=False))
print()

# Statistical Summary
print("2. STATISTICAL SUMMARY")
print("-" * 70)
print(df.describe().to_string())
print()

# Performance Rankings
print("3. MODEL RANKINGS BY METRIC")
print("-" * 70)

metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR', 'MCC']
for metric in metrics:
    ranked = df.nlargest(5, metric)[['Model', metric]]
    print(f"\n{metric}:")
    for idx, row in ranked.iterrows():
        print(f"  {idx+1}. {row['Model']:<20} {row[metric]:.4f}")

print(f"\nTraining Time (Fastest to Slowest):")
ranked_time = df.nsmallest(5, 'Train Time (s)')[['Model', 'Train Time (s)']]
for idx, row in ranked_time.iterrows():
    print(f"  {idx+1}. {row['Model']:<20} {row['Train Time (s)']:.2f}s")
print()

# Best Model Analysis
print("4. BEST MODEL IDENTIFICATION")
print("-" * 70)

# Calculate composite score (weighted average of key metrics)
weights = {
    'F1-Score': 0.30,
    'AUC-ROC': 0.25,
    'AUC-PR': 0.20,
    'MCC': 0.15,
    'Precision': 0.10
}

df['Composite_Score'] = (
    df['F1-Score'] * weights['F1-Score'] +
    df['AUC-ROC'] * weights['AUC-ROC'] +
    df['AUC-PR'] * weights['AUC-PR'] +
    df['MCC'] * weights['MCC'] +
    df['Precision'] * weights['Precision']
)

best_model = df.loc[df['Composite_Score'].idxmax()]
print(f"Best Overall Model: {best_model['Model']}")
print(f"Composite Score: {best_model['Composite_Score']:.4f}")
print()
print("Performance Breakdown:")
print(f"  Precision:     {best_model['Precision']:.4f}")
print(f"  Recall:        {best_model['Recall']:.4f}")
print(f"  F1-Score:      {best_model['F1-Score']:.4f}")
print(f"  AUC-ROC:       {best_model['AUC-ROC']:.4f}")
print(f"  AUC-PR:        {best_model['AUC-PR']:.4f}")
print(f"  MCC:           {best_model['MCC']:.4f}")
print(f"  Training Time: {best_model['Train Time (s)']:.2f}s")
print()

# Efficiency Analysis
print("5. EFFICIENCY ANALYSIS")
print("-" * 70)
df['Efficiency_Ratio'] = df['F1-Score'] / (df['Train Time (s)'] / 60)  # per minute
df_efficiency = df[['Model', 'Train Time (s)', 'F1-Score', 'Efficiency_Ratio']].sort_values('Efficiency_Ratio', ascending=False)
print(df_efficiency.to_string(index=False))
print()
print("Note: Efficiency Ratio = F1-Score / Training Time (minutes)")
print("Higher values indicate better performance relative to training time")
print()

# Precision-Recall Analysis
print("6. PRECISION-RECALL TRADE-OFF ANALYSIS")
print("-" * 70)
df['PR_Balance'] = 2 * (df['Precision'] * df['Recall']) / (df['Precision'] + df['Recall'])
df_pr = df[['Model', 'Precision', 'Recall', 'F1-Score', 'PR_Balance']].sort_values('F1-Score', ascending=False)
print(df_pr.to_string(index=False))
print()
print("Analysis:")
for idx, row in df.iterrows():
    if row['Precision'] > 0.8 and row['Recall'] > 0.7:
        print(f"  ✓ {row['Model']}: Excellent balance (High Precision & High Recall)")
    elif row['Recall'] > 0.8 and row['Precision'] < 0.1:
        print(f"  ✗ {row['Model']}: Poor precision - too many false positives")
    elif row['Precision'] > 0.6 and row['Recall'] < 0.6:
        print(f"  ⚠ {row['Model']}: Moderate performance - room for improvement")
print()

# AUC Comparison
print("7. AUC METRICS COMPARISON")
print("-" * 70)
df_auc = df[['Model', 'AUC-ROC', 'AUC-PR']].copy()
df_auc['AUC_Gap'] = df_auc['AUC-ROC'] - df_auc['AUC-PR']
df_auc = df_auc.sort_values('AUC-ROC', ascending=False)
print(df_auc.to_string(index=False))
print()
print("Interpretation:")
print("  - AUC-ROC: Overall model discrimination ability")
print("  - AUC-PR: Performance on imbalanced data (more relevant for fraud)")
print("  - Gap: Larger gaps suggest challenges with class imbalance")
print()

# Performance vs Training Time
print("8. PERFORMANCE VS TRAINING TIME ANALYSIS")
print("-" * 70)
df_perf = df[['Model', 'F1-Score', 'Train Time (s)']].sort_values('F1-Score', ascending=False)
print(df_perf.to_string(index=False))
print()
fastest = df.loc[df['Train Time (s)'].idxmin()]
slowest = df.loc[df['Train Time (s)'].idxmax()]
print(f"Fastest: {fastest['Model']} ({fastest['Train Time (s)']:.2f}s)")
print(f"Slowest: {slowest['Model']} ({slowest['Train Time (s)']:.2f}s)")
print(f"Speed Difference: {slowest['Train Time (s)'] / fastest['Train Time (s)']:.1f}x")
print()

# Key Findings
print("9. KEY FINDINGS & RECOMMENDATIONS")
print("=" * 70)

findings = [
    ("Best Overall Model", 
     f"{best_model['Model']} with F1-Score of {best_model['F1-Score']:.4f} and AUC-ROC of {best_model['AUC-ROC']:.4f}"),
    
    ("Fastest Training", 
     f"{fastest['Model']} completes training in just {fastest['Train Time (s)']:.2f} seconds"),
    
    ("Best Recall", 
     f"{df.loc[df['Recall'].idxmax(), 'Model']} achieves highest recall ({df['Recall'].max():.4f}), "
     f"but with very low precision ({df.loc[df['Recall'].idxmax(), 'Precision']:.4f})"),
    
    ("Most Balanced", 
     f"{df.loc[df['PR_Balance'].idxmax(), 'Model']} shows best precision-recall balance"),
    
    ("Production Recommendation", 
     f"{best_model['Model']} is recommended for production deployment due to:\n"
     f"     - Highest overall performance metrics\n"
     f"     - Fast training time ({best_model['Train Time (s)']:.2f}s)\n"
     f"     - Excellent balance between precision and recall\n"
     f"     - Best AUC-ROC score ({best_model['AUC-ROC']:.4f})")
]

for i, (title, finding) in enumerate(findings, 1):
    print(f"\n{i}. {title}:")
    print(f"   {finding}")

print()
print("10. IMPLEMENTATION CONSIDERATIONS")
print("=" * 70)

considerations = [
    "Class Imbalance: All models handle the highly imbalanced fraud dataset differently. "
    "Monitor AUC-PR closely as it's more informative than AUC-ROC for imbalanced data.",
    
    "False Positive Cost: In fraud detection, false positives (legitimate transactions flagged) "
    "can frustrate customers. XGBoost and Random Forest offer the best precision (>0.85).",
    
    "False Negative Cost: Missing actual fraud is costly. While Logistic Regression has highest "
    "recall (0.86), its low precision (0.04) makes it impractical - it would flag 96% incorrectly.",
    
    "Scalability: XGBoost's fast training time (8.8s) vs Random Forest (141.6s) is crucial for "
    "frequent model retraining as fraud patterns evolve.",
    
    "Interpretability: While Neural Networks and Autoencoders offer deep learning capabilities, "
    "tree-based models (XGBoost, Random Forest) provide better interpretability for regulatory compliance.",
    
    "Real-time Scoring: All models can perform real-time inference, but XGBoost typically offers "
    "the best latency-performance trade-off for production systems.",
    
    f"Recommended Threshold Tuning: Start with {best_model['Model']} and adjust the classification "
    "threshold based on the specific cost ratio of false positives to false negatives in your use case."
]

for i, consideration in enumerate(considerations, 1):
    print(f"\n{i}. {consideration}")

print()
print("="*70)
print("END OF ANALYSIS")
print("="*70)
print()
print("For interactive visualizations, please open the HTML dashboard.")
print()

# Save summary statistics
summary_stats = {
    'Best Model': best_model['Model'],
    'Best F1-Score': best_model['F1-Score'],
    'Best AUC-ROC': best_model['AUC-ROC'],
    'Fastest Training': fastest['Model'],
    'Training Time Range': f"{df['Train Time (s)'].min():.2f}s - {df['Train Time (s)'].max():.2f}s",
    'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

print("SUMMARY STATISTICS")
print("-" * 70)
for key, value in summary_stats.items():
    print(f"{key}: {value}")
