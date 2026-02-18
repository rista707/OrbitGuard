#!/usr/bin/env python3
"""
Generate ROC-AUC Curves for Multi-class Classification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pickle
import torch
from itertools import cycle

# Set high-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

print("=" * 80)
print("GENERATING ROC-AUC CURVES (300 DPI)")
print("=" * 80)

# Load results
print("\n[1] Loading model results...")
with open('/home/ubuntu/leo_attack_detection/results.pkl', 'rb') as f:
    results = pickle.load(f)

with open('/home/ubuntu/leo_attack_detection/processed_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

label_encoder = data_dict['label_encoder']
n_classes = len(label_encoder.classes_)

# Get predictions and true labels
y_true = results['y_true']
y_pred_proba = results['y_pred_proba']

print(f"✓ Loaded {len(y_true)} test samples")
print(f"✓ Number of classes: {n_classes}")
print(f"✓ Classes: {label_encoder.classes_}")

# Binarize the labels for multi-class ROC
y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

print("\n[2] Computing ROC curves...")

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f"✓ Class {label_encoder.classes_[i]}: AUC = {roc_auc[i]:.4f}")

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(f"✓ Micro-average AUC: {roc_auc['micro']:.4f}")

# Compute macro-average ROC curve and ROC area
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
print(f"✓ Macro-average AUC: {roc_auc['macro']:.4f}")

# Create output directory
import os
os.makedirs('/home/ubuntu/leo_attack_detection/high_quality_plots', exist_ok=True)

print("\n[3] Generating ROC curve plots...")

# ============================================================================
# Plot 1: All Classes in One Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curve for each class
colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
for i, color in zip(range(n_classes), colors):
    ax.plot(fpr[i], tpr[i], color=color, lw=2.5,
            label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.4f})')

# Plot micro-average
ax.plot(fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})',
        color='deeppink', linestyle='--', linewidth=2.5)

# Plot macro-average
ax.plot(fpr["macro"], tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})',
        color='navy', linestyle='--', linewidth=2.5)

# Plot diagonal (random classifier)
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curves - All Classes', fontweight='bold', fontsize=18)
ax.legend(loc="lower right", frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/16_roc_all_classes.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 16_roc_all_classes.png")

# ============================================================================
# Plot 2-5: Individual ROC Curves for Each Class
# ============================================================================

for i in range(n_classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot this class
    ax.plot(fpr[i], tpr[i], color='steelblue', lw=3,
            label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.4f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')
    
    # Fill area under curve
    ax.fill_between(fpr[i], tpr[i], alpha=0.2, color='steelblue')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(f'ROC Curve - {label_encoder.classes_[i]} Attack', 
                 fontweight='bold', fontsize=18)
    ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add AUC text box
    textstr = f'AUC = {roc_auc[i]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.6, 0.2, textstr, transform=ax.transAxes, fontsize=16,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/leo_attack_detection/high_quality_plots/{17+i:02d}_roc_{label_encoder.classes_[i].lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {17+i:02d}_roc_{label_encoder.classes_[i].lower()}.png")

# ============================================================================
# Plot 6: Micro vs Macro Average
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Plot micro-average
ax.plot(fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})',
        color='deeppink', linewidth=3)

# Plot macro-average
ax.plot(fpr["macro"], tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})',
        color='navy', linewidth=3)

# Plot diagonal
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier')

# Fill areas
ax.fill_between(fpr["micro"], tpr["micro"], alpha=0.2, color='deeppink')
ax.fill_between(fpr["macro"], tpr["macro"], alpha=0.2, color='navy')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontweight='bold')
ax.set_ylabel('True Positive Rate', fontweight='bold')
ax.set_title('ROC Curves - Micro vs Macro Average', fontweight='bold', fontsize=18)
ax.legend(loc="lower right", frameon=True, shadow=True, fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/21_roc_micro_macro.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 21_roc_micro_macro.png")

# ============================================================================
# Summary Table
# ============================================================================

print("\n" + "=" * 80)
print("ROC-AUC SUMMARY")
print("=" * 80)
print(f"\n{'Class':<15} {'AUC Score':>12}")
print("-" * 30)
for i in range(n_classes):
    print(f"{label_encoder.classes_[i]:<15} {roc_auc[i]:>12.4f}")
print("-" * 30)
print(f"{'Micro-average':<15} {roc_auc['micro']:>12.4f}")
print(f"{'Macro-average':<15} {roc_auc['macro']:>12.4f}")
print("=" * 80)

print("\n✅ All ROC-AUC curves generated successfully!")
print(f"Location: /home/ubuntu/leo_attack_detection/high_quality_plots/")
print(f"Total ROC plots: 6")
print("\nPlot List:")
print("  16. ROC Curves - All Classes (combined)")
print("  17. ROC Curve - Baseline Attack")
print("  18. ROC Curve - Blackhole Attack")
print("  19. ROC Curve - DDoS Attack")
print("  20. ROC Curve - Sinkhole Attack")
print("  21. ROC Curves - Micro vs Macro Average")
print("=" * 80)
