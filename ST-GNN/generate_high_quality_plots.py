#!/usr/bin/env python3
"""
Generate High-Quality Individual Plots (300 DPI)
Each metric/analysis in a separate image file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set high-quality plotting defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

print("=" * 80)
print("GENERATING HIGH-QUALITY INDIVIDUAL PLOTS (300 DPI)")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('/home/ubuntu/upload/dataset_crosslayer_step12.csv')
with open('/home/ubuntu/leo_attack_detection/results.pkl', 'rb') as f:
    results = pickle.load(f)
with open('/home/ubuntu/leo_attack_detection/processed_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

label_encoder = data_dict['label_encoder']
feature_cols = data_dict['feature_cols']

print("✓ Data loaded")

# Create output directory
import os
os.makedirs('/home/ubuntu/leo_attack_detection/high_quality_plots', exist_ok=True)

# ============================================================================
# DATASET EXPLORATION PLOTS
# ============================================================================

print("\n[2] Generating dataset exploration plots...")

# Plot 1: Attack Class Distribution
fig, ax = plt.subplots(figsize=(10, 6))
class_counts = df['attack'].value_counts()
bars = ax.bar(class_counts.index, class_counts.values, color='steelblue', alpha=0.8, edgecolor='black')
ax.set_xlabel('Attack Type', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Attack Class Distribution', fontweight='bold', fontsize=18)
ax.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/01_attack_class_distribution.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_attack_class_distribution.png")

# Plot 2: Attack Phase Distribution
fig, ax = plt.subplots(figsize=(10, 6))
phase_counts = df['attack_phase'].value_counts()
bars = ax.bar(phase_counts.index, phase_counts.values, color='coral', alpha=0.8, edgecolor='black')
ax.set_xlabel('Attack Phase', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Attack Phase Distribution', fontweight='bold', fontsize=18)
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/02_attack_phase_distribution.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_attack_phase_distribution.png")

# Plot 3: Attack Class Percentage (Pie Chart)
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.05, 0, 0, 0)  # Explode baseline slice
wedges, texts, autotexts = ax.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                                    colors=colors, explode=explode, startangle=90, textprops={'fontsize': 14})
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
ax.set_title('Attack Class Percentage', fontweight='bold', fontsize=18)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/03_attack_class_percentage.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_attack_class_percentage.png")

# Plot 4: Throughput by Attack Type
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='throughput_Mbps', by='attack', ax=ax, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7),
           medianprops=dict(color='red', linewidth=2))
ax.set_xlabel('Attack Type', fontweight='bold')
ax.set_ylabel('Throughput (Mbps)', fontweight='bold')
ax.set_title('Throughput by Attack Type', fontweight='bold', fontsize=18)
plt.suptitle('')  # Remove default title
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/04_throughput_by_attack.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_throughput_by_attack.png")

# Plot 5: Loss Ratio by Attack Type
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='loss_ratio', by='attack', ax=ax, patch_artist=True,
           boxprops=dict(facecolor='lightcoral', alpha=0.7),
           medianprops=dict(color='darkred', linewidth=2))
ax.set_xlabel('Attack Type', fontweight='bold')
ax.set_ylabel('Loss Ratio', fontweight='bold')
ax.set_title('Loss Ratio by Attack Type', fontweight='bold', fontsize=18)
plt.suptitle('')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/05_loss_ratio_by_attack.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_loss_ratio_by_attack.png")

# Plot 6: Path Length by Attack Type
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='path_len', by='attack', ax=ax, patch_artist=True,
           boxprops=dict(facecolor='lightgreen', alpha=0.7),
           medianprops=dict(color='darkgreen', linewidth=2))
ax.set_xlabel('Attack Type', fontweight='bold')
ax.set_ylabel('Path Length (Hops)', fontweight='bold')
ax.set_title('Path Length by Attack Type', fontweight='bold', fontsize=18)
plt.suptitle('')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/06_path_length_by_attack.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_path_length_by_attack.png")

# ============================================================================
# FEATURE CORRELATION
# ============================================================================

print("\n[3] Generating feature correlation heatmap...")

# Use only features available in raw dataset
available_features = [col for col in feature_cols if col in df.columns]
print(f"Using {len(available_features)} available features for correlation")

fig, ax = plt.subplots(figsize=(14, 12))
correlation_matrix = df[available_features].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=18, pad=20)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/07_feature_correlation_heatmap.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_feature_correlation_heatmap.png")

# ============================================================================
# TRAINING RESULTS
# ============================================================================

print("\n[4] Generating training result plots...")

history = results['history']

# Plot 7: Training Loss
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history['train_loss'], label='Training Loss', linewidth=2.5, color='#1f77b4', marker='o', markersize=4)
ax.plot(history['val_loss'], label='Validation Loss', linewidth=2.5, color='#ff7f0e', marker='s', markersize=4)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Loss', fontweight='bold')
ax.set_title('Training and Validation Loss', fontweight='bold', fontsize=18)
ax.legend(loc='upper right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/08_training_loss.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 08_training_loss.png")

# Plot 8: Training Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history['train_acc'], label='Training Accuracy', linewidth=2.5, color='#2ca02c', marker='o', markersize=4)
ax.plot(history['val_acc'], label='Validation Accuracy', linewidth=2.5, color='#d62728', marker='s', markersize=4)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Accuracy', fontweight='bold')
ax.set_title('Training and Validation Accuracy', fontweight='bold', fontsize=18)
ax.legend(loc='lower right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.7, 1.0])
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/09_training_accuracy.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 09_training_accuracy.png")

# Plot 9: F1 Scores
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history['val_f1_macro'], label='F1-Macro', linewidth=2.5, color='#9467bd', marker='o', markersize=4)
ax.plot(history['val_f1_weighted'], label='F1-Weighted', linewidth=2.5, color='#8c564b', marker='s', markersize=4)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_title('F1 Score Evolution', fontweight='bold', fontsize=18)
ax.legend(loc='lower right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.7, 1.0])
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/10_f1_scores.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 10_f1_scores.png")

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

print("\n[5] Generating confusion matrix...")

fig, ax = plt.subplots(figsize=(10, 8))
cm = results['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
            cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=14)
ax.set_ylabel('True Label', fontweight='bold', fontsize=14)
ax.set_title('Confusion Matrix (Test Set)', fontweight='bold', fontsize=18, pad=20)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/11_confusion_matrix.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 11_confusion_matrix.png")

# ============================================================================
# PER-CLASS PERFORMANCE
# ============================================================================

print("\n[6] Generating per-class performance plots...")

precision = results['per_class_metrics']['precision']
recall = results['per_class_metrics']['recall']
f1 = results['per_class_metrics']['f1']

# Plot 11: Precision by Class
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(label_encoder.classes_))
bars = ax.bar(x, precision, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Attack Class', fontweight='bold')
ax.set_ylabel('Precision', fontweight='bold')
ax.set_title('Precision by Attack Class', fontweight='bold', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(label_encoder.classes_, rotation=0)
ax.set_ylim([0.9, 1.01])
ax.grid(True, alpha=0.3, axis='y')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{precision[i]:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/12_precision_by_class.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 12_precision_by_class.png")

# Plot 12: Recall by Class
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, recall, color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Attack Class', fontweight='bold')
ax.set_ylabel('Recall', fontweight='bold')
ax.set_title('Recall by Attack Class', fontweight='bold', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(label_encoder.classes_, rotation=0)
ax.set_ylim([0.9, 1.01])
ax.grid(True, alpha=0.3, axis='y')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{recall[i]:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/13_recall_by_class.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 13_recall_by_class.png")

# Plot 13: F1-Score by Class
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, f1, color='lightgreen', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Attack Class', fontweight='bold')
ax.set_ylabel('F1-Score', fontweight='bold')
ax.set_title('F1-Score by Attack Class', fontweight='bold', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(label_encoder.classes_, rotation=0)
ax.set_ylim([0.9, 1.01])
ax.grid(True, alpha=0.3, axis='y')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{f1[i]:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/14_f1_score_by_class.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 14_f1_score_by_class.png")

# Plot 14: All Metrics Comparison (Grouped Bar Chart)
fig, ax = plt.subplots(figsize=(12, 7))
width = 0.25
x = np.arange(len(label_encoder.classes_))
bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='steelblue', edgecolor='black')
bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='coral', edgecolor='black')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightgreen', edgecolor='black')
ax.set_xlabel('Attack Class', fontweight='bold', fontsize=14)
ax.set_ylabel('Score', fontweight='bold', fontsize=14)
ax.set_title('Per-Class Performance Metrics Comparison', fontweight='bold', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(label_encoder.classes_, rotation=0)
ax.legend(loc='lower right', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.9, 1.05])
# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/leo_attack_detection/high_quality_plots/15_all_metrics_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 15_all_metrics_comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("ALL HIGH-QUALITY PLOTS GENERATED!")
print("=" * 80)
print(f"\nLocation: /home/ubuntu/leo_attack_detection/high_quality_plots/")
print(f"Total plots: 15")
print(f"Resolution: 300 DPI (publication quality)")
print("\nPlot List:")
print("  01. Attack Class Distribution")
print("  02. Attack Phase Distribution")
print("  03. Attack Class Percentage (Pie Chart)")
print("  04. Throughput by Attack Type")
print("  05. Loss Ratio by Attack Type")
print("  06. Path Length by Attack Type")
print("  07. Feature Correlation Heatmap")
print("  08. Training Loss")
print("  09. Training Accuracy")
print("  10. F1 Score Evolution")
print("  11. Confusion Matrix")
print("  12. Precision by Class")
print("  13. Recall by Class")
print("  14. F1-Score by Class")
print("  15. All Metrics Comparison")
print("=" * 80)
