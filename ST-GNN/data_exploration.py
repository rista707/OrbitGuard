import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load dataset
print("\n[1] Loading dataset...")
df = pd.read_csv('/home/ubuntu/ubuntu/dataset_crosslayer_step12.csv')
print(f"Dataset shape: {df.shape}")
print(f"Total samples: {len(df):,}")

# Display column names
print("\n[2] Dataset columns:")
print(df.columns.tolist())

# Display first few rows
print("\n[3] First 5 rows:")
print(df.head())

# Data types
print("\n[4] Data types:")
print(df.dtypes)

# Missing values
print("\n[5] Missing values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")

# Basic statistics
print("\n[6] Basic statistics:")
print(df.describe())

# Attack class distribution
attack_counts = df['attack'].value_counts()
print("\nAbsolute counts:")
print(attack_counts)
print("\nPercentage distribution:")
print(df['attack'].value_counts(normalize=True) * 100)

# Attack phase distribution
phase_counts = df['attack_phase'].value_counts()
print("\nAbsolute counts:")
print(phase_counts)
print("\nPercentage distribution:")
print(df['attack_phase'].value_counts(normalize=True) * 100)

# Cross-tabulation: attack vs phase
crosstab = pd.crosstab(df['attack'], df['attack_phase'], margins=True)
print(crosstab)

# Attack target distribution
if 'attack_target' in df.columns:
    target_counts = df['attack_target'].value_counts()
    print(f"\nUnique targets: {df['attack_target'].nunique()}")
    print(f"\nTop 10 targets:")
    print(target_counts.head(10))

# Feature groups analysis

flow_features = ['burst_id', 'src', 'dst', 't_start_s', 'window_s', 
                 'bytes_sent', 'bytes_recv', 'loss_ratio', 'throughput_Mbps']
routing_features = ['path_len', 'includes_s_bad', 'path_changed']
isl_features = ['isl_util_mean_path', 'isl_util_max_path', 'isl_util_std_path',
                'isl_delta_mean_path', 'num_hot_links', 'frac_hot_links', 
                'path_congestion_score']
reroute_features = ['reroute_pressure']

print(f"\nFlow-level features ({len([f for f in flow_features if f in df.columns])} found): {[f for f in flow_features if f in df.columns]}")
print(f"\nRouting features ({len([f for f in routing_features if f in df.columns])} found): {[f for f in routing_features if f in df.columns]}")
print(f"\nISL features ({len([f for f in isl_features if f in df.columns])} found): {[f for f in isl_features if f in df.columns]}")
print(f"\nReroute features ({len([f for f in reroute_features if f in df.columns])} found): {[f for f in reroute_features if f in df.columns]}")

# Calculate class imbalance ratio
attack_counts = df['attack'].value_counts()
max_class = attack_counts.max()
min_class = attack_counts.min()
imbalance_ratio = max_class / min_class
print(f"\nMajority class: {attack_counts.idxmax()} with {max_class:,} samples")
print(f"Minority class: {attack_counts.idxmin()} with {min_class:,} samples")
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

# Visualizations
print("\n[13] Generating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Attack class distribution
ax1 = plt.subplot(2, 3, 1)
attack_counts.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
ax1.set_title('Attack Class Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Attack Type', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(attack_counts.values):
    ax1.text(i, v + max(attack_counts.values)*0.01, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Attack phase distribution
ax2 = plt.subplot(2, 3, 2)
phase_counts.plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
ax2.set_title('Attack Phase Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Phase', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(phase_counts.values):
    ax2.text(i, v + max(phase_counts.values)*0.01, str(v), ha='center', va='bottom', fontweight='bold')

# 3. Attack class pie chart
ax3 = plt.subplot(2, 3, 3)
colors = sns.color_palette("husl", len(attack_counts))
wedges, texts, autotexts = ax3.pie(attack_counts.values, labels=attack_counts.index, 
                                     autopct='%1.1f%%', startangle=90, colors=colors)
ax3.set_title('Attack Class Percentage', fontsize=14, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 4. Throughput distribution by attack
ax4 = plt.subplot(2, 3, 4)
df.boxplot(column='throughput_Mbps', by='attack', ax=ax4)
ax4.set_title('Throughput by Attack Type', fontsize=14, fontweight='bold')
ax4.set_xlabel('Attack Type', fontsize=12)
ax4.set_ylabel('Throughput (Mbps)', fontsize=12)
ax4.tick_params(axis='x', rotation=45)
plt.suptitle('')

# 5. Loss ratio distribution by attack
ax5 = plt.subplot(2, 3, 5)
df.boxplot(column='loss_ratio', by='attack', ax=ax5)
ax5.set_title('Loss Ratio by Attack Type', fontsize=14, fontweight='bold')
ax5.set_xlabel('Attack Type', fontsize=12)
ax5.set_ylabel('Loss Ratio', fontsize=12)
ax5.tick_params(axis='x', rotation=45)
plt.suptitle('')

# 6. Path length distribution by attack
ax6 = plt.subplot(2, 3, 6)
df.boxplot(column='path_len', by='attack', ax=ax6)
ax6.set_title('Path Length by Attack Type', fontsize=14, fontweight='bold')
ax6.set_xlabel('Attack Type', fontsize=12)
ax6.set_ylabel('Path Length', fontsize=12)
ax6.tick_params(axis='x', rotation=45)
plt.suptitle('')

plt.tight_layout()
plt.savefig('/home/ubuntu/orbitguard/data_exploration.png', dpi=300, bbox_inches='tight')
print("✓ Saved: data_exploration.png")

# Correlation heatmap
print("\n[14] Generating correlation heatmap...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove identifier columns
numeric_cols = [c for c in numeric_cols if c not in ['burst_id', 'src', 'dst', 't_start_s']]

if len(numeric_cols) > 0:
    fig2, ax = plt.subplots(figsize=(16, 14))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/orbitguard/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_heatmap.png")

# Per-class feature statistics
print("\n[15] Per-class feature statistics:")
print("=" * 80)
for attack_type in df['attack'].unique():
    print(f"\n{attack_type}:")
    subset = df[df['attack'] == attack_type]
    print(f"  Samples: {len(subset):,}")
    print(f"  Avg throughput: {subset['throughput_Mbps'].mean():.4f} Mbps")
    print(f"  Avg loss_ratio: {subset['loss_ratio'].mean():.4f}")
    print(f"  Avg path_len: {subset['path_len'].mean():.2f}")
    if 'includes_s_bad' in subset.columns:
        print(f"  Includes bad satellite: {subset['includes_s_bad'].mean():.2%}")
    if 'path_changed' in subset.columns:
        print(f"  Path changed: {subset['path_changed'].mean():.2%}")

# Save summary statistics
print("\n[16] Saving summary statistics...")
summary = {
    'total_samples': len(df),
    'num_features': len(df.columns),
    'attack_classes': df['attack'].nunique(),
    'class_distribution': df['attack'].value_counts().to_dict(),
    'imbalance_ratio': float(imbalance_ratio),
    'majority_class': attack_counts.idxmax(),
    'minority_class': attack_counts.idxmin()
}

import json
with open('/home/ubuntu/orbitguard/data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved: data_summary.json")

print("\n" + "=" * 80)
print("DATA EXPLORATION COMPLETE!")
print("=" * 80)
