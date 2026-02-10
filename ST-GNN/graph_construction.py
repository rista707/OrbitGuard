#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("\n[1] Loading dataset...")
df = pd.read_csv('/home/ubuntu/orbitguard/dataset_crosslayer_step12.csv')
print(f"Dataset shape: {df.shape}")

# Sort by time for temporal ordering
df = df.sort_values(['run_name', 'burst_id', 't_start_s']).reset_index(drop=True)
print(f"Sorted by temporal order")

# Extract unique satellites (nodes)
print("\n[2] Extracting satellite network topology...")
all_nodes = sorted(list(set(df['src'].unique()) | set(df['dst'].unique())))
num_nodes = len(all_nodes)
node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
print(f"Total satellites (nodes): {num_nodes}")
print(f"Node range: {min(all_nodes)} to {max(all_nodes)}")

# Build edge list from flows (src -> dst connections)
print("\n[3] Building edge list from traffic flows...")
edges = df[['src', 'dst']].drop_duplicates()
edge_list = [(node_to_idx[row['src']], node_to_idx[row['dst']]) for _, row in edges.iterrows()]
print(f"Total unique edges: {len(edge_list)}")

# Create adjacency matrix
print("\n[4] Creating adjacency matrix...")
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
for src, dst in edge_list:
    adj_matrix[src, dst] = 1.0
    adj_matrix[dst, src] = 1.0  # Bidirectional ISLs
print(f"Adjacency matrix shape: {adj_matrix.shape}")
print(f"Network density: {adj_matrix.sum() / (num_nodes * num_nodes):.4f}")

# Domain-specific feature engineering
print("\n[5] Domain-specific feature engineering...")

# 5.1 Traffic anomaly features
df['throughput_anomaly'] = (df['throughput_Mbps'] - df.groupby('src')['throughput_Mbps'].transform('mean')) / \
                            (df.groupby('src')['throughput_Mbps'].transform('std') + 1e-6)
df['loss_spike'] = (df['loss_ratio'] > 0.9).astype(int)
df['zero_throughput'] = (df['throughput_Mbps'] == 0).astype(int)

# 5.2 Path-based features
df['path_efficiency'] = 1.0 / (df['path_len'] + 1)  # Shorter paths are more efficient
df['path_anomaly_score'] = df['path_len'] * df['loss_ratio']  # Combined path quality metric
df['routing_instability'] = df['path_changed'].astype(int)

# 5.3 ISL congestion features
df['isl_congestion_level'] = pd.cut(df['isl_util_max_path'], 
                                     bins=[0, 0.3, 0.6, 0.9, 1.0], 
                                     labels=[0, 1, 2, 3]).astype(int)
df['isl_variance_ratio'] = df['isl_util_std_path'] / (df['isl_util_mean_path'] + 1e-6)

# 5.4 Temporal features
df['time_window'] = (df['t_start_s'] / df['window_s']).astype(int)
df['flow_duration_ratio'] = df['window_s'] / (df['t_start_s'] + 1e-6)

# 5.5 Attack-specific indicators
df['bad_satellite_indicator'] = df['includes_s_bad'].astype(int)
df['high_path_len_indicator'] = (df['path_len'] > df['path_len'].quantile(0.75)).astype(int)
df['low_throughput_indicator'] = (df['throughput_Mbps'] < df['throughput_Mbps'].quantile(0.25)).astype(int)

# 5.6 Cross-layer interaction features
df['congestion_loss_interaction'] = df['path_congestion_score'] * df['loss_ratio']
df['path_util_interaction'] = df['path_len'] * df['isl_util_mean_path']

print(f"✓ Created {len(df.columns) - 20} new engineered features")

# Feature list for model
feature_cols = [
    # Flow-level features
    'loss_ratio', 'throughput_Mbps', 'window_s',
    # Routing features
    'path_len', 'includes_s_bad', 'path_changed',
    # ISL features
    'isl_util_mean_path', 'isl_util_max_path', 'isl_util_std_path',
    'isl_delta_mean_path', 'path_congestion_score',
    # Engineered features
    'throughput_anomaly', 'loss_spike', 'zero_throughput',
    'path_efficiency', 'path_anomaly_score', 'routing_instability',
    'isl_congestion_level', 'isl_variance_ratio',
    'bad_satellite_indicator', 'high_path_len_indicator', 'low_throughput_indicator',
    'congestion_loss_interaction', 'path_util_interaction'
]

print(f"\nTotal features for model: {len(feature_cols)}")
print(f"Feature list: {feature_cols}")

# Create temporal sequences
print("\n[6] Creating temporal sequences for ST-GNN...")
sequence_length = 10  # Number of time steps per sequence
stride = 5  # Stride for sliding window

sequences = []
labels = []
metadata = []

# Group by run and burst
for run_name in df['run_name'].unique():
    run_df = df[df['run_name'] == run_name].sort_values('t_start_s')
    
    # Create sequences with sliding window
    for i in range(0, len(run_df) - sequence_length + 1, stride):
        seq_data = run_df.iloc[i:i+sequence_length]
        
        # Extract features
        seq_features = seq_data[feature_cols].values
        
        # Get label (majority vote in sequence)
        seq_label = seq_data['attack'].mode()[0]
        
        # Get metadata
        seq_meta = {
            'run_name': run_name,
            'start_idx': i,
            'end_idx': i + sequence_length,
            'src_nodes': seq_data['src'].values,
            'dst_nodes': seq_data['dst'].values,
            'attack_phase': seq_data['attack_phase'].mode()[0]
        }
        
        sequences.append(seq_features)
        labels.append(seq_label)
        metadata.append(seq_meta)

sequences = np.array(sequences, dtype=np.float32)
print(f"✓ Created {len(sequences)} temporal sequences")
print(f"Sequence shape: {sequences.shape}")  # (num_sequences, sequence_length, num_features)

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
print(f"\nLabel encoding:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label}: {i} ({np.sum(labels_encoded == i)} sequences)")

# Normalize features
print("\n[7] Normalizing features...")
from sklearn.preprocessing import StandardScaler

# Reshape for scaling
original_shape = sequences.shape
sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])

scaler = StandardScaler()
sequences_normalized = scaler.fit_transform(sequences_reshaped)
sequences_normalized = sequences_normalized.reshape(original_shape)

print(f"✓ Features normalized using StandardScaler")

# Create node feature matrix (aggregate per satellite)
print("\n[8] Creating node feature matrix...")
node_features = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)

for node_id in all_nodes:
    node_idx = node_to_idx[node_id]
    # Aggregate features for flows involving this satellite
    node_flows = df[(df['src'] == node_id) | (df['dst'] == node_id)]
    if len(node_flows) > 0:
        node_features[node_idx] = node_flows[feature_cols].mean().values

# Normalize node features
node_features = scaler.transform(node_features)
print(f"Node feature matrix shape: {node_features.shape}")

# Save processed data
print("\n[9] Saving processed data...")
data_dict = {
    'sequences': sequences_normalized,
    'labels': labels_encoded,
    'metadata': metadata,
    'adj_matrix': adj_matrix,
    'node_features': node_features,
    'node_to_idx': node_to_idx,
    'label_encoder': label_encoder,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'sequence_length': sequence_length
}

with open('/home/ubuntu/orbitguard/processed_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
print("✓ Saved: processed_data.pkl")

# Save feature importance reference
feature_importance_ref = pd.DataFrame({
    'feature': feature_cols,
    'category': ['flow']*3 + ['routing']*3 + ['isl']*5 + ['engineered']*13
})
feature_importance_ref.to_csv('/home/ubuntu/orbitguard/feature_reference.csv', index=False)
print("✓ Saved: feature_reference.csv")

# Statistics
print("\n[10] Data statistics:")
print(f"  Total sequences: {len(sequences_normalized)}")
print(f"  Sequence length: {sequence_length}")
print(f"  Number of features: {len(feature_cols)}")
print(f"  Number of nodes: {num_nodes}")
print(f"  Number of edges: {len(edge_list)}")
print(f"  Number of classes: {len(label_encoder.classes_)}")
print(f"  Class distribution:")
for i, label in enumerate(label_encoder.classes_):
    count = np.sum(labels_encoded == i)
    print(f"    {label}: {count} ({count/len(labels_encoded)*100:.2f}%)")

print("\n" + "=" * 80)
print("GRAPH CONSTRUCTION COMPLETE!")
print("=" * 80)
