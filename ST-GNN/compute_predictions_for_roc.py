#!/usr/bin/env python3
"""
Compute predictions with probabilities for ROC curve generation
"""

import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

print("=" * 80)
print("COMPUTING PREDICTIONS WITH PROBABILITIES")
print("=" * 80)

# Load model and data
print("\n[1] Loading model and data...")
device = torch.device('cpu')

# Load model architecture
import sys
sys.path.append('/home/ubuntu/leo_attack_detection')
from efficient_stgnn_05 import EfficientSTGNN

# Load processed data
with open('/home/ubuntu/leo_attack_detection/processed_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

sequences = data_dict['sequences']
labels = data_dict['labels']
label_encoder = data_dict['label_encoder']

print(f"✓ Total sequences: {len(sequences)}")
print(f"✓ Number of classes: {len(label_encoder.classes_)}")

# Split data (same as training: 70% train, 15% val, 15% test)
X_temp, X_test, y_temp, y_test = train_test_split(
    sequences, labels, test_size=0.15, random_state=42, stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"✓ Test set size: {len(X_test)}")

# Load trained model
checkpoint = torch.load('/home/ubuntu/leo_attack_detection/best_efficient_model.pt', 
                        map_location=device)

input_dim = X_test.shape[2]
hidden_dim = 64
num_classes = len(label_encoder.classes_)

model = EfficientSTGNN(
    num_features=input_dim,
    num_classes=num_classes,
    hidden_dim=hidden_dim
)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("✓ Model loaded successfully")

# Create DataLoader
print("\n[2] Computing predictions...")
test_dataset = TensorDataset(
    torch.FloatTensor(X_test),
    torch.LongTensor(y_test)
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Get predictions with probabilities
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for sequences_batch, labels_batch in test_loader:
        sequences_batch = sequences_batch.to(device)
        
        # Forward pass
        outputs = model(sequences_batch)
        
        # Get probabilities using softmax
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_pred_proba = np.array(all_probs)

print(f"✓ Predictions computed for {len(y_true)} samples")
print(f"✓ Probability matrix shape: {y_pred_proba.shape}")

# Verify accuracy matches previous results
accuracy = (y_true == y_pred).mean()
print(f"✓ Test accuracy: {accuracy:.4f}")

# Print per-class sample counts
print("\nTest set distribution:")
for i, class_name in enumerate(label_encoder.classes_):
    count = (y_true == i).sum()
    print(f"  {class_name}: {count} samples")

# Save predictions
print("\n[3] Saving predictions...")
predictions = {
    'y_true': y_true,
    'y_pred': y_pred,
    'y_pred_proba': y_pred_proba,
    'label_encoder': label_encoder
}

with open('/home/ubuntu/leo_attack_detection/predictions_with_proba.pkl', 'wb') as f:
    pickle.dump(predictions, f)

print("✓ Predictions saved to: predictions_with_proba.pkl")
print("=" * 80)
print("✅ DONE! Ready for ROC curve generation")
print("=" * 80)
