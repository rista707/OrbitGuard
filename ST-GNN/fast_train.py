#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

from efficient_stgnn import EfficientSTGNN, FocalLoss

print("=" * 80)
print("EFFICIENT ST-GNN TRAINING")
print("=" * 80)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Load data
print("\n[1] Loading data...")
with open('/home/ubuntu/orbitguard/processed_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

sequences = data_dict['sequences']
labels = data_dict['labels']
label_encoder = data_dict['label_encoder']

print(f"✓ Sequences: {sequences.shape}")
print(f"✓ Classes: {label_encoder.classes_}")

# Split data
print("\n[2] Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(
    sequences, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Calculate class weights
num_classes = len(label_encoder.classes_)
class_counts = np.bincount(y_train, minlength=num_classes)
class_weights = len(y_train) / (num_classes * class_counts)
class_weights_tensor = torch.FloatTensor(class_weights)

print(f"\n[3] Class weights: {class_weights}")

# Create weighted sampler
sample_weights = class_weights_tensor[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Dataset
class SatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SatDataset(X_train, y_train)
val_dataset = SatDataset(X_val, y_val)
test_dataset = SatDataset(X_test, y_test)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n[4] DataLoaders ready (batch_size={batch_size})")

# Model
print("\n[5] Initializing model...")
model = EfficientSTGNN(num_features=24, num_classes=4, hidden_dim=64)
device = torch.device('cpu')
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model parameters: {total_params:,}")

# Loss and optimizer
criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

print(f"✓ Focal Loss with class weights")
print(f"✓ AdamW optimizer")

# Training functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader), accuracy, f1_macro, f1_weighted, all_preds, all_labels

# Training loop
print("\n[6] Training...")
print("=" * 80)

num_epochs = 30
best_f1 = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
           'val_f1_macro': [], 'val_f1_weighted': []}

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = evaluate(model, val_loader, criterion, device)
    
    scheduler.step(val_f1_macro)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1_macro'].append(val_f1_macro)
    history['val_f1_weighted'].append(val_f1_weighted)
    
    print(f"Epoch {epoch+1:02d}/{num_epochs} | "
          f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
          f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
          f"F1: {val_f1_macro:.4f}/{val_f1_weighted:.4f}")
    
    if val_f1_macro > best_f1:
        best_f1 = val_f1_macro
        torch.save(model.state_dict(), '/home/ubuntu/orbitguard/best_efficient_model.pt')
        print(f"  ✓ Best model saved (F1={val_f1_macro:.4f})")

print("\n" + "=" * 80)
print(f"Training complete! Best F1-Macro: {best_f1:.4f}")

# Load best model and evaluate on test set
print("\n[7] Evaluating on test set...")
model.load_state_dict(torch.load('/home/ubuntu/orbitguard/best_efficient_model.pt'))
test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels = evaluate(
    model, test_loader, criterion, device
)

print(f"\nTest Results:")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1-Macro: {test_f1_macro:.4f}")
print(f"  F1-Weighted: {test_f1_weighted:.4f}")

# Classification report
print("\n[8] Classification Report:")
print("=" * 80)
report = classification_report(test_labels, test_preds, target_names=label_encoder.classes_, digits=4)
print(report)

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    test_labels, test_preds, average=None, labels=range(num_classes)
)

print("\n[9] Per-Class Metrics:")
print("=" * 80)
for i, class_name in enumerate(label_encoder.classes_):
    print(f"{class_name:12s} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | "
          f"F1: {f1[i]:.4f} | Support: {support[i]}")

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
print("\n[10] Confusion Matrix:")
print(cm)

# Save results
results = {
    'history': history,
    'test_accuracy': test_acc,
    'test_f1_macro': test_f1_macro,
    'test_f1_weighted': test_f1_weighted,
    'classification_report': report,
    'confusion_matrix': cm,
    'per_class_metrics': {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist()
    }
}

with open('/home/ubuntu/orbitguard/results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Visualizations
print("\n[11] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Training curves
axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history['train_acc'], label='Train', linewidth=2)
axes[0, 1].plot(history['val_acc'], label='Val', linewidth=2)
axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1 scores
axes[1, 0].plot(history['val_f1_macro'], label='F1-Macro', linewidth=2)
axes[1, 0].plot(history['val_f1_weighted'], label='F1-Weighted', linewidth=2)
axes[1, 0].set_title('F1 Scores', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
axes[1, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')

plt.tight_layout()
plt.savefig('/home/ubuntu/orbitguard/final_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: final_results.png")

# Per-class performance bar chart
fig2, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(label_encoder.classes_))
width = 0.25

ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
ax.bar(x, recall, width, label='Recall', alpha=0.8)
ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Attack Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(label_encoder.classes_, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

for i in range(len(label_encoder.classes_)):
    ax.text(i - width, precision[i] + 0.02, f'{precision[i]:.3f}', ha='center', fontsize=9)
    ax.text(i, recall[i] + 0.02, f'{recall[i]:.3f}', ha='center', fontsize=9)
    ax.text(i + width, f1[i] + 0.02, f'{f1[i]:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/home/ubuntu/orbitguard/per_class_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: per_class_performance.png")

print("\n" + "=" * 80)
print("ALL DONE!")
print("=" * 80)
