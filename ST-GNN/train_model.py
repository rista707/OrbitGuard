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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import model
import sys
sys.path.append('/home/ubuntu/orbitguard')
from stgnn_model import NovelSTGNN, FocalLoss, get_class_weights

print("=" * 80)
print("ST-GNN MODEL TRAINING WITH CLASS BALANCING")
print("=" * 80)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load processed data
print("\n[1] Loading processed data...")
with open('/home/ubuntu/orbitguard/processed_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

sequences = data_dict['sequences']
labels = data_dict['labels']
adj_matrix = data_dict['adj_matrix']
label_encoder = data_dict['label_encoder']

print(f"✓ Loaded {len(sequences)} sequences")
print(f"✓ Sequence shape: {sequences.shape}")
print(f"✓ Number of classes: {len(label_encoder.classes_)}")

# Class distribution
print("\n[2] Class distribution:")
for i, label in enumerate(label_encoder.classes_):
    count = np.sum(labels == i)
    print(f"  {label}: {count} ({count/len(labels)*100:.2f}%)")

# Create edge index from adjacency matrix
print("\n[3] Creating edge index from adjacency matrix...")
edge_list = []
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if adj_matrix[i, j] > 0:
            edge_list.append([i, j])
edge_index = torch.LongTensor(edge_list).t().contiguous()
print(f"✓ Edge index shape: {edge_index.shape}")

# Stratified train/val/test split
print("\n[4] Splitting data (stratified)...")
X_temp, X_test, y_temp, y_test = train_test_split(
    sequences, labels, test_size=0.15, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=42
)

print(f"✓ Train set: {len(X_train)} samples")
print(f"✓ Val set: {len(X_val)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Calculate class weights for balanced training
print("\n[5] Calculating class weights...")
num_classes = len(label_encoder.classes_)
class_weights = get_class_weights(y_train, num_classes)
print(f"✓ Class weights: {class_weights.numpy()}")

# Create weighted sampler for balanced batches
print("\n[6] Creating weighted sampler...")
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
print(f"✓ Weighted sampler created")

# Custom Dataset
class SatelliteDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Create datasets
train_dataset = SatelliteDataset(X_train, y_train)
val_dataset = SatelliteDataset(X_val, y_val)
test_dataset = SatelliteDataset(X_test, y_test)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n[7] DataLoaders created:")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Initialize model
print("\n[8] Initializing model...")
num_features = sequences.shape[-1]
hidden_dim = 128
num_blocks = 3

model = NovelSTGNN(
    num_features=num_features,
    num_classes=num_classes,
    hidden_dim=hidden_dim,
    num_blocks=num_blocks
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
edge_index = edge_index.to(device)
class_weights = class_weights.to(device)

print(f"✓ Model initialized on {device}")
print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss functions
print("\n[9] Setting up loss functions...")
# Focal loss with class weights
focal_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
# Auxiliary loss
aux_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

print(f"✓ Focal loss (gamma=2.0) with class weights")
print(f"✓ Auxiliary cross-entropy loss")

# Optimizer with weight decay for regularization
print("\n[10] Setting up optimizer...")
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
print(f"✓ AdamW optimizer (lr=0.001, weight_decay=1e-4)")
print(f"✓ ReduceLROnPlateau scheduler")

# Training function
def train_epoch(model, loader, optimizer, edge_index, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, aux_logits = model(batch_x, edge_index)
        
        # Calculate losses
        main_loss = focal_loss_fn(logits, batch_y)
        aux_loss = aux_loss_fn(aux_logits, batch_y)
        loss = main_loss + 0.3 * aux_loss  # Weighted combination
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Validation function
def validate(model, loader, edge_index, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits, aux_logits = model(batch_x, edge_index)
            
            main_loss = focal_loss_fn(logits, batch_y)
            aux_loss = aux_loss_fn(aux_logits, batch_y)
            loss = main_loss + 0.3 * aux_loss
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1_macro, f1_weighted, all_preds, all_labels

# Training loop
print("\n[11] Starting training...")
print("=" * 80)

num_epochs = 50
best_val_f1 = 0
patience_counter = 0
early_stop_patience = 15

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'val_f1_macro': [], 'val_f1_weighted': []
}

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, edge_index, device)
    
    # Validate
    val_loss, val_acc, val_f1_macro, val_f1_weighted, _, _ = validate(
        model, val_loader, edge_index, device
    )
    
    # Update scheduler
    scheduler.step(val_f1_macro)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1_macro'].append(val_f1_macro)
    history['val_f1_weighted'].append(val_f1_weighted)
    
    # Print progress
    print(f"Epoch {epoch+1:02d}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
          f"F1-Macro: {val_f1_macro:.4f} F1-Weighted: {val_f1_weighted:.4f}")
    
    # Save best model
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1_macro': val_f1_macro,
            'val_f1_weighted': val_f1_weighted,
        }, '/home/ubuntu/orbitguard/best_model.pt')
        print(f"  ✓ Best model saved (F1-Macro: {val_f1_macro:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
        break

print("\n" + "=" * 80)
print(f"TRAINING COMPLETE!")
print(f"Best validation F1-Macro: {best_val_f1:.4f}")
print("=" * 80)

# Save training history
with open('/home/ubuntu/orbitguard/training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
print("\n✓ Saved: training_history.pkl")

# Plot training curves
print("\n[12] Generating training curves...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1 Scores
axes[1, 0].plot(history['val_f1_macro'], label='F1-Macro', linewidth=2, color='green')
axes[1, 0].plot(history['val_f1_weighted'], label='F1-Weighted', linewidth=2, color='orange')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('F1 Score', fontsize=12)
axes[1, 0].set_title('Validation F1 Scores', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning rate (if available)
axes[1, 1].text(0.5, 0.5, f'Best Val F1-Macro:\n{best_val_f1:.4f}', 
                ha='center', va='center', fontsize=24, fontweight='bold',
                transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/ubuntu/orbitguard/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_curves.png")

print("\n" + "=" * 80)
print("TRAINING SCRIPT COMPLETE!")
print("=" * 80)
