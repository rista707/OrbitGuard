#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np

class TemporalAttention(nn.Module):
    """Multi-head temporal attention mechanism"""
    def __init__(self, hidden_dim, num_heads=4):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape
        
        # Multi-head projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out(attn_output), attn_weights


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution with attention"""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super(SpatialGraphConv, self).__init__()
        self.gat1 = GATConv(in_dim, out_dim // num_heads, heads=num_heads, dropout=0.3)
        self.gat2 = GATConv(out_dim, out_dim // num_heads, heads=num_heads, dropout=0.3)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, x, edge_index):
        # x: (num_nodes, features)
        # edge_index: (2, num_edges)
        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = F.elu(self.bn2(self.gat2(x, edge_index)))
        return x


class SpatioTemporalBlock(nn.Module):
    """Combined spatio-temporal processing block"""
    def __init__(self, feature_dim, hidden_dim, num_heads=4):
        super(SpatioTemporalBlock, self).__init__()
        
        # Temporal processing
        self.temporal_lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.temporal_attn = TemporalAttention(hidden_dim * 2, num_heads)
        
        # Spatial processing
        self.spatial_conv = SpatialGraphConv(hidden_dim * 2, hidden_dim, num_heads)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x, edge_index):
        # x: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # Temporal processing
        temporal_out, (h_n, c_n) = self.temporal_lstm(x)
        temporal_attn_out, attn_weights = self.temporal_attn(temporal_out)
        
        # Take last time step for spatial processing
        last_temporal = temporal_attn_out[:, -1, :]  # (batch, hidden_dim*2)
        
        # Spatial processing (process each sample in batch)
        spatial_features = []
        for i in range(batch_size):
            # Create node features for this sample
            node_feat = last_temporal[i:i+1].expand(edge_index.max().item() + 1, -1)
            spatial_out = self.spatial_conv(node_feat, edge_index)
            # Aggregate spatial features
            spatial_agg = spatial_out.mean(dim=0, keepdim=True)
            spatial_features.append(spatial_agg)
        
        spatial_out = torch.cat(spatial_features, dim=0)  # (batch, hidden_dim)
        
        # Fusion
        combined = torch.cat([last_temporal, spatial_out], dim=-1)
        fused = self.fusion(combined)
        
        return fused, attn_weights


class NovelSTGNN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128, num_blocks=3, num_heads=4):
        super(NovelSTGNN, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Spatio-temporal blocks
        self.st_blocks = nn.ModuleList([
            SpatioTemporalBlock(hidden_dim if i == 0 else hidden_dim, hidden_dim, num_heads)
            for i in range(num_blocks)
        ])
        
        # Residual projections
        self.residual_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])
        
        # Global context aggregation
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim * num_blocks, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Classification head with multiple branches
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Auxiliary classifier for regularization
        self.aux_classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, return_attention=False):
        # x: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # Input embedding
        x_embedded = torch.stack([self.input_embedding(x[:, t, :]) for t in range(seq_len)], dim=1)
        
        # Process through ST blocks with residual connections
        block_outputs = []
        attention_weights = []
        x_current = x_embedded
        
        for i, (st_block, res_proj) in enumerate(zip(self.st_blocks, self.residual_projs)):
            # ST block
            block_out, attn = st_block(x_current, edge_index)
            
            # Residual connection
            if i > 0:
                block_out = block_out + res_proj(block_outputs[-1])
            
            block_outputs.append(block_out)
            attention_weights.append(attn)
            
            # Prepare for next block
            x_current = block_out.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Global context aggregation
        global_feat = torch.cat(block_outputs, dim=-1)
        global_feat = self.global_context(global_feat)
        
        # Main classification
        logits = self.classifier(global_feat)
        
        # Auxiliary classification (for training stability)
        aux_logits = self.aux_classifier(block_outputs[-1])
        
        if return_attention:
            return logits, aux_logits, attention_weights
        return logits, aux_logits


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(labels, num_classes):
    """Calculate class weights for balanced training"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.FloatTensor(class_weights)


# Test model architecture
if __name__ == "__main__":
    # Model parameters
    num_features = 24
    num_classes = 4
    hidden_dim = 128
    num_blocks = 3
    batch_size = 16
    seq_len = 10
    num_nodes = 22
    
    # Create model
    model = NovelSTGNN(num_features, num_classes, hidden_dim, num_blocks)
    print(f"\n✓ Model created successfully")
    print(f"  Input features: {num_features}")
    print(f"  Output classes: {num_classes}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of ST blocks: {num_blocks}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    
    print(f"\n✓ Testing forward pass...")
    print(f"  Input shape: {x.shape}")
    print(f"  Edge index shape: {edge_index.shape}")
    
    logits, aux_logits, attn_weights = model(x, edge_index, return_attention=True)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Auxiliary logits shape: {aux_logits.shape}")
    print(f"  Number of attention weight tensors: {len(attn_weights)}")
    
    # Test loss functions
    print(f"\n✓ Testing loss functions...")
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Standard cross-entropy
    ce_loss = F.cross_entropy(logits, labels)
    print(f"  Cross-entropy loss: {ce_loss.item():.4f}")
    
    # Focal loss
    class_weights = torch.ones(num_classes)
    focal_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    focal_loss = focal_loss_fn(logits, labels)
    print(f"  Focal loss: {focal_loss.item():.4f}")
    
    print("\n" + "=" * 80)
    print("ARCHITECTURE TEST COMPLETE!")
    print("=" * 80)
