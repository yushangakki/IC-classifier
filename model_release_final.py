import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass
from utils_release_final import extract_graph_features, process_geometric_neighbors, get_graph_feature


@dataclass
class ModelConfig:
    """Configuration for network architecture"""
    k: int  # Number of nearest neighbors
    emb_dims: int  # Embedding dimensions
    dropout_rate: float  # Dropout probability
    num_classes: int = 67  # Number of output classe

class ChannelWiseAttention(nn.Module):
    """Channel-wise attention module that learns to focus on important feature channels"""
    
    def __init__(self, in_dim: int):
        super().__init__()
        
        # Dimensions for the query and key projections
        hidden_dim = 128
        
        # Projection layers
        self.query_proj = nn.Sequential(
            nn.Conv1d(1024, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.key_proj = nn.Sequential(
            nn.Conv1d(1024, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.value_proj = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, 1, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.ReLU()
        )
        
        # Learnable scaling factor
        self.alpha = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, channels, num_points)
        x_trans = x.permute(0, 2, 1)
        
        # Project to query and key spaces
        keys = self.key_proj(x_trans).permute(0, 2, 1)
        queries = self.query_proj(x_trans)
        
        # Compute attention scores
        attention_scores = torch.bmm(keys, queries)
        
        # Compute attention weights using max-subtraction for numerical stability
        max_scores = torch.max(attention_scores, -1, keepdim=True)[0]
        attention_weights = self.softmax(max_scores.expand_as(attention_scores) - attention_scores)
        
        # Apply attention to values
        values = self.value_proj(x)
        attended_values = torch.bmm(attention_weights, values)
        
        # Residual connection with learnable scaling
        return self.alpha * attended_values + x

class GraphAttention(nn.Module):
    """Graph attention module that processes local geometric structures"""
    
    def __init__(self, in_dim: int, out_dim: int, k: int):
        super().__init__()
        
        self.k = k
        
        # Edge feature processing
        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
        
        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(out_dim, in_dim, (1, k), bias=False),
            nn.BatchNorm2d(in_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Residual path
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Local feature aggregation
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, (1, k), bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Channel-wise attention modules
        self.attention1 = ChannelWiseAttention(out_dim)
        self.attention2 = ChannelWiseAttention(out_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get edge features
        edge_features = extract_graph_features(x, k=self.k)
        features_for_local = edge_features
        
        # Process edge features
        edge_features = self.edge_conv1(edge_features)
        identity = edge_features
        
        edge_features = self.edge_conv2(edge_features)
        edge_features = torch.squeeze(edge_features, -1)
        
        # Compute residual
        delta = edge_features - x
        delta_features = extract_graph_features(delta, k=self.k)
        delta_features = self.residual_conv(delta_features)
        
        # Combine features
        combined = identity + delta_features
        global_features = combined.max(dim=-1, keepdim=False)[0]
        global_features = self.attention1(global_features)
        
        
        # Process local features
        local_features = self.local_conv(features_for_local)
        local_features = torch.squeeze(local_features, -1)
        local_features = self.attention2(local_features)
        return global_features, local_features

class GeometricBackbone(nn.Module):
    """Main network combining graph attention modules with global feature aggregation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Graph attention layers
        self.attention_layers = nn.ModuleList([
            GraphAttention(14, 64, config.k),
            GraphAttention(64, 64, config.k),
            GraphAttention(64, 128, config.k),
            GraphAttention(128, 256, config.k)
        ])
        
        # Global feature processing
        self.global_conv = nn.Sequential(
            nn.Conv1d(1024, config.emb_dims, 1, bias=False),
            nn.BatchNorm1d(config.emb_dims),
            nn.LeakyReLU(0.2)
        )
        
        self.channel_attention = ChannelWiseAttention(1024)
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.emb_dims * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=config.dropout_rate),            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(256, 67)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size = x.size(0)
        
        # Preprocess input
        x = process_geometric_neighbors(x)
        
        # Process through attention layers
        features = []
        
        for attention in self.attention_layers:
            global_feat, local_feat = attention(x)
            features.extend([global_feat, local_feat])
            x = global_feat
        # Combine all features
        x = torch.cat(features, dim=1)
        x = self.global_conv(x)
        x = self.channel_attention(x)
        
        # Global pooling
        global_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        global_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat([global_max, global_avg], dim=1)
        
        # Classification
        return self.classifier(x)

class DGCNN(nn.Module):
    """Dynamic Graph CNN implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Edge convolution layers
        self.edge_conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(6, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(64 * 2, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(64 * 2, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(128 * 2, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            )
        ])
        
        # Global feature processing
        self.global_conv = nn.Sequential(
            nn.Conv1d(512, config.emb_dims, 1, bias=False),
            nn.BatchNorm1d(config.emb_dims),
            nn.LeakyReLU(0.2)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.emb_dims * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=config.dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=config.dropout_rate),
            
            nn.Linear(256, config.num_classes)
        )
        
        self.k = config.k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Process through edge convolution layers
        features = []
        for i, conv in enumerate(self.edge_conv_layers):
            edge_features = get_graph_feature(x if i == 0 else features[-1], k=self.k)
            x = conv(edge_features).max(dim=-1, keepdim=False)[0]
            features.append(x)
        
        # Combine features
        x = torch.cat(features, dim=1)
        x = self.global_conv(x)
        
        # Global pooling
        global_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        global_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat([global_max, global_avg], dim=1)
        
        # Classification
        return self.classifier(x)

# Example usage:
def create_model(model_type: str, config: ModelConfig) -> nn.Module:
    """Factory function to create the specified model type"""
    if model_type.lower() == 'gb':
        return GeometricBackbone(config)
    elif model_type.lower() == 'dgcnn':
        return DGCNN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")