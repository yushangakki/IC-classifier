import math
from typing import Optional, Tuple
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from dataclasses import dataclass

def calculate_smoothed_loss(predictions, targets, enable_smoothing=True, smoothing_factor=0.2):
    """
    Calculate cross entropy loss with optional label smoothing.
    
    Label smoothing helps prevent overconfident predictions by distributing
    some probability mass across all classes.
    
    Args:
        predictions (torch.Tensor): Model predictions logits
        targets (torch.Tensor): Ground truth labels
        enable_smoothing (bool): Whether to apply label smoothing
        smoothing_factor (float): Amount of smoothing to apply (0-1)
    
    Returns:
        torch.Tensor: Scalar loss value
    
    Note:
        When smoothing is enabled, the target probability distribution is:
        - (1 - smoothing_factor) for the correct class
        - smoothing_factor/(num_classes - 1) for all other classes
    """
    # Ensure targets are in the correct shape
    targets = targets.contiguous().view(-1)
    
    if not enable_smoothing:
        return F.cross_entropy(predictions, targets, reduction='mean')
        
    # Get number of classes from prediction shape
    num_classes = predictions.size(1)
    
    # Create one-hot encoding of targets
    one_hot_targets = torch.zeros_like(predictions).scatter(
        dim=1, 
        index=targets.view(-1, 1), 
        value=1
    )
    
    # Apply label smoothing
    smooth_factor = smoothing_factor
    target_dist = one_hot_targets * (1 - smooth_factor) + \
                 (1 - one_hot_targets) * smooth_factor / (num_classes - 1)
    
    # Calculate log probabilities
    log_probabilities = F.log_softmax(predictions, dim=1)
    
    # Calculate loss
    loss = -(target_dist * log_probabilities).sum(dim=1).mean()
    
    return loss

class IOStream:
    """File stream wrapper for logging."""
    
    def __init__(self, path: str):
        self.f = open(path, 'a')
        
    def cprint(self, text: str) -> None:
        print(text)
        self.f.write(text + '\n')
        self.f.flush()
        
    def close(self) -> None:
        self.f.close()

def compute_determinant(matrix):
    # Extract matrix elements
    a11 = matrix[..., 0, 0]
    a12 = matrix[..., 0, 1]
    a13 = matrix[..., 0, 2]
    a21 = matrix[..., 1, 0]
    a22 = matrix[..., 1, 1]
    a23 = matrix[..., 1, 2]
    a31 = matrix[..., 2, 0]
    a32 = matrix[..., 2, 1]
    a33 = matrix[..., 2, 2]
    
    # Calculate cofactor terms
    term1 = a11 * (a22 * a33 - a23 * a32)
    term2 = a12 * (a21 * a33 - a23 * a31)
    term3 = a13 * (a21 * a32 - a22 * a31)
    
    # Calculate determinant using cofactor expansion along first row
    det = term1 - term2 + term3
    
    return det

def compute_eigenvals(A: Tensor) -> Tensor:
    """Compute eigenvalues of 3x3 matrices using analytical method."""
    A_11, A_12, A_13 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    A_22, A_23, A_33 = A[..., 1, 1], A[..., 1, 2], A[..., 2, 2]
    
    I = torch.eye(3, device=A.device)
    p1 = torch.square(A_12) + torch.square(A_13) + torch.square(A_23)
    q = torch.sum(torch.diagonal(A, dim1=2, dim2=3), dim=2) / 3
    p2 = (torch.square(A_11 - q) + torch.square(A_22 - q) + 
          torch.square(A_33 - q) + 2 * p1)
    p = torch.sqrt(p2 / 6) + 1e-8
    
    q_4d = q.view(A.size(0), -1, 1, 1)
    p_4d = p.view(A.size(0), -1, 1, 1)
    B = (1 / p_4d) * (A - q_4d * I)
    
    r = torch.clip(compute_determinant(B) / 2, -1, 1)
    phi = torch.acos(r) / 3
    
    eig1 = q + 2 * p * torch.cos(phi)
    eig3 = q + 2 * p * torch.cos(phi + (2 * math.pi / 3))
    eig2 = 3 * q - eig1 - eig3
    
    return torch.abs(torch.stack([eig1, eig2, eig3], dim=2))

def calculate_geometric_features(grouped_xyz, nsample):
    """Calculate geometric features from grouped xyz coordinates."""
    device = torch.device('cuda')
    
    # Calculate mean and offsets
    xyz_mean = torch.mean(grouped_xyz, 2)
    xyz_offset = grouped_xyz - torch.tile(torch.unsqueeze(xyz_mean, 2), [1, 1, nsample, 1])
    
    # Compute covariance matrix
    offset_expanded1 = torch.unsqueeze(xyz_offset, -1)
    offset_expanded2 = torch.unsqueeze(xyz_offset, 3)
    covariance = torch.sum(torch.multiply(offset_expanded1, offset_expanded2), 2) / nsample

    # Compute and normalize eigenvalues
    eigenvalues = compute_eigenvals(covariance).type(torch.cuda.FloatTensor)
    eigenvalues = normalize_eigenvalues(eigenvalues)
    
    # Extract individual eigenvalues
    lambda_features = extract_lambda_features(eigenvalues)
    
    # Calculate geometric features
    shape_features = calculate_shape_features(*lambda_features)
    
    # Calculate height-based features
    height_features = calculate_height_features(grouped_xyz, nsample)
    
    # Combine all features
    neigh_geofeat = combine_geometric_features(shape_features, height_features, nsample)
    
    return neigh_geofeat

def normalize_eigenvalues(eigenvalues):
    """Normalize eigenvalues by their sum."""
    e_sum = torch.tile(torch.unsqueeze(torch.sum(eigenvalues, 2), 2), [1, 1, 3])
    return torch.div(eigenvalues, e_sum)

def extract_lambda_features(eigenvalues):
    """Extract individual eigenvalues."""
    lambda1 = eigenvalues[:, :, 0:1]
    lambda2 = eigenvalues[:, :, 1:2]
    lambda3 = eigenvalues[:, :, 2:3]
    return lambda1, lambda2, lambda3

def calculate_shape_features(lambda1, lambda2, lambda3):
    """Calculate shape-based geometric features."""
    return {
        'sum': lambda1 + lambda2 + lambda3,
        'omnivariance': torch.pow(torch.abs(lambda1 * lambda2 * lambda3), 1.0 / 3),
        'anisotropy': (lambda1 - lambda3) / lambda1,
        'planarity': (lambda2 - lambda3) / lambda1,
        'linearity': (lambda1 - lambda2) / lambda1,
        'surface_variation': lambda3 / (lambda1 + lambda2 + lambda3),
        'sphericity': lambda3 / lambda1
    }

def calculate_height_features(grouped_xyz, nsample):
    """Calculate height-based geometric features."""
    # Extract z-coordinates
    z_coords = grouped_xyz[:, :, :, 2:3]
    
    # Calculate z statistics
    z_max = torch.amax(grouped_xyz, 2)[:, :, 2:3]
    z_min = torch.amin(grouped_xyz, 2)[:, :, 2:3]
    
    # Create tiled versions for broadcasting
    z_max_tiled = torch.tile(torch.unsqueeze(z_max, -1), [1, 1, nsample, 1])
    z_min_tiled = torch.tile(torch.unsqueeze(z_min, -1), [1, 1, nsample, 1])
    
    # Calculate z-based features
    z_var = torch.var(z_coords.view(z_coords.shape[0], z_coords.shape[1], 1, -1), 
                     dim=3, unbiased=False, keepdim=True)
    z_var = torch.tile(z_var, [1, 1, nsample, 1])
    
    return {
        'z_range': z_max - z_min,
        'z_var': z_var,
        'z_max_diff': z_max_tiled - z_coords,
        'z_min_diff': z_coords - z_min_tiled
    }

def combine_geometric_features(shape_features, height_features, nsample):
    """Combine all geometric features into final feature tensor."""
    # Combine shape features
    combined_features = torch.cat([
        shape_features['sum'],
        shape_features['omnivariance'],
        shape_features['anisotropy'],
        shape_features['planarity'],
        shape_features['linearity'],
        shape_features['surface_variation'],
        shape_features['sphericity'],
        height_features['z_range']
    ], axis=2)
    
    # Expand features
    combined_features = torch.tile(torch.unsqueeze(combined_features, 2), [1, 1, nsample, 1])
    
    # Add height features
    final_features = torch.cat([
        combined_features,
        height_features['z_var'],
        height_features['z_max_diff'],
        height_features['z_min_diff']
    ], axis=-1)
    
    return final_features

def process_geometric_neighbors(x, k=3, idx=None):
    """Process geometric neighbors to extract edge features."""
    # Initialize indices if not provided
    if idx is None:
        idx = knn(x, k=k)
    
    # Get batch information
    batch_size = x.size(0)
    num_points = x.size(2)
    original_x = x
    
    # Reshape and prepare indices
    x = prepare_point_indices(x, batch_size, num_points, idx)
    
    # Extract neighbor features
    neighbors = extract_neighbors(x, batch_size, num_points, k, idx)
    
    # Calculate edge features
    edge_features = calculate_edge_features(neighbors, original_x)
    
    return edge_features

def prepare_point_indices(x, batch_size, num_points, idx):
    """Prepare point indices for neighbor extraction."""
    device = torch.device('cuda')
    x = x.view(batch_size, -1, num_points)
    
    # Create and adjust index base
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor) + idx_base
    
    return x.transpose(2, 1).contiguous()

def extract_neighbors(x, batch_size, num_points, k, idx):
    """Extract neighboring points based on indices."""
    _, num_dims = x.view(batch_size * num_points, -1).size()
    
    # Get neighbors and reshape
    neighbors = x.view(batch_size * num_points, -1)[idx.view(-1), :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)
    
    return neighbors.permute(0, 3, 1, 2)

def calculate_edge_features(neighbors, original_x):
    """Calculate edge features from neighbors."""
    # Extract first and second neighbors
    neighbor_1st = torch.squeeze(torch.index_select(neighbors, -1, torch.cuda.LongTensor([1])), -1)
    neighbor_2nd = torch.squeeze(torch.index_select(neighbors, -1, torch.cuda.LongTensor([2])), -1)
    
    # Calculate edges
    edge1 = neighbor_1st - original_x
    edge2 = neighbor_2nd - original_x
    
    # Calculate geometric features
    normals = torch.cross(edge1, edge2, dim=1)
    dist1 = torch.norm(edge1, dim=1, keepdim=True)
    dist2 = torch.norm(edge2, dim=1, keepdim=True)
    
    return torch.cat((original_x, normals, dist1, dist2, edge1, edge2), 1)

def extract_graph_features(x, k=20, idx=None):
    """Extract graph features with geometric processing."""
    # Get batch information and prepare data
    batch_size, num_dims, num_points = x.size()
    x, idx = prepare_graph_data(x, k, idx)
    
    # Extract features
    feature, x = extract_features(x, batch_size, num_points, k, num_dims, idx)
    
    # Apply geometric processing if needed
    if x.shape[-1] == 3:
        feature = apply_geometric_processing(feature, k)
    
    # Combine features
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature

def prepare_graph_data(x, k, idx):
    """Prepare data for graph feature extraction."""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    return x.transpose(2, 1).contiguous(), idx

def extract_features(x, batch_size, num_points, k, num_dims, idx):
    """Extract initial features from points."""
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return feature, x

def apply_geometric_processing(feature, k):
    """Apply geometric processing to features."""
    device = torch.device('cuda')
    
    # Calculate geometric features
    G_p = calculate_geometric_features(feature, k)
    G_p = torch.permute(G_p, (0, 3, 1, 2)).cuda()
    feature = torch.permute(feature, (0, 3, 1, 2))
    
    # Apply convolution and normalization
    num_channel = feature.shape[1]
    conv2d = torch.nn.Conv2d(
        in_channels=G_p.shape[1],
        out_channels=num_channel,
        kernel_size=1,
        device=device
    )
    
    G_p = F.conv2d(G_p, conv2d.weight)
    G_p = F.softmax(G_p, dim=3) * k
    
    # Apply geometric features
    feature = torch.mul(G_p, feature)
    feature = torch.permute(feature, (0, 2, 3, 1))
    
    return feature

def knn(points, k):
    """
    Find k-nearest neighbors for each point using Euclidean distance.
    
    Args:
        points (torch.Tensor): Input points tensor of shape (batch_size, dims, num_points)
        k (int): Number of nearest neighbors to find
    
    Returns:
        torch.Tensor: Indices of k-nearest neighbors for each point
        
    Note:
        Uses an optimized version of the Euclidean distance formula:
        ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    """
    # Calculate squared L2 norm for each point ||x||^2
    point_square = torch.sum(points**2, dim=1, keepdim=True)  # shape: (batch_size, 1, num_points)
    
    # Calculate dot product term -2<x,y>
    # Transpose to align dimensions for matrix multiplication
    dot_product = -2 * torch.matmul(points.transpose(2, 1), points)  
    
    # Combine terms to get pairwise distances
    # Using broadcasting: point_square + dot_product + point_square.T
    # Negative sign makes smaller distances appear first in topk
    distance_matrix = -point_square - dot_product - point_square.transpose(2, 1)
    
    # Get indices of k nearest neighbors
    # Only need indices [1], not values [0]
    neighbor_indices = distance_matrix.topk(k=k, dim=-1)[1]
    
    return neighbor_indices

def proj(x: np.ndarray, p: np.ndarray) -> Tensor:
    """Project vector x onto the plane perpendicular to vector p."""
    p = p / np.linalg.norm(p)
    result = x - p.T @ x.T @ p
    return torch.from_numpy(result).float().cuda()

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature