import torch
import torch.nn.functional as F


# 特徴量のL2ノルム
def feature_l2_norm(features: torch.Tensor) -> float:
    # features: (batch, ...)
    return features.view(features.size(0), -1).norm(p=2, dim=1).mean().item()


# 活性化のスパース性（0に近い値の割合）
def activation_sparsity(features: torch.Tensor, threshold=1e-3) -> float:
    # features: (batch, ...)
    flat = features.view(features.size(0), -1)
    sparsity = (torch.abs(flat) < threshold).float().mean().item()
    return sparsity


# 予測確率の最大値
def max_softmax_prob(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=1)
    max_probs = probs.max(dim=1)[0]
    return max_probs.mean().item()


# 使い方例:
# l2 = feature_l2_norm(activation)
# sp = activation_sparsity(activation)
# mp = max_softmax_prob(logits)
