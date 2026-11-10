# """Stage 2 Bahdanau-style additive temporal attention."""

# from __future__ import annotations

# import torch
# from torch import Tensor, nn


# class AdditiveTemporalAttention(nn.Module):
#     """Pool ``[batch, time, features]`` with normalized learned scores."""

#     def __init__(
#         self,
#         input_dim: int = 128,
#         attention_dim: int = 64,
#         dropout: float = 0.10,
#     ) -> None:
#         super().__init__()
#         if input_dim <= 0 or attention_dim <= 0:
#             raise ValueError("attention dimensions must be positive")
#         self.input_dim = input_dim
#         self.score = nn.Sequential(
#             nn.Linear(input_dim, attention_dim),
#             nn.Tanh(),
#             nn.Dropout(dropout),
#             nn.Linear(attention_dim, 1, bias=True),
#         )

#     def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
#         if features.ndim != 3 or features.shape[-1] != self.input_dim:
#             raise ValueError(
#                 f"features must have shape [batch, time, {self.input_dim}]"
#             )
#         scores = self.score(features).squeeze(-1)
#         weights = torch.softmax(scores, dim=1)
#         context = torch.sum(features * weights.unsqueeze(-1), dim=1)
#         return context, weights


# class AttentionClassifier(nn.Module):
#     """Frozen-backbone temporal attention classifier returning raw logits."""

#     def __init__(
#         self,
#         input_dim: int = 128,
#         *,
#         attention_dim: int = 64,
#         dropout: float = 0.30,
#     ) -> None:
#         super().__init__()
#         self.attention = AdditiveTemporalAttention(
#             input_dim=input_dim,
#             attention_dim=attention_dim,
#             dropout=0.10,
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1),
#         )

#     def forward(self, features: Tensor) -> Tensor:
#         context, _ = self.attention(features)
#         return self.classifier(context).squeeze(-1)

#     def forward_with_attention(self, features: Tensor) -> tuple[Tensor, Tensor]:
#         context, weights = self.attention(features)
#         return self.classifier(context).squeeze(-1), weights
"""Four-head additive temporal attention for the EEG seizure-detection Stage 2.

Input: [batch, time, 128] frozen CNN-BiLSTM temporal features.
Output: one raw seizure logit per 20-second window.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiHeadAdditiveTemporalAttention(nn.Module):
    """Parallel Bahdanau/additive attention heads over the temporal axis.

    Each head learns an independent relevance distribution over the 128
    recurrent time steps. Head contexts are concatenated and projected back
    to input_dim so the residual path has a stable shape.
    """
    def __init__(self, input_dim: int = 128, num_heads: int = 4,
                 head_dim: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        if input_dim <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("attention dimensions and head count must be positive")
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scorers = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, head_dim), nn.Tanh(),
                          nn.Dropout(dropout), nn.Linear(head_dim, 1))
            for _ in range(num_heads)
        ])
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim * num_heads, input_dim),
            nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [batch, time, {self.input_dim}]")
        contexts, weights = [], []
        for scorer in self.scorers:
            alpha = torch.softmax(scorer(features).squeeze(-1), dim=1)
            context = torch.sum(features * alpha.unsqueeze(-1), dim=1)
            contexts.append(context)
            weights.append(alpha)
        fused = self.output_projection(torch.cat(contexts, dim=-1))
        return fused, torch.stack(weights, dim=1)


class GatedResidualFusion(nn.Module):
    """Learn how much attended context to combine with mean-pooled context."""
    def __init__(self, input_dim: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim * 2, input_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, original: Tensor, attended: Tensor) -> Tensor:
        gate = self.gate(torch.cat((original, attended), dim=-1))
        fused = gate * attended + (1.0 - gate) * original
        return self.norm(original + self.dropout(fused))


class AttentionClassifier(nn.Module):
    """Four-head attention + gated residual fusion + compact MLP classifier."""
    def __init__(self, input_dim: int = 128, *, num_heads: int = 4,
                 head_dim: int = 32, attention_dropout: float = 0.15,
                 classifier_dropout: float = 0.30) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.attention = MultiHeadAdditiveTemporalAttention(
            input_dim=input_dim, num_heads=num_heads,
            head_dim=head_dim, dropout=attention_dropout,
        )
        self.fusion = GatedResidualFusion(input_dim, dropout=attention_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(classifier_dropout), nn.Linear(64, 32), nn.GELU(),
            nn.Dropout(classifier_dropout * 0.5), nn.Linear(32, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        context, _ = self.attention(features)
        pooled = features.mean(dim=1)
        fused = self.fusion(pooled, context)
        return self.classifier(fused).squeeze(-1)

    def forward_with_attention(self, features: Tensor) -> tuple[Tensor, Tensor]:
        context, weights = self.attention(features)
        pooled = features.mean(dim=1)
        fused = self.fusion(pooled, context)
        return self.classifier(fused).squeeze(-1), weights
