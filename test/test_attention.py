import torch

from backend.dl.attention import AdditiveTemporalAttention, AttentionClassifier
from backend.dl.model import CNNBiLSTM


def test_temporal_features_are_not_mean_pooled() -> None:
    model = CNNBiLSTM(2, recurrent_length=8)
    features = model.extract_temporal_features(torch.zeros((2, 2, 5120)))
    assert features.shape == (2, 8, 128)


def test_attention_shapes_weights_and_logits() -> None:
    attention = AdditiveTemporalAttention()
    features = torch.randn(4, 8, 128)
    context, weights = attention(features)
    assert context.shape == (4, 128)
    assert weights.shape == (4, 8)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(context).all()

    classifier = AttentionClassifier()
    logits = classifier(features)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
