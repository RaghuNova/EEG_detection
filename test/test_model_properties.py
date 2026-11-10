"""Model behavior, dimensions, attention properties, and CUDA tests."""

import pytest
import torch
import torch.nn as nn

from backend.dl.attention import AdditiveTemporalAttention, AttentionClassifier
from backend.dl.model import CNNBiLSTM


def test_cnn_bilstm_shapes_and_gradients() -> None:
    batch_size = 4
    channels = 18
    samples = 5120
    model = CNNBiLSTM(input_channels=channels, lstm_hidden=64, recurrent_length=128)

    x = torch.randn(batch_size, channels, samples, requires_grad=True)
    logits = model(x)
    assert logits.shape == (batch_size,)

    temp_features = model.extract_temporal_features(x)
    assert temp_features.shape == (batch_size, 128, 128)

    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Verify parameter updates
    param = list(model.parameters())[0]
    assert param.grad is not None


def test_cnn_bilstm_batch_size_1_and_variable_batches() -> None:
    model = CNNBiLSTM(input_channels=2, lstm_hidden=32, recurrent_length=64)
    model.eval()

    # Batch size 1
    x1 = torch.randn(1, 2, 5120)
    out1 = model(x1)
    assert out1.shape == (1,)

    # Batch size 7
    x7 = torch.randn(7, 2, 5120)
    out7 = model(x7)
    assert out7.shape == (7,)


def test_cnn_bilstm_invalid_dimensions() -> None:
    model = CNNBiLSTM(input_channels=2)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 5120))  # 2D instead of 3D

    with pytest.raises(ValueError, match="expected 2 channels"):
        model(torch.randn(2, 4, 5120))  # 4 channels instead of 2


def test_attention_properties() -> None:
    attn = AdditiveTemporalAttention(input_dim=128, attention_dim=64)
    features = torch.randn(5, 128, 128)
    context, weights = attn(features)

    assert context.shape == (5, 128)
    assert weights.shape == (5, 128)
    # 1. Non-negativity
    assert (weights >= 0.0).all()
    # 2. Sum to one over time dimension
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-5)
    # 3. All finite
    assert torch.isfinite(context).all()
    assert torch.isfinite(weights).all()


def test_attention_classifier_gradients() -> None:
    classifier = AttentionClassifier(input_dim=128, attention_dim=64)
    features = torch.randn(3, 128, 128, requires_grad=True)
    logits, weights = classifier.forward_with_attention(features)
    assert logits.shape == (3,)
    assert weights.shape == (3, 128)

    loss = logits.sum()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_cuda_execution_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    device = torch.device("cuda")
    model = CNNBiLSTM(input_channels=2, lstm_hidden=16, recurrent_length=32).to(device)
    x = torch.randn(2, 2, 5120, device=device)
    out = model(x)
    assert out.device.type == "cuda"
    assert out.shape == (2,)
