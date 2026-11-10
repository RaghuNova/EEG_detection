"""Version 1 raw-signal CNN + Bi-LSTM baseline.

The model is designed to run on the NVIDIA CUDA device during CV and training;
the caller may select CPU explicitly for tests or CUDA-free environments.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CNNBiLSTM(nn.Module):
    """Classify one 20-second multichannel EEG window.

    The input is ``[batch, channels, 5120]``. The recurrent layer models the
    temporal representation within one window; it does not join recordings.
    """

    def __init__(
        self,
        input_channels: int,
        *,
        lstm_hidden: int = 64,
        dropout: float = 0.30,
        recurrent_length: int = 128,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or lstm_hidden <= 0 or recurrent_length <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_channels = input_channels
        self.recurrent_length = recurrent_length
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(recurrent_length),
        )
        self.bilstm = nn.LSTM(
            input_size=96,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, signal: Tensor) -> Tensor:
        temporal = self.extract_temporal_features(signal)
        pooled = temporal.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)

    def extract_temporal_features(self, signal: Tensor) -> Tensor:
        """Return the complete Bi-LSTM sequence before temporal pooling."""
        if signal.ndim != 3:
            raise ValueError("signal must have shape [batch, channels, samples]")
        if signal.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} channels, got {signal.shape[1]}"
            )
        encoded = self.cnn(signal.float()).transpose(1, 2)
        recurrent, _ = self.bilstm(encoded)
        return recurrent
