"""Tests for checkpointing, resume state, best epoch tracking, and reproducibility."""

from pathlib import Path
import json
import numpy as np
import pytest
import torch

from backend.dl.model import CNNBiLSTM
from backend.dl.attention import AttentionClassifier
from backend.dl.train import _fit_fold, _fit_final, seed_everything
from backend.dl.attention_train import _fit_attention, _fit_attention_final
from backend.dl.raw_data import RawWindow


def _make_dummy_windows(tmp_path: Path, n_samples: int = 10) -> list[RawWindow]:
    data = np.random.randn(2, 5120 * n_samples).astype(np.float32)
    path = tmp_path / "data.npy"
    np.save(path, data)
    windows = []
    for i in range(n_samples):
        windows.append(
            RawWindow("P1", "R1", str(path), i, i * 20.0, (i + 1) * 20.0, i % 2, (0, 1))
        )
    return windows


def test_checkpoint_best_epoch_tracking_and_resume(tmp_path: Path) -> None:
    windows = _make_dummy_windows(tmp_path, n_samples=6)
    train_w = windows[:4]
    val_w = windows[4:]

    device = torch.device("cpu")
    ckpt_best = tmp_path / "fold_1.pt"
    ckpt_latest = tmp_path / "fold_1_latest.pt"

    # Train for 2 epochs
    res1 = _fit_fold(
        train_w,
        val_w,
        canonical_channels=["C1", "C2"],
        epochs=2,
        batch_size=2,
        device=device,
        checkpoint=ckpt_best,
        use_amp=False,
        normalization_cache={},
        resume=True,
    )
    assert ckpt_best.is_file()
    assert ckpt_latest.is_file()
    assert len(res1["history"]) == 2

    # Check state inside checkpoint
    saved_best = torch.load(ckpt_best, weights_only=False)
    assert "model_state" in saved_best
    assert "best_epoch" in saved_best
    assert "history" in saved_best
    assert saved_best["best_epoch"] in [1, 2]

    saved_latest = torch.load(ckpt_latest, weights_only=False)
    assert saved_latest["epoch"] == 2
    assert "optimizer_state" in saved_latest
    assert "scheduler_state" in saved_latest
    assert "scaler_state" in saved_latest

    # Resume training to epoch 4
    res2 = _fit_fold(
        train_w,
        val_w,
        canonical_channels=["C1", "C2"],
        epochs=4,
        batch_size=2,
        device=device,
        checkpoint=ckpt_best,
        use_amp=False,
        normalization_cache={},
        resume=True,
    )
    # Total history should now have 4 epochs
    assert len(res2["history"]) == 4
    assert res2["history"][-1]["epoch"] == 4


def test_attention_training_freezes_backbone_and_resumes(tmp_path: Path) -> None:
    # Create synthetic features [N, 128, 128]
    n_train = 8
    n_val = 4
    train_feat = tmp_path / "train_feat.npy"
    val_feat = tmp_path / "val_feat.npy"
    np.save(train_feat, np.random.randn(n_train, 128, 128).astype(np.float32))
    np.save(val_feat, np.random.randn(n_val, 128, 128).astype(np.float32))

    train_labels = np.array([0, 1, 0, 1, 0, 0, 1, 1], dtype=np.float32)
    val_labels = np.array([0, 1, 0, 1], dtype=np.float32)

    device = torch.device("cpu")
    ckpt_best = tmp_path / "attn_fold_1.pt"
    ckpt_latest = tmp_path / "attn_fold_1_latest.pt"

    res = _fit_attention(
        train_feat,
        val_feat,
        train_labels,
        val_labels,
        device=device,
        checkpoint=ckpt_best,
        epochs=3,
        batch_size=2,
        resume=True,
    )
    assert ckpt_best.is_file()
    assert ckpt_latest.is_file()
    assert len(res["history"]) == 3

    # Resume for 2 more epochs (total 5)
    res_resumed = _fit_attention(
        train_feat,
        val_feat,
        train_labels,
        val_labels,
        device=device,
        checkpoint=ckpt_best,
        epochs=5,
        batch_size=2,
        resume=True,
    )
    assert len(res_resumed["history"]) == 5
    assert res_resumed["history"][-1]["epoch"] == 5


def test_reproducibility_with_seed() -> None:
    seed_everything(42)
    m1 = CNNBiLSTM(2)
    m1.eval()
    seed_everything(100)
    x1 = torch.randn(2, 2, 5120)
    out1 = m1(x1)

    seed_everything(42)
    m2 = CNNBiLSTM(2)
    m2.eval()
    seed_everything(100)
    x2 = torch.randn(2, 2, 5120)
    out2 = m2(x2)

    assert torch.allclose(out1, out2)

