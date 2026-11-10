"""End-to-end integration pipeline tests on synthetic data."""

from pathlib import Path
import json
import numpy as np
import pytest
import torch

from backend.dl.model import CNNBiLSTM
from backend.dl.attention import AttentionClassifier
from backend.dl.train import _fit_fold, _fit_final
from backend.dl.attention_train import extract_features, _fit_attention, _fit_attention_final
from backend.dl.comparison import compute_comparison
from backend.dl.raw_data import RawWindow


def test_mini_synthetic_pipeline_e2e(tmp_path: Path) -> None:
    # 1. Create synthetic recording shards
    device = torch.device("cpu")
    n_windows = 4
    data_train = np.random.randn(2, 5120 * n_windows).astype(np.float32)
    data_val = np.random.randn(2, 5120 * n_windows).astype(np.float32)

    path_train = tmp_path / "train.npy"
    path_val = tmp_path / "val.npy"
    np.save(path_train, data_train)
    np.save(path_val, data_val)

    train_windows = [
        RawWindow("P_dev", "R1", str(path_train), i, i * 20.0, (i + 1) * 20.0, i % 2, (0, 1))
        for i in range(n_windows)
    ]
    val_windows = [
        RawWindow("P_val", "R2", str(path_val), i, i * 20.0, (i + 1) * 20.0, i % 2, (0, 1))
        for i in range(n_windows)
    ]

    # 2. Train 1 fold of Stage 1 (CNN-BiLSTM)
    ckpt_s1 = tmp_path / "fold_1.pt"
    gt_events = {
        ("P_val", "R2"): [(10.0, 30.0)]
    }
    durations = {
        ("P_val", "R2"): 80.0
    }

    fold_s1_res = _fit_fold(
        train_windows,
        val_windows,
        canonical_channels=["C1", "C2"],
        epochs=1,
        batch_size=2,
        device=device,
        checkpoint=ckpt_s1,
        use_amp=False,
        normalization_cache={},
        ground_truth_events=gt_events,
        recording_durations=durations,
    )
    assert ckpt_s1.is_file()
    assert "event_evaluation" in fold_s1_res

    # 3. Extract temporal features from frozen Stage 1 model
    feat_dir = tmp_path / "features"
    mean = np.zeros(2, dtype=np.float32)
    std = np.ones(2, dtype=np.float32)

    train_feat_path, _, ordered_train = extract_features(
        ckpt_s1,
        train_windows,
        canonical_channels=["C1", "C2"],
        mean=mean,
        std=std,
        output_dir=feat_dir,
        split_name="train",
        fold=1,
        device=device,
        batch_size=2,
    )
    val_feat_path, _, ordered_val = extract_features(
        ckpt_s1,
        val_windows,
        canonical_channels=["C1", "C2"],
        mean=mean,
        std=std,
        output_dir=feat_dir,
        split_name="validation",
        fold=1,
        device=device,
        batch_size=2,
    )
    assert train_feat_path.is_file()
    assert val_feat_path.is_file()

    # Verify extracted shape [N, 128, 128]
    val_feats = np.load(val_feat_path)
    assert val_feats.shape == (n_windows, 128, 128)

    # 4. Train Stage 2 (Attention classifier)
    ckpt_attn = tmp_path / "attn_fold_1.pt"
    val_meta = [
        {"patient_id": w.patient_id, "recording_id": w.recording_id, "start_seconds": w.start_seconds, "end_seconds": w.end_seconds, "label": w.label}
        for w in ordered_val
    ]
    fold_attn_res = _fit_attention(
        train_feat_path,
        val_feat_path,
        np.array([w.label for w in ordered_train], dtype=np.float32),
        np.array([w.label for w in ordered_val], dtype=np.float32),
        device=device,
        checkpoint=ckpt_attn,
        epochs=1,
        batch_size=2,
        validation_metadata=val_meta,
        ground_truth_events=gt_events,
        recording_durations=durations,
    )
    assert ckpt_attn.is_file()
    assert "event_evaluation" in fold_attn_res

    # 5. Model comparison
    comp = compute_comparison(
        fold_s1_res["event_evaluation"],
        fold_attn_res["event_evaluation"],
        comparison_name="E2E Synthetic Test",
    )
    assert "window_level" in comp
    assert "event_level" in comp
    assert "delta_attention_minus_baseline" in comp["event_level"]["event_sensitivity"]
