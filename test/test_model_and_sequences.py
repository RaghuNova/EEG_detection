import numpy as np
import torch

from backend.dl.model import CNNBiLSTM
from backend.dl.raw_data import RawEEGDataset, RawWindow
from backend.dl.train import select_device


def _window() -> RawWindow:
    return RawWindow(
        patient_id="p1",
        recording_id="r1",
        path="unused",
        window_id=0,
        start_seconds=0.0,
        end_seconds=20.0,
        label=1,
        channel_indices=(0, 1),
    )


def test_raw_dataset_contract_is_channel_by_sample() -> None:
    dataset = RawEEGDataset([_window()], np.zeros(2), np.ones(2))
    assert dataset.windows[0].label == 1
    assert dataset.windows[0].end_seconds == 20.0


def test_model_returns_one_logit_and_supports_backward() -> None:
    model = CNNBiLSTM(2, recurrent_length=8)
    logits = model(torch.zeros((2, 2, 5120)))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    logits.sum().backward()


def test_cuda_is_selected_when_requested_and_available() -> None:
    if torch.cuda.is_available():
        assert select_device("cuda").type == "cuda"
    else:
        try:
            select_device("cuda")
        except RuntimeError as exc:
            assert "CUDA" in str(exc)
        else:
            raise AssertionError("CUDA request must fail when CUDA is unavailable")
