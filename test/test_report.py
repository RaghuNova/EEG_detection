from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.report import generate_model_report


def test_model_report_is_html_and_documents_api(tmp_path: Path) -> None:
    dataset = SimpleNamespace(
        y_train=np.array([0, 1]),
        y_validation=np.array([0, 1]),
        y_test=np.array([0, 1]),
        feature_names=["feature_a", "feature_b"],
        train_patients=["CHB01"],
        validation_patients=["CHB02"],
        test_patients=["CHB03"],
    )
    bundle = SimpleNamespace(model_version="test-model")
    results = {
        "validation": {"accuracy": 0.5},
        "test": {"accuracy": 0.5},
        "resampling_report": {},
    }
    output = generate_model_report(dataset, results, bundle, tmp_path / "report.html")
    contents = output.read_text(encoding="utf-8")
    assert output.suffix == ".html"
    assert "POST /predict-features" in contents
    assert "uv run python -m backend.mlout serve" in contents
