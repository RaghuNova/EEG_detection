from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from backend import report


def test_feature_categories_follow_the_persisted_schema_order() -> None:
    names = [
        "channel_mean_bandpower_alpha",
        "channel_max_bandpower_theta",
        "channel_mean_hjorth_complexity",
        "channel_max_hjorth_mobility",
        "channel_mean_kurtosis",
        "channel_max_zero_crossing_rate",
        "global_mean_abs_channel_correlation",
        "global_max_abs_channel_correlation",
    ]

    categories = report.build_feature_category_groups(names)

    assert categories == [
        ("Spectral band power", "Features 1–2", names[:2]),
        ("Hjorth parameters", "Features 3–4", names[2:4]),
        ("Time-domain statistics", "Features 5–6", names[4:6]),
        ("Cross-channel synchrony", "Features 7–8", names[6:]),
    ]


def test_report_places_complete_feature_catalogue_last(tmp_path) -> None:
    names = ["feature_a", "feature_b", "feature_c"]
    dataset = SimpleNamespace(
        X_train_unbalanced=np.zeros((2, 3), dtype=np.float32),
        y_train=np.array([0, 1]),
        y_validation=np.array([0, 1]),
        y_test=np.array([0, 1]),
        feature_names=names,
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

    output = report.generate_model_report(dataset, results, bundle, tmp_path / "report.html")
    contents = output.read_text(encoding="utf-8")

    assert "Feature groups" in contents
    assert contents.find("feature_a") > contents.find("Limitations and next experiments")
    assert contents.rfind("feature_c") < contents.find("</footer>")
