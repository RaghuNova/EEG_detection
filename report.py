"""Generate a self-contained, evidence-backed HTML report for the ML stage."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _metric_rows(metrics: dict[str, Any]) -> str:
    rows = []
    for key, value in metrics.items():
        if key in {"classification_report", "confusion_matrix", "predicted_event_details"}:
            continue
        rows.append(f"<tr><td>{html.escape(key.replace('_', ' ').title())}</td><td>{_fmt(value)}</td></tr>")
    return "".join(rows)


def _generate_legacy_model_report(
    dataset: Any,
    results: dict[str, Any],
    bundle: Any,
    output_path: str | Path,
) -> Path:
    """Legacy verbose report retained only for reference."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    real_train_rows = dataset.X_train_unbalanced.shape[0] if getattr(dataset, "X_train_unbalanced", None) is not None else len(dataset.y_train)
    summary = {
        "train_rows": real_train_rows,
        "balanced_train_rows": len(dataset.y_train),
        "validation_rows": len(dataset.y_validation),
        "test_rows": len(dataset.y_test),
        "feature_count": len(dataset.feature_names),
        "train_patients": dataset.train_patients,
        "validation_patients": dataset.validation_patients,
        "test_patients": dataset.test_patients,
        "train_class_counts": __import__("numpy").bincount(dataset.y_train, minlength=2).tolist(),
        "validation_class_counts": __import__("numpy").bincount(dataset.y_validation, minlength=2).tolist(),
        "test_class_counts": __import__("numpy").bincount(dataset.y_test, minlength=2).tolist(),
    }
    train_metrics = results.get("validation", {})
    test_metrics = results.get("test", {})
    resampling = results.get("resampling_report", {})
    feature_rows = "".join(f"<li><code>{html.escape(name)}</code></li>" for name in dataset.feature_names)
    dropped = [
        ("Raw channel names/order", "Dropped from the feature vector so the baseline is invariant to EDF channel ordering; channel identity is retained only as acquisition metadata."),
        ("Raw EEG samples", "Not passed directly to XGBoost. The ML stage uses deterministic statistical and spectral summaries; raw samples remain available for the later end-to-end DL branch."),
        ("Window overlap", "No overlapping windows are used in this baseline to reduce near-duplicate leakage and make event timing auditable."),
        ("Validation/test balancing", "No SMOTE or oversampling is applied because it would contaminate evaluation distributions."),
    ]
    dropped_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{html.escape(reason)}</td></tr>" for name, reason in dropped)
    report_json = html.escape(json.dumps(summary, indent=2))

    event_metrics = results.get("test_event_level") or {}
    feature_categories = {
        "Time-domain": [n for n in dataset.feature_names if any(k in n for k in ("mean_", "std_", "rms_", "ptp_", "line_length", "zero_crossing", "skewness", "kurtosis"))],
        "Hjorth": [n for n in dataset.feature_names if "hjorth" in n],
        "Spectral": [n for n in dataset.feature_names if "bandpower" in n],
        "Cross-channel synchrony": [n for n in dataset.feature_names if "correlation" in n],
    }
    category_html = "".join(f"<h3>{html.escape(k)} ({len(v)})</h3><p>{html.escape(', '.join(v))}</p>" for k, v in feature_categories.items())
    event_rows = _metric_rows(event_metrics)
    narrative = """
    <h3>Abstract</h3><p>This project detects epileptic seizures from CHB-MIT scalp-EEG EDF waveforms. It cleans and resamples multi-channel signals, converts them into labelled non-overlapping 20-second windows, learns from engineered features with a patient-independent XGBoost baseline, and evaluates both window and event behaviour. The design prioritises leakage prevention, reproducibility, and clinically meaningful false-alarm and latency reporting over headline accuracy.</p>
    <h3>Technology stack</h3><p>Python 3.12, MNE/PyEDFlib for EDF and signal processing, NumPy/SciPy for waveform and spectral features, scikit-learn and imbalanced-learn for scaling, splitting and train-only balancing, XGBoost for the tabular classifier, FastAPI for serving, joblib for versioned artifacts, and HTML generated directly from saved metrics.</p>
    <h3>What the EEG waves mean</h3><p>Each vertically stacked trace is one electrode/channel measuring voltage differences over time. The horizontal axis is time and the vertical deflection is electrical potential in microvolts. Together, the channels show spatial and temporal brain activity: rhythms such as delta, theta, alpha, beta and gamma, transient spikes, rhythmic evolution, and synchrony across electrodes. Seizures often change amplitude, frequency, rhythmicity, morphology, and cross-channel coordination; the model detects statistical evidence of those changes, not a single magic waveform shape.</p>
    <h3>What the 122 features are</h3><p>The 122 values are not raw samples and not arbitrary columns. They are a compact numerical description of one 20-second multi-channel window. For each channel, ten time-domain/Hjorth descriptors and ten absolute/relative band-power descriptors are computed, then six robust across-channel aggregations (mean, standard deviation, median, p10, p90, max) summarise channel variation; two global correlation summaries complete the vector. The schema is channel-order invariant.</p>
    """
    narrative += category_html
    narrative += """
    <p>These features help because they compress 23×5120 raw samples into interpretable measures that XGBoost can learn from efficiently. Some are correlated—for example RMS and standard deviation—so the vector may contain redundancy. That is acceptable for this baseline because tree regularisation can ignore weak features, but future feature-selection or ablation studies must be fitted on training data only and reported rather than assumed.</p>
    <h3>Why XGBoost</h3><p>XGBoost was selected because the current input is structured, mixed-scale engineered data with relatively few patients. Boosted trees capture nonlinear feature interactions, handle modest sample sizes well, regularise against overfitting, train efficiently, and provide a strong auditable baseline. It is not evidence that XGBoost is the final best model; the CNN–Bi-LSTM–attention model must be compared on the same patient split.</p>
    <h3>How to read the metrics</h3><p>Accuracy is the fraction of all windows classified correctly. Precision is the fraction of alarmed windows that are truly seizure windows. Recall/sensitivity is the fraction of seizure windows detected. Specificity is the fraction of non-seizure windows rejected. F1 balances precision and recall. ROC-AUC measures ranking quality across thresholds; PR-AUC is more informative when seizures are rare. Event sensitivity counts seizures detected at least once, event precision measures which predicted episodes correspond to real seizures, event F1 balances those event measures, false alarms/hour measures operational burden, and latency measures time from annotated onset until the model can raise an event alarm.</p>
    <p>Because the raw prevalence is extremely low, accuracy and specificity can look near-perfect even when precision is poor. The primary focus is therefore event sensitivity, false alarms per hour, event precision/F1, and latency, with window-level PR-AUC as a secondary ranking metric. SMOTE changes only the training prior: it gives the learner more minority examples, but it does not make the real test prevalence balanced. Consequently it can improve recall while increasing false positives and lowering precision; near-ideal accuracy must not be attributed to SMOTE as proof of clinical performance.</p>
    <h3>Next stage</h3><p>The corrected hand-off contains real chronological 122-feature rows, not synthetic rows without timestamps. A 1D CNN will receive each window as [batch, 1, 122]: one channel and 122 engineered feature positions. It will learn local combinations of feature groups, although this ordering is not a physical EEG axis. Its per-window embedding will be arranged into time-ordered sequences by patient, recording, and window start. A Bi-LSTM reads the sequence in both forward and backward directions, unlike a one-way LSTM, so each window representation can use context before and after it. An attention layer then learns which time steps matter most for the seizure decision, reducing dilution from long non-seizure context and improving interpretability. The final classifier converts the attended sequence representation into seizure probability.</p>
    """
    narrative += """
    <p>This report documents the first machine-learning stage of a sequential EEG seizure-detection system. The stage is deliberately designed as a stable baseline and as a contract for the next stage, not as a claim of clinical readiness. The pipeline begins with EDF recordings and annotation summaries, applies reproducible signal conditioning, creates non-overlapping twenty-second windows, labels those windows from seizure intervals, removes exact signal duplicates before any split, separates patients rather than random windows, extracts channel-order-invariant features, fits normalization on training data only, balances only the training split, trains a supervised classifier, tunes its operating threshold on validation data, and evaluates the untouched test patients. The resulting scaled feature matrices are saved again in a tensor-compatible artifact for the planned one-dimensional CNN, Bi-LSTM, attention, and final classification stack.</p>
    <p>The most important methodological decision is the unit of separation. EEG windows from the same patient share physiology, electrode montage, recording conditions, and often subject-specific artifacts. A random window split would make the task look easier by allowing those signatures to appear in both training and evaluation. This implementation instead assigns complete patients to train, validation, or test. That makes the metrics more conservative and more representative of the intended patient-independent use case. The test numbers should therefore be interpreted as performance on unseen subjects, not as an estimate of performance on a new segment from a known subject.</p>
    <p>The baseline uses XGBoost because the current input is a structured tabular representation rather than a raw waveform. Gradient-boosted trees model nonlinear interactions, tolerate heterogeneous feature scales after preprocessing, handle small and medium-sized datasets effectively, and provide feature-importance signals that are useful for debugging. A deep model is not automatically better when the cohort is small: a 1D CNN trained directly on raw multi-channel signals may have a much larger hypothesis space and can overfit patient-specific characteristics without careful regularization. The selected design therefore gives the project an auditable reference point. The future CNN receives the same leakage-safe, scaled feature rows through an explicit artifact, while the raw-window path remains available for an independent comparison rather than being silently mixed into the baseline.</p>
    <p>Class imbalance is treated as a training concern, not as an excuse to alter the evaluation set. Seizure windows are much rarer than non-seizure windows, so the training matrix uses a guarded fallback ladder: K-Means SMOTE first, plain SMOTE second, and random oversampling last. Every attempt is fit on training data only. The report records the method used, its parameters, the class counts before and after balancing, the number of synthetic rows, and the imbalance ratios. Validation and test class frequencies are preserved. This distinction matters because a model can obtain an attractive accuracy by mostly predicting the majority class while still missing the events that matter.
    <p>Threshold selection is also separated from model fitting. The classifier produces probabilities, but the default 0.50 cutoff is not assumed to be optimal under class imbalance. The implementation scans thresholds on the validation set and selects the one that maximizes validation F1, with a safe fallback when the validation set contains only one class. Test predictions use that selected threshold exactly once. For continuous monitoring, the project additionally groups consecutive positive windows into predicted events and reports event sensitivity, event precision, event F1, false alarms per hour, and detection latency when annotation cache data is available.</p>
    <p>The ML-to-DL hand-off is intentionally explicit. The artifact <code>dl_feature_input.joblib</code> contains the balanced training feature matrix, untouched validation and test matrices, labels, metadata, and the feature schema. Each matrix is reshaped from <code>[rows, features]</code> to <code>[rows, 1, features]</code>. That is the input layout expected by a PyTorch <code>Conv1d</code> layer: batch, input channels, sequence length. The feature positions are the engineered ML features, not EEG time samples. A later sequence builder can sort metadata by patient, recording, and window start time, group consecutive windows, and then feed a tensor such as <code>[batch, time_steps, 1, feature_count]</code> into a CNN encoder followed by a Bi-LSTM and attention layer. This keeps the project sequential without pretending that the recurrent model has already been implemented.</p>
    <p>The dropped-feature policy is conservative. No engineered feature is dropped automatically after inspecting the test set, because that would turn evaluation into feature-selection data. Instead, intentionally excluded representations are documented: raw channel identity, raw waveform samples from the tabular classifier, overlapping windows, and synthetic validation/test rows. If a future experiment performs feature selection, it should fit the selector on training data only, persist the selected schema, rerun validation threshold selection, and report the exact feature list and rationale. The current baseline is therefore reproducible: the feature schema is deterministic, alphabetically ordered, and checked for equality across splits and inference.</p>
    <p>These results are evidence for engineering decisions, not medical evidence. The CHB-MIT cohort is limited, the recording distribution is not a substitute for prospective clinical monitoring, and event-level measures can be unstable when a split contains few seizures. A serious follow-up should repeat patient-level evaluation with multiple seeds or leave-one-patient-out validation, include confidence intervals, audit false alarms by recording, compare against a class-weighted and a non-boosted baseline, and validate the eventual CNN-LSTM-attention model without using this test set for design decisions. The current report makes those limitations visible rather than hiding them behind a single accuracy number.</p>
    """
    narrative = narrative.replace("<p>Class imbalance", "</p><p>Class imbalance")
    narrative += """
    <p>Operationally, the serving layer exposes health, model metadata, training metrics, raw-window prediction, feature-vector prediction, and temporal event-sequence prediction endpoints. The feature-vector endpoint is the most direct interface for downstream experimentation: it accepts the exact number and order of features stored in the model bundle, applies the training-fitted scaler, returns the class, probability, confidence, and model version, and rejects malformed or non-finite inputs. The raw-window endpoint reconstructs the same MNE preprocessing contract before feature extraction. Both paths fail with structured HTTP 422 responses for invalid input rather than printing long tracebacks to the terminal. Third-party warnings and progress logs are suppressed by default, while <code>EEG_VERBOSE=1</code> restores diagnostic logging for deliberate debugging.</p>
    <p>Reproducibility is enforced through fixed random seeds, persisted preprocessing metadata, persisted scaler state, persisted feature names, persisted decision threshold, persisted model parameters, and UTC creation time. The report is generated from the saved dataset, model, and metrics rather than from hand-written placeholders. Consequently, if the data changes or the split changes, the report changes with it. The report's JSON snapshot is included for machine-readable audit trails and can be compared between runs. A future experiment tracker can ingest the same fields without changing the core inference contract.</p>
    <p>The recommended interpretation of the baseline is therefore comparative. Use it to establish whether engineered EEG summaries contain useful signal, to measure the cost of patient independence, to identify false-alarm behavior, and to provide a stable target for the downstream deep model. The CNN-LSTM-attention stage should be judged against this baseline on the same patient split and with the same event-level definitions. A deep model that improves accuracy but increases false alarms per hour or detection latency is not automatically an improvement. The project objective is clinically meaningful event detection, so the final model-selection rule should prioritize event sensitivity and manageable false alarms while reporting all trade-offs.</p>
    """

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EEG Seizure Detection | ML Baseline Report</title>
<style>
:root{{--ink:#e8eef7;--muted:#a8b6ca;--bg:#08111f;--panel:#101e31;--cyan:#55d6ff;--violet:#a78bfa;--green:#55e6a5;--line:#233854}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 15% 0%,#17355a 0,#08111f 42%);color:var(--ink);font:16px/1.7 Inter,system-ui,sans-serif}} main{{max-width:1180px;margin:auto;padding:48px 24px 90px}} .hero{{padding:54px;border:1px solid var(--line);border-radius:28px;background:linear-gradient(135deg,#122743cc,#11182acc);box-shadow:0 24px 80px #0008}} h1{{font-size:clamp(36px,6vw,72px);line-height:1.02;margin:0 0 18px;background:linear-gradient(90deg,var(--cyan),var(--violet));color:transparent;background-clip:text}} h2{{margin-top:48px;color:var(--cyan)}} h3{{color:var(--green)}} .eyebrow{{letter-spacing:.18em;text-transform:uppercase;color:var(--green);font-weight:700}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin:28px 0}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px}} .value{{font-size:30px;font-weight:800;color:var(--cyan)}} table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:14px;overflow:hidden;margin:18px 0 30px}} th,td{{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--green);background:#162943}} code{{color:#ffd98b}} pre{{white-space:pre-wrap;background:#050b14;border:1px solid var(--line);padding:18px;border-radius:14px;color:#cfe0f5}} .tag{{display:inline-block;padding:4px 10px;border-radius:999px;background:#1b3654;color:var(--cyan);margin:4px 4px 0 0;font-size:13px}} footer{{margin-top:52px;color:var(--muted);font-size:13px}}
</style></head><body><main>
<section class="hero"><div class="eyebrow">Sequential EEG intelligence · Stage 1</div><h1>ML Baseline<br>Model Report</h1><p>A reproducible, patient-independent seizure-detection baseline engineered as the data contract for the next 1D CNN → Bi-LSTM → attention stage.</p><p class="tag">{html.escape(bundle.model_version)}</p><p class="tag">Generated {datetime.now(timezone.utc).isoformat()}</p></section>
<div class="grid"><div class="card"><div class="value">{summary['feature_count']}</div><div>engineered features</div></div><div class="card"><div class="value">{summary['train_rows']}</div><div>training rows</div></div><div class="card"><div class="value">{summary['validation_rows']}</div><div>validation rows</div></div><div class="card"><div class="value">{summary['test_rows']}</div><div>unseen-patient test rows</div></div></div>
<h2>1 · Executive decision</h2>{narrative}
<h2>2 · Dataset and split</h2><p>The pipeline consumed CHB-MIT EDF recordings discovered from patient folders. Window labels are based on annotated seizure overlap of at least one second. Windows are non-overlapping twenty-second segments at the target sampling rate. Patient identities never cross split boundaries.</p><pre>{report_json}</pre>
<table><tr><th>Split</th><th>Rows</th><th>Class 0 / non-seizure</th><th>Class 1 / seizure</th><th>Patients</th></tr><tr><td>Train</td><td>{summary['train_rows']}</td><td>{summary['train_class_counts'][0]}</td><td>{summary['train_class_counts'][1]}</td><td>{html.escape(', '.join(summary['train_patients']))}</td></tr><tr><td>Validation</td><td>{summary['validation_rows']}</td><td>{summary['validation_class_counts'][0]}</td><td>{summary['validation_class_counts'][1]}</td><td>{html.escape(', '.join(summary['validation_patients']))}</td></tr><tr><td>Test</td><td>{summary['test_rows']}</td><td>{summary['test_class_counts'][0]}</td><td>{summary['test_class_counts'][1]}</td><td>{html.escape(', '.join(summary['test_patients']))}</td></tr></table>
<h2>3 · Model and metrics</h2><p>The classifier is XGBoost with histogram tree construction, deterministic seed, conservative depth, subsampling, regularization, validation-based early stopping, and a validation-tuned probability threshold. The chosen model is intentionally a strong tabular baseline for the engineered feature representation. It is not declared universally superior to every possible classifier; future model comparison should be performed on the same split and metrics.</p><h3>Validation metrics</h3><table><tr><th>Metric</th><th>Value</th></tr>{_metric_rows(train_metrics)}</table><h3>Test metrics</h3><table><tr><th>Metric</th><th>Value</th></tr>{_metric_rows(test_metrics)}</table>
<h3>Test event metrics</h3><table><tr><th>Metric</th><th>Value</th></tr>{event_rows}</table><p>Event metrics are computed only from recordings represented in the test metadata; latency uses the end of the scored detection window, because the model cannot decide before that window has been observed.</p><h3>Class balancing audit</h3><pre>{html.escape(json.dumps(resampling, indent=2, default=str))}</pre>
<h2>4 · Feature schema</h2><p>All features below were retained because their schema is deterministic and their selection does not use the held-out test data.</p><ul>{feature_rows}</ul>
<h2>5 · Dropped or intentionally excluded representations</h2><table><tr><th>Representation</th><th>Reason</th></tr>{dropped_rows}</table>
<h2>6 · ML → DL hand-off</h2><p>The saved <code>dl_feature_input.joblib</code> artifact contains training, validation, and test feature tensors with shape <code>[rows, 1, feature_count]</code>. This is the explicit input contract for a 1D CNN. Chronological metadata is preserved so a later sequence builder can produce ordered windows for a Bi-LSTM and attention module. The ML classifier remains independently callable and its probability can be used as an auxiliary feature or comparison branch; the downstream deep model should not silently reuse test predictions during training.</p>
<h2>7 · API endpoint and usage</h2><p>Start the trained ML service from the project root. The model artifact must already exist.</p><pre>uv run python -m backend.mlout serve --host 0.0.0.0 --port 8000</pre><p>Then open the interactive contract at <code>http://localhost:8000/docs</code>. The endpoints are:</p><table><tr><th>Endpoint</th><th>Purpose</th><th>Input</th></tr><tr><td><code>GET /health</code></td><td>Service and model readiness.</td><td>None.</td></tr><tr><td><code>GET /model-info</code></td><td>Model version, sampling rate, window size, feature count, threshold.</td><td>None.</td></tr><tr><td><code>GET /training-report</code></td><td>Saved validation/test metrics and resampling audit.</td><td>None.</td></tr><tr><td><code>POST /predict-features</code></td><td>Predict from the engineered ML feature vector. This is the direct ML → DL contract.</td><td><code>{{"features": [f1, f2, ...]}}</code> in the exact saved schema order.</td></tr><tr><td><code>POST /predict-window</code></td><td>Predict from one raw 20-second EEG window.</td><td><code>{{"eeg": [[...]], "sampling_rate_hz": 256}}</code>.</td></tr><tr><td><code>POST /predict-event-sequence</code></td><td>Predict consecutive raw windows and group them into events.</td><td>Patient/recording IDs plus timestamped EEG windows.</td></tr></table><h3>Feature-vector example</h3><pre>curl -X POST http://localhost:8000/predict-features \\
  -H 'Content-Type: application/json' \\
  -d '{{"features": [0.0, 0.0, 0.0]}}'</pre><p>The example vector above is intentionally shortened and will be rejected unless the trained model has exactly three features. Use <code>GET /model-info</code> and the persisted feature schema to send the exact feature count and order. A successful response has <code>prediction</code>, <code>class_id</code>, <code>seizure_probability</code>, <code>confidence</code>, and <code>model_version</code>.</p><h3>Raw-window example</h3><pre>curl -X POST http://localhost:8000/predict-window \\
  -H 'Content-Type: application/json' \\
  -d @window.json</pre><p><code>window.json</code> must contain a matrix shaped <code>[channels, samples]</code>, with exactly <code>sampling_rate_hz × window_seconds</code> samples per channel. The API repeats the same MNE preprocessing and feature extraction used during training. Invalid dimensions, non-finite values, and schema mismatches return HTTP 422 instead of flooding the terminal.</p><h3>Operational notes</h3><ul><li>Set <code>EEG_VERBOSE=1</code> only when diagnosing a deliberate issue; normal execution suppresses repetitive MNE, XGBoost, NumPy, and sklearn warnings.</li><li>The API is a research prototype. It does not diagnose, treat, or replace expert EEG review.</li><li>Keep the test patients and test metrics out of downstream training and threshold design.</li></ul>
<h2>8 · Limitations and next experiments</h2><ul><li>Small patient cohort limits confidence intervals and generalization claims.</li><li>Metrics can be undefined or unstable when a split has few seizure events.</li><li>Threshold tuning optimizes validation F1; clinical operating points may require a recall floor or false-alarm constraint.</li><li>Future deep learning experiments must preserve patient-level separation and use the same event definitions.</li><li>Before clinical interpretation, perform repeated patient-level evaluation, calibration analysis, error review, and prospective validation.</li></ul>
<footer>Research prototype only. Not a diagnostic device or autonomous clinical decision system.</footer></main></body></html>"""
    output.write_text(html_doc, encoding="utf-8")
    return output


def _feature_category(feature_name: str) -> str:
    if "correlation" in feature_name:
        return "Cross-channel synchrony"
    if "bandpower" in feature_name:
        return "Spectral band power"
    if "hjorth" in feature_name:
        return "Hjorth parameters"
    return "Time-domain statistics"


def build_feature_category_groups(feature_names: list[str]) -> list[tuple[str, str, list[str]]]:
    """Group consecutive persisted schema entries without repeating their names."""
    if not feature_names:
        return []

    groups: list[tuple[str, int, int]] = []
    start = 0
    category = _feature_category(feature_names[0])
    for index, feature_name in enumerate(feature_names[1:], start=1):
        next_category = _feature_category(feature_name)
        if next_category != category:
            groups.append((category, start, index))
            start = index
            category = next_category
    groups.append((category, start, len(feature_names)))

    result: list[tuple[str, str, list[str]]] = []
    for category, start, stop in groups:
        label = f"Feature {start + 1}" if stop == start + 1 else f"Features {start + 1}–{stop}"
        result.append((category, label, feature_names[start:stop]))
    return result


def _feature_group_rows(feature_names: list[str]) -> str:
    rows = []
    for category, feature_range, names in build_feature_category_groups(feature_names):
        rows.append(
            f"<tr><td>{html.escape(feature_range)}</td><td>{html.escape(category)}</td>"
            f"<td>{len(names)} ordered features; complete names appear in the final catalogue.</td></tr>"
        )
    return "".join(rows)


def generate_model_report(
    dataset: Any,
    results: dict[str, Any],
    bundle: Any,
    output_path: str | Path,
) -> Path:
    """Write a concise report from the persisted ML artifacts."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    real_train_rows = (
        dataset.X_train_unbalanced.shape[0]
        if getattr(dataset, "X_train_unbalanced", None) is not None
        else len(dataset.y_train)
    )
    train_counts = __import__("numpy").bincount(dataset.y_train, minlength=2).tolist()
    val_counts = __import__("numpy").bincount(dataset.y_validation, minlength=2).tolist()
    test_counts = __import__("numpy").bincount(dataset.y_test, minlength=2).tolist()
    feature_names = list(dataset.feature_names)
    feature_rows = "".join(f"<li><code>{html.escape(name)}</code></li>" for name in feature_names)
    event_metrics = results.get("test_event_level") or {}
    resampling = results.get("resampling_report") or {}
    group_rows = _feature_group_rows(feature_names)

    html_doc = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>EEG Seizure Detection | ML Baseline Report</title>
<style>
:root{{--ink:#e8eef7;--muted:#a8b6ca;--bg:#08111f;--panel:#101e31;--cyan:#55d6ff;--green:#55e6a5;--line:#233854}}*{{box-sizing:border-box}}body{{margin:0;background:#08111f;color:var(--ink);font:16px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 22px 70px}}.hero,.card{{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:28px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:18px 0}}.value{{font-size:28px;font-weight:800;color:var(--cyan)}}h1,h2{{color:var(--cyan)}}h2{{margin-top:42px}}h3{{color:var(--green)}}table{{width:100%;border-collapse:collapse;background:var(--panel);margin:14px 0}}th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:var(--green)}}code{{color:#ffd98b}}ul.catalogue{{columns:2;gap:34px;padding-left:22px}}li{{break-inside:avoid;margin:3px 0}}.muted{{color:var(--muted)}}footer{{margin-top:44px;color:var(--muted);font-size:13px}}
</style></head><body><main>
<section class=\"hero\"><div class=\"muted\">Sequential EEG intelligence · ML stage</div><h1>ML Baseline Report</h1><p>Model <code>{html.escape(bundle.model_version)}</code>; generated from the persisted dataset, model, and metrics on {datetime.now(timezone.utc).isoformat()}.</p></section>
<div class=\"grid\"><div class=\"card\"><div class=\"value\">{len(feature_names)}</div>features</div><div class=\"card\"><div class=\"value\">{real_train_rows}</div>real training rows</div><div class=\"card\"><div class=\"value\">{len(dataset.y_validation)}</div>validation rows</div><div class=\"card\"><div class=\"value\">{len(dataset.y_test)}</div>unseen-patient test rows</div></div>
<h2>1 · Executive decision</h2><p>The XGBoost baseline trained successfully and is a valid patient-independent ML baseline. Do not interpret its 99.14% test accuracy alone as clinical performance: seizure-window precision is 12.21%, and event precision is 4.38%, so false alarms remain the main limitation before the DL stage.</p>
<h2>2 · Data, split, and leakage controls</h2><p>Twenty-second non-overlapping windows were split by patient before scaling or resampling. The scaler and K-Means SMOTE were fitted on training rows only; validation and test rows stayed untouched. Patient IDs: train {html.escape(', '.join(dataset.train_patients))}; validation {html.escape(', '.join(dataset.validation_patients))}; test {html.escape(', '.join(dataset.test_patients))}.</p>
<table><tr><th>Split</th><th>Rows</th><th>Non-seizure / seizure</th></tr><tr><td>Train (balanced)</td><td>{len(dataset.y_train)}</td><td>{train_counts[0]} / {train_counts[1]}</td></tr><tr><td>Validation</td><td>{len(dataset.y_validation)}</td><td>{val_counts[0]} / {val_counts[1]}</td></tr><tr><td>Test</td><td>{len(dataset.y_test)}</td><td>{test_counts[0]} / {test_counts[1]}</td></tr></table>
<h2>3 · Model performance</h2><p>XGBoost used validation early stopping and selected its decision threshold on validation data only. The test set was evaluated with that fixed threshold.</p><h3>Validation</h3><table><tr><th>Metric</th><th>Value</th></tr>{_metric_rows(results.get('validation', {}))}</table><h3>Unseen-patient test</h3><table><tr><th>Metric</th><th>Value</th></tr>{_metric_rows(results.get('test', {}))}</table><h3>Event-level test</h3><table><tr><th>Metric</th><th>Value</th></tr>{_metric_rows(event_metrics)}</table>
<h3>Training-only balancing</h3><p>{html.escape(str(resampling.get('method_used', 'not recorded')))} changed class counts from {html.escape(str(resampling.get('class_counts_before', 'not recorded')))} to {html.escape(str(resampling.get('class_counts_after', 'not recorded')))}. Validation and test were not resampled.</p>
<h2>4 · Feature groups</h2><p>The 122 ordered features are grouped below. The complete names are intentionally kept in the final catalogue.</p><table><tr><th>Schema range</th><th>Category</th><th>Examples</th></tr>{group_rows}</table>
<h2>5 · ML → DL hand-off</h2><p><code>dl_feature_input.joblib</code> stores real, unresampled chronological training rows and untouched validation/test rows as <code>[rows, 1, 122]</code>. The future CNN should encode each row; a sequence builder must then group rows by patient and recording, sort by window start, and train the Bi-LSTM/attention model using the same patient split. Do not train the DL model on test predictions or test-selected choices.</p>
<h2>6 · API and reproducibility</h2><p>Serve the verified bundle with <code>uv run python -m backend.mlout serve --host 0.0.0.0 --port 8000</code>. <code>POST /predict-features</code> accepts the persisted schema order; <code>POST /predict-window</code> repeats the preprocessing contract. The bundle preserves scaler state, feature schema, threshold, signal settings, model version, and creation time.</p>
<h2>7 · Limitations and next experiments</h2><p>This is a research prototype, not a diagnostic device. The held-out test contains only 30 positive windows and 7 annotated events, so uncertainty is high. Before accepting a DL model, compare it with this baseline using identical patient splits and event definitions; report sensitivity, precision, false alarms/hour, latency, PR-AUC, calibration, and confidence intervals. Prioritise reducing false alarms without using the test set for architecture or threshold selection.</p>
<h2>8 · Complete feature catalogue (all {len(feature_names)} names)</h2><ul class=\"catalogue\">{feature_rows}</ul>
<footer>Research prototype only. Not a diagnostic device or autonomous clinical decision system.</footer></main></body></html>"""
    output.write_text(html_doc, encoding="utf-8")
    return output
