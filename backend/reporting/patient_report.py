"""patient_report.py
------------------
Clinical-grade PDF report generator for NEUROSCAN EEG Intelligence Platform.

Generates a multi-page PDF using ReportLab with:
- NEUROSCAN Cover & Header
- Patient & Analysis Provenance
- Preprocessing & Montage Information
- Model 1 (XGBoost) Inference Summary & Detections
- Model 2 (CNN-BiLSTM) Deep Temporal Analysis
- Model 3 (AttentionClassifier) Dynamic Thresholding & Persistent Episodes
- Multi-Model Comparison & Concordance Table
- Seizure Event Catalog
- Comprehensive Research & Medical Disclaimer
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas:
    """Canvas that performs two passes to render 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        from reportlab.pdfgen import canvas
        self._canvas_class = canvas.Canvas
        self.canvas_args = args
        self.canvas_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        class _InnerCanvas(self._canvas_class):
            def __init__(self, *c_args, **c_kwargs):
                super().__init__(*c_args, **c_kwargs)
                self._saved_page_states = []

            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                num_pages = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self.draw_footer(num_pages)
                    super().showPage()
                super().save()

            def draw_footer(self, page_count):
                self.saveState()
                self.setFont("Helvetica", 8)
                self.setFillColor(colors.HexColor("#64748B"))
                # Header rule on later pages
                if self._pageNumber > 1:
                    self.setStrokeColor(colors.HexColor("#CBD5E1"))
                    self.setLineWidth(0.5)
                    self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
                    self.drawString(54, letter[1] - 35, "NEUROSCAN — EEG Intelligence & Seizure Analysis Platform")
                    self.drawRightString(letter[0] - 54, letter[1] - 35, "Inference Intelligence Report")

                # Footer
                self.setStrokeColor(colors.HexColor("#E2E8F0"))
                self.setLineWidth(0.5)
                self.line(54, 45, letter[0] - 54, 45)
                self.drawString(
                    54, 32,
                    "Confidential — Medical Research & Clinical Decision Support. Not a Standalone Diagnosis."
                )
                self.drawRightString(letter[0] - 54, 32, f"Page {self._pageNumber} of {page_count}")
                self.restoreState()

        return _InnerCanvas(*args, **kwargs)


def generate_patient_pdf_report(
    analysis_data: Dict[str, Any],
    output_path: Optional[Path | str] = None,
) -> bytes:
    """Compile analysis JSON into a PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer if output_path is None else str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#0F172A")
    accent_blue = colors.HexColor("#2563EB")
    accent_cyan = colors.HexColor("#0891B2")
    accent_purple = colors.HexColor("#7C3AED")
    dark_gray = colors.HexColor("#334155")
    light_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=primary_color,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#64748B"),
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=dark_gray,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=dark_gray,
    )
    bold_style = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=body_style,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0284C7"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("NEUROSCAN", title_style))
    story.append(Paragraph("EEG Intelligence & Multi-Model Seizure Detection Platform", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=accent_blue, spaceBefore=4, spaceAfter=14))

    # 2. Provenance Meta Block
    patient_id = analysis_data.get("patient_id", "UNKNOWN")
    analysis_id = analysis_data.get("analysis_id", "N/A")
    created_at = analysis_data.get("created_at", datetime.now(timezone.utc).isoformat())

    meta_table_data = [
        [
            Paragraph("<b>Patient Identifier:</b>", body_style),
            Paragraph(str(patient_id), bold_style),
            Paragraph("<b>Analysis Session ID:</b>", body_style),
            Paragraph(str(analysis_id)[:18] + "...", body_style),
        ],
        [
            Paragraph("<b>Date of Analysis:</b>", body_style),
            Paragraph(str(created_at).replace("T", " ")[:19] + " UTC", body_style),
            Paragraph("<b>Report Classification:</b>", body_style),
            Paragraph("<b>INFERENCE REPORT</b> (Unlabeled Cohort)", badge_style),
        ],
        [
            Paragraph("<b>Total Recordings:</b>", body_style),
            Paragraph(f"{analysis_data.get('total_recordings', 0)} files", body_style),
            Paragraph("<b>Total EEG Duration:</b>", body_style),
            Paragraph(f"{analysis_data.get('total_duration_seconds', 0.0):.1f} seconds ({analysis_data.get('total_windows', 0)} windows)", body_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[110, 140, 120, 134])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_bg),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Patient-Level Executive Summary
    story.append(Paragraph("Patient-Level Multi-Model Summary", h1_style))
    summary = analysis_data.get("summary", {})

    summary_table_data = [
        [
            Paragraph("<b>Model Architecture</b>", bold_style),
            Paragraph("<b>Input Representation</b>", bold_style),
            Paragraph("<b>Threshold Used</b>", bold_style),
            Paragraph("<b>Positive Windows</b>", bold_style),
            Paragraph("<b>Detected Seizure Events</b>", bold_style),
        ],
        [
            Paragraph("<b>1. ML Classifier</b><br/><font color='#64748B'>XGBoost + Scaler</font>", body_style),
            Paragraph("122 Channel-Invariant Statistical & Spectral Features", body_style),
            Paragraph("<b>0.2790</b> (Fixed F1 Tuned)", body_style),
            Paragraph(str(summary.get("ml_positive_windows", 0)), bold_style),
            Paragraph(str(summary.get("ml_total_events", 0)), bold_style),
        ],
        [
            Paragraph("<b>2. CNN-BiLSTM</b><br/><font color='#64748B'>Raw 1D Conv + Recurrent</font>", body_style),
            Paragraph("22 Canonical Bipolar Channels (5120 samples @ 256Hz)", body_style),
            Paragraph("<b>0.5000</b> (Standard Logistic)", body_style),
            Paragraph(str(summary.get("cnn_positive_windows", 0)), bold_style),
            Paragraph(str(summary.get("cnn_total_events", 0)), bold_style),
        ],
        [
            Paragraph("<b>3. AttentionClassifier</b><br/><font color='#64748B'>4-Head Attention + Residual</font>", body_style),
            Paragraph("Recurrent Temporal Features [128, 128] from CNN-BiLSTM", body_style),
            Paragraph("<b>Dynamic Adaptive</b><br/><font size='7' color='#7C3AED'>Hybrid Quantile Floor 0.03</font>", body_style),
            Paragraph(str(summary.get("attention_positive_windows", 0)), bold_style),
            Paragraph(str(summary.get("attention_total_events", 0)), bold_style),
        ],
    ]
    summary_table = Table(summary_table_data, colWidths=[120, 140, 100, 70, 74])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), primary_color),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (3, 1), (-1, -1), "CENTER"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # 4. Recording-Wise Breakdown
    story.append(Paragraph("EDF Recording Breakdown & Concordance", h1_style))
    recordings = analysis_data.get("recordings", [])

    rec_table_data = [
        [
            Paragraph("<b>Recording File</b>", bold_style),
            Paragraph("<b>Duration</b>", bold_style),
            Paragraph("<b>Windows</b>", bold_style),
            Paragraph("<b>ML Positive</b>", bold_style),
            Paragraph("<b>CNN Positive</b>", bold_style),
            Paragraph("<b>Attention Positive</b>", bold_style),
            Paragraph("<b>Calibrated Th.</b>", bold_style),
        ]
    ]

    for rec in recordings:
        if rec.get("status") == "completed":
            models = rec.get("models", {})
            ml_pos = models.get("ml", {}).get("positive_window_count", 0)
            cnn_pos = models.get("cnn_bilstm", {}).get("positive_window_count", 0)
            attn_pos = models.get("attention", {}).get("positive_window_count", 0)
            attn_th = models.get("attention", {}).get("calibrated_threshold", 0.03)

            rec_table_data.append([
                Paragraph(f"<b>{rec.get('filename')}</b>", body_style),
                Paragraph(f"{rec.get('duration_seconds', 0.0):.0f}s", body_style),
                Paragraph(str(rec.get("total_windows", 0)), body_style),
                Paragraph(str(ml_pos), body_style),
                Paragraph(str(cnn_pos), body_style),
                Paragraph(str(attn_pos), body_style),
                Paragraph(f"{attn_th:.4f}", body_style),
            ])
        else:
            rec_table_data.append([
                Paragraph(f"<b>{rec.get('filename')}</b>", body_style),
                Paragraph("FAILED", ParagraphStyle("Err", parent=body_style, textColor=colors.red)),
                Paragraph("0", body_style),
                Paragraph("-", body_style),
                Paragraph("-", body_style),
                Paragraph("-", body_style),
                Paragraph("-", body_style),
            ])

    rec_table = Table(rec_table_data, colWidths=[130, 60, 50, 64, 64, 76, 60])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 1, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 14))

    # 5. Detected Seizure Events Catalog
    story.append(Paragraph("Detected Seizure Event Catalog", h1_style))
    all_events: List[Dict[str, Any]] = []

    for rec in recordings:
        if rec.get("status") == "completed":
            for model_key, model_label in [
                ("ml", "ML Model"),
                ("cnn_bilstm", "CNN-BiLSTM"),
                ("attention", "AttentionClassifier"),
            ]:
                for evt in rec.get("models", {}).get(model_key, {}).get("events", []):
                    all_events.append({
                        "recording": rec.get("filename"),
                        "model": model_label,
                        "event_id": evt.get("event_id"),
                        "start_seconds": evt.get("start_seconds"),
                        "end_seconds": evt.get("end_seconds"),
                        "duration_seconds": evt.get("duration_seconds"),
                        "max_probability": evt.get("max_probability"),
                    })

    if all_events:
        event_table_data = [
            [
                Paragraph("<b>Recording</b>", bold_style),
                Paragraph("<b>Model</b>", bold_style),
                Paragraph("<b>Start Time</b>", bold_style),
                Paragraph("<b>End Time</b>", bold_style),
                Paragraph("<b>Duration</b>", bold_style),
                Paragraph("<b>Peak Prob.</b>", bold_style),
            ]
        ]
        # Sort chronologically by recording and start time
        all_events.sort(key=lambda x: (x["recording"], x["start_seconds"]))
        for e in all_events[:25]:  # limit to top 25 for page fit
            event_table_data.append([
                Paragraph(e["recording"], body_style),
                Paragraph(e["model"], body_style),
                Paragraph(f"{e['start_seconds']:.1f}s", body_style),
                Paragraph(f"{e['end_seconds']:.1f}s", body_style),
                Paragraph(f"{e['duration_seconds']:.1f}s", body_style),
                Paragraph(f"{e['max_probability']:.4f}", body_style),
            ])

        event_table = Table(event_table_data, colWidths=[120, 110, 68, 68, 68, 70])
        event_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 1, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ]))
        story.append(event_table)
    else:
        story.append(Paragraph("<i>No seizure alarm episodes detected across any recording under current calibration thresholds.</i>", body_style))

    story.append(Spacer(1, 18))

    # 6. Medical & Evaluation Disclaimer
    story.append(KeepTogether([
        Paragraph("Important Research & Regulatory Disclaimer", h2_style),
        Paragraph(
            "<b>NEUROSCAN is an EEG research and analysis support platform.</b> Its model outputs are not a standalone "
            "medical diagnosis and should be interpreted by qualified healthcare professionals. Because patient recordings "
            "are analyzed without ground-truth clinical annotations in this inference mode, sensitivity, specificity, "
            "ROC-AUC, and clinical false-alarm rates are deliberately NOT fabricated or claimed. Threshold calculations "
            "are strictly recording-specific following the pre-calibrated hybrid quantile policy.",
            disclaimer_style,
        ),
    ]))

    doc.build(story, canvasmaker=NumberedCanvas())
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
