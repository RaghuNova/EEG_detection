"""storage_manager.py
-------------------
Modular structured file storage for uploads, analysis results, and generated reports.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("neuroscan.storage_manager")

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
DEFAULT_ANALYSIS_DIR = BASE_DIR / "storage" / "analyses"
DEFAULT_REPORT_DIR = BASE_DIR / "storage" / "reports"


class StorageManager:
    """Manages file storage, session persistence, and analysis records."""

    def __init__(
        self,
        upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
        analysis_dir: Path | str = DEFAULT_ANALYSIS_DIR,
        report_dir: Path | str = DEFAULT_REPORT_DIR,
    ) -> None:
        self.upload_dir = Path(upload_dir).resolve()
        self.analysis_dir = Path(analysis_dir).resolve()
        self.report_dir = Path(report_dir).resolve()

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_analysis_result(self, analysis_id: str, data: Dict[str, Any]) -> Path:
        """Persist full patient analysis results as JSON."""
        file_path = self.analysis_dir / f"{analysis_id}.json"
        temp_path = file_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(file_path)
        LOGGER.info("Saved analysis %s (%d recordings) to %s", analysis_id, data.get("total_recordings", 0), file_path)
        return file_path

    def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an analysis result by ID."""
        file_path = self.analysis_dir / f"{analysis_id}.json"
        if not file_path.is_file():
            return None
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.error("Error reading analysis %s: %s", analysis_id, exc)
            return None

    def list_analyses(self) -> List[Dict[str, Any]]:
        """List summary records of all stored analysis sessions."""
        records: List[Dict[str, Any]] = []
        for path in sorted(self.analysis_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                # Analysis JSON may contain many recordings/windows. Read only
                # the small top-level summary before `recordings` so history
                # remains responsive even for older files with embedded traces.
                data = self._read_analysis_summary(path)
                records.append({
                    "analysis_id": data.get("analysis_id", path.stem),
                    "patient_id": data.get("patient_id", "Unknown"),
                    "created_at": data.get("created_at"),
                    "status": data.get("status", "completed"),
                    "total_recordings": data.get("total_recordings", 0),
                    "successful_recordings": data.get("successful_recordings", 0),
                    "total_windows": data.get("total_windows", 0),
                    "summary": data.get("summary", {}),
                    "has_report": (self.report_dir / f"{path.stem}.pdf").is_file(),
                })
            except Exception as exc:
                LOGGER.warning("Could not read analysis file %s: %s", path.name, exc)
        return records

    @staticmethod
    def _read_analysis_summary(path: Path) -> Dict[str, Any]:
        prefix: List[str] = []
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if re.match(r'\s*"recordings"\s*:', line):
                    break
                prefix.append(line)
        if not prefix:
            raise ValueError("Analysis file has no summary")
        summary = "".join(prefix).rstrip()
        if summary.endswith(","):
            summary = summary[:-1]
        return json.loads(summary + "}")

    def save_report_pdf(self, analysis_id: str, pdf_bytes: bytes) -> Path:
        """Save compiled PDF report to disk."""
        path = self.report_dir / f"{analysis_id}.pdf"
        path.write_bytes(pdf_bytes)
        LOGGER.info("Saved PDF report for %s to %s (%d bytes)", analysis_id, path, len(pdf_bytes))
        return path

    def get_report_pdf_path(self, analysis_id: str) -> Optional[Path]:
        """Get path to a stored PDF report if available."""
        path = self.report_dir / f"{analysis_id}.pdf"
        return path if path.is_file() else None

    def create_upload_session_dir(self, analysis_id: str) -> Path:
        """Create an isolated upload directory for a session."""
        session_dir = self.upload_dir / analysis_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def cleanup_old_uploads(self, max_age_days: int = 7) -> int:
        """Clean up temporary upload files older than configured days."""
        now = datetime.now(timezone.utc).timestamp()
        removed = 0
        for item in self.upload_dir.iterdir():
            if item.is_dir():
                mtime = item.stat().st_mtime
                if (now - mtime) > (max_age_days * 86400):
                    shutil.rmtree(item, ignore_errors=True)
                    removed += 1
        return removed
