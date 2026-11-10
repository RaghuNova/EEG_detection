from backend.storage.storage_manager import StorageManager


def test_history_reads_summary_without_loading_recording_payload(tmp_path):
    analyses = tmp_path / "analyses"
    reports = tmp_path / "reports"
    uploads = tmp_path / "uploads"
    manager = StorageManager(uploads, analyses, reports)
    path = analyses / "large-session.json"
    path.write_text(
        '{\n'
        '  "analysis_id": "large-session",\n'
        '  "patient_id": "CHB_TEST",\n'
        '  "status": "completed",\n'
        '  "total_recordings": 22,\n'
        '  "successful_recordings": 22,\n'
        '  "total_windows": 3833,\n'
        '  "summary": {"attention_positive_windows": 64},\n'
        '  "recordings": [\n'
        + ('    this intentionally is not parsed as JSON\n' * 10000)
    )

    records = manager.list_analyses()

    assert len(records) == 1
    assert records[0]["analysis_id"] == "large-session"
    assert records[0]["successful_recordings"] == 22
    assert records[0]["summary"] == {"attention_positive_windows": 64}
