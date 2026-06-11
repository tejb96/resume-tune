"""Tests for application tracker spreadsheet helpers."""

from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from tracker import HEADERS, ensure_tracker, log_application


def test_ensure_tracker_creates_formatted_workbook(tmp_path: Path) -> None:
    tracker_path = tmp_path / "applications.xlsx"
    ensure_tracker(tracker_path)

    assert tracker_path.is_file()

    wb = load_workbook(tracker_path)
    ws = wb.active
    assert [ws.cell(row=1, column=i).value for i in range(1, len(HEADERS) + 1)] == list(
        HEADERS
    )
    assert all(ws.cell(row=1, column=i).font.bold for i in range(1, len(HEADERS) + 1))
    assert ws.freeze_panes == "A2"
    assert ws.column_dimensions["A"].width == 14
    assert ws.column_dimensions["B"].width == 22


def test_ensure_tracker_is_idempotent(tmp_path: Path) -> None:
    tracker_path = tmp_path / "applications.xlsx"
    ensure_tracker(tracker_path)
    ensure_tracker(tracker_path)
    assert tracker_path.is_file()
    wb = load_workbook(tracker_path)
    assert wb.active.max_row == 1


def test_log_application_appends_row(tmp_path: Path) -> None:
    tracker_path = tmp_path / "applications.xlsx"
    row = {
        "Date Applied": "2026-06-11",
        "Company": "Acme Corp",
        "Role / Job Title": "Software Engineer",
        "Location": "Remote",
        "Status": "Applied",
        "ATS Keywords": "Python, AWS",
        "Resume File": "/tmp/resume.docx",
        "Notes": "Referred by friend",
    }
    log_application(tracker_path, row)

    wb = load_workbook(tracker_path)
    ws = wb.active
    assert ws.max_row == 2
    assert [ws.cell(row=2, column=i).value for i in range(1, len(HEADERS) + 1)] == [
        row[header] for header in HEADERS
    ]
    assert all(ws.cell(row=1, column=i).font.bold for i in range(1, len(HEADERS) + 1))


def test_log_application_appends_second_row(tmp_path: Path) -> None:
    tracker_path = tmp_path / "applications.xlsx"
    base_row = {
        "Date Applied": "2026-06-11",
        "Company": "Acme Corp",
        "Role / Job Title": "Software Engineer",
        "Location": "Remote",
        "Status": "Applied",
        "ATS Keywords": "Python",
        "Resume File": "/tmp/resume1.docx",
        "Notes": "",
    }
    log_application(tracker_path, base_row)
    second = {**base_row, "Company": "Beta Inc", "Resume File": "/tmp/resume2.docx"}
    log_application(tracker_path, second)

    wb = load_workbook(tracker_path)
    ws = wb.active
    assert ws.max_row == 3
    assert ws.cell(row=3, column=2).value == "Beta Inc"
    assert ws.cell(row=3, column=7).value == "/tmp/resume2.docx"
