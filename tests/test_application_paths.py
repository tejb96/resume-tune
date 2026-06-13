"""Tests for application folder layout helpers."""

from __future__ import annotations

from pathlib import Path

from resume_tune.tracker.paths import (
    JD_SNAPSHOT_FILENAME,
    application_folder_basename,
    organize_application_files,
    resolve_application_dir,
    resolve_stored_path,
)


def test_application_folder_basename() -> None:
    assert application_folder_basename("Acme Corp", "Software Engineer", "2026-06-12") == (
        "2026-06-12_Acme_Corp_Software_Engineer"
    )


def test_resolve_application_dir_adds_suffix_on_collision(tmp_path: Path) -> None:
    root = tmp_path / "Applications"
    first = resolve_application_dir(root, "Acme", "Engineer", "2026-06-12")
    first.mkdir(parents=True)
    second = resolve_application_dir(root, "Acme", "Engineer", "2026-06-12")
    assert first.name == "2026-06-12_Acme_Engineer"
    assert second.name == "2026-06-12_Acme_Engineer_2"


def test_organize_application_files_moves_resume_and_writes_jd(tmp_path: Path) -> None:
    root = tmp_path / "Applications"
    root.mkdir()
    resume = root / "Jane_Doe_Resume.docx"
    resume.write_bytes(b"docx-bytes")
    jd = "Looking for Python and AWS experience."

    dest_resume, dest_jd = organize_application_files(
        root,
        resume_path=resume,
        company="Acme Corp",
        role="Software Engineer",
        date_applied="2026-06-12",
        job_description=jd,
    )

    folder = root / "2026-06-12_Acme_Corp_Software_Engineer"
    assert dest_resume == (folder / "resume.docx").resolve()
    assert dest_jd == (folder / JD_SNAPSHOT_FILENAME).resolve()
    assert dest_resume.read_bytes() == b"docx-bytes"
    assert dest_jd.read_text(encoding="utf-8") == jd
    assert not resume.exists()


def test_resolve_stored_path_supports_relative_and_absolute(tmp_path: Path) -> None:
    root = tmp_path / "Applications"
    folder = root / "2026-06-12_Acme_Engineer"
    folder.mkdir(parents=True)
    resume = folder / "resume.pdf"
    resume.write_bytes(b"pdf")

    assert resolve_stored_path("2026-06-12_Acme_Engineer/resume.pdf", root) == resume.resolve()
    assert resolve_stored_path(str(resume), root) == resume.resolve()
    assert resolve_stored_path("", root) == Path()
