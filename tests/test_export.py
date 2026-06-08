"""Tests for resume export helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from resume import (
    build_resume,
    build_resume_artifacts,
    docx_to_html,
    docx_to_pdf,
    fit_resume_to_page_budget,
    libreoffice_available,
    load_background,
    pdf_page_count,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_background() -> dict:
    return load_background(ROOT / "background.example.md")


@pytest.fixture
def sample_ai_output() -> dict:
    return {
        "summary": (
            "Software engineer with 5+ years building scalable backend systems and APIs "
            "handling 2M+ daily requests. Deep experience in Python, Go, AWS, and Kubernetes."
        ),
        "skill_categories": [
            {"name": "Languages", "skills": ["Python", "Go", "SQL"]},
            {"name": "Infrastructure", "skills": ["AWS", "Docker", "Kubernetes"]},
        ],
    }


def test_docx_to_html_contains_candidate_name(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    docx_bytes = build_resume(sample_background, sample_ai_output)
    html = docx_to_html(docx_bytes)

    assert "<html>" in html
    assert sample_background["header"]["name"] in html


def test_build_resume_artifacts_returns_docx_and_html(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    artifacts = build_resume_artifacts(sample_background, sample_ai_output)

    assert artifacts["docx_bytes"]
    assert artifacts["ai_output"]["skill_categories"]
    assert sample_background["header"]["name"] in artifacts["html"]
    assert "pdf_bytes" in artifacts
    assert "page_count" in artifacts


def test_docx_to_pdf_returns_none_without_libreoffice(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    docx_bytes = build_resume(sample_background, sample_ai_output)

    with patch("resume.libreoffice_available", return_value=False):
        assert docx_to_pdf(docx_bytes) is None


@pytest.mark.skipif(not libreoffice_available(), reason="LibreOffice not installed")
def test_docx_to_pdf_returns_bytes_when_libreoffice_available(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    docx_bytes = build_resume(sample_background, sample_ai_output)
    pdf_bytes = docx_to_pdf(docx_bytes)

    assert pdf_bytes
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_page_count(pdf_bytes) >= 1


@pytest.mark.skipif(not libreoffice_available(), reason="LibreOffice not installed")
def test_fit_resume_to_page_budget_returns_page_count(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    fitted, _docx, pdf_bytes, page_count = fit_resume_to_page_budget(
        sample_background,
        sample_ai_output,
        max_pages=2,
    )

    assert fitted["summary"]
    assert fitted["skill_categories"]
    assert pdf_bytes
    assert page_count is not None
    assert page_count >= 1


def test_legacy_flat_skills_migrated_in_build_resume(
    sample_background: dict,
) -> None:
    ai_output = {
        "summary": "Engineer with Python and Go experience on AWS.",
        "skills": ["Python", "Go", "AWS"],
    }
    docx_bytes = build_resume(sample_background, ai_output)
    html = docx_to_html(docx_bytes)

    assert "Python" in html
    assert "Go" in html
