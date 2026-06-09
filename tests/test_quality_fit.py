"""Tests for quality-first skills layout and static-only page fitting."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ai import enforce_skills_layout, skill_category_line_length
from resume import build_resume, docx_to_html, fit_resume_to_page_budget, load_background

ROOT = Path(__file__).resolve().parent.parent


def test_enforce_skills_layout_caps_categories_and_skills() -> None:
    categories = [
        {"name": "A", "skills": ["1", "2", "3", "4", "5", "6"]},
        {"name": "B", "skills": ["x"]},
        {"name": "C", "skills": ["y"]},
        {"name": "D", "skills": ["z"]},
        {"name": "E", "skills": ["w"]},
    ]
    result = enforce_skills_layout(
        categories,
        max_categories=3,
        max_skills_per_category=2,
        max_chars_per_line=88,
    )
    assert len(result) == 3
    assert len(result[0]["skills"]) == 2


def test_enforce_skills_layout_trims_line_to_max_chars() -> None:
    categories = [
        {
            "name": "Backend",
            "skills": [
                "Node.js",
                "TypeScript",
                "Express.js",
                "NestJS",
                "FastAPI",
                "PostgreSQL",
            ],
        },
    ]
    result = enforce_skills_layout(
        categories,
        max_categories=4,
        max_skills_per_category=6,
        max_chars_per_line=40,
    )
    assert result
    line_len = skill_category_line_length(result[0]["name"], result[0]["skills"])
    assert line_len <= 40


def test_fit_resume_preserves_skills_while_trimming_static() -> None:
    heavy_background = {
        "header": {"name": "Test User", "email": "test@example.com"},
        "experience": [
            {
                "company": "Co",
                "title": "Engineer",
                "start": "2020-01",
                "end": "present",
                "bullets": [f"Bullet {i} with enough text to consume vertical space." for i in range(6)],
            }
        ],
        "education": [],
        "projects": [],
        "certifications": [],
    }
    ai_output = {
        "summary": "",
        "skill_categories": [{"name": "Stack", "skills": ["Python", "Go", "AWS"]}],
    }
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": list(range(6))}],
        "project_selections": [],
        "education_indices": [],
    }
    original_skills = ai_output["skill_categories"]

    call_count = {"n": 0}

    def fake_page_count(_pdf: bytes) -> int:
        call_count["n"] += 1
        return 2 if call_count["n"] == 1 else 1

    with patch("resume.docx_to_pdf", return_value=b"%PDF-fake"):
        with patch("resume.pdf_page_count", side_effect=fake_page_count):
            fitted_ai, fitted_selection, _docx, _pdf, page_count, diagnostics = (
                fit_resume_to_page_budget(
                    heavy_background,
                    ai_output,
                    max_pages=1,
                    sections=["skills", "experience"],
                    content_selection=selection,
                )
            )

    assert fitted_ai["skill_categories"] == original_skills
    assert page_count == 1
    assert len(fitted_selection["experience_selections"][0]["bullet_indices"]) == 5
    assert diagnostics["trim_stalled"] is False


def test_fit_resume_trim_stalled_without_value_error() -> None:
    heavy_background = {
        "header": {"name": "Test User", "email": "test@example.com"},
        "experience": [
            {
                "company": "Co",
                "title": "Engineer",
                "start": "2020-01",
                "end": "present",
                "bullets": ["Only one bullet that still overflows somehow"],
            }
        ],
        "education": [],
        "projects": [],
        "certifications": [],
    }
    ai_output = {
        "summary": "",
        "skill_categories": [{"name": "Stack", "skills": ["Python"]}],
    }
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0]}],
        "project_selections": [],
        "education_indices": [],
    }

    with patch("resume.docx_to_pdf", return_value=b"%PDF-fake"):
        with patch("resume.pdf_page_count", return_value=2):
            fitted_ai, _sel, _docx, _pdf, page_count, diagnostics = fit_resume_to_page_budget(
                heavy_background,
                ai_output,
                max_pages=1,
                sections=["skills", "experience"],
                content_selection=selection,
            )

    assert page_count == 2
    assert diagnostics["trim_stalled"] is True
    assert fitted_ai["skill_categories"] == ai_output["skill_categories"]


@pytest.fixture
def sample_background() -> dict:
    return load_background(ROOT / "background.example.md")


def test_build_resume_max_certifications_renders_first_only(sample_background: dict) -> None:
    ai_output = {"summary": "", "skill_categories": []}
    docx_bytes = build_resume(
        sample_background,
        ai_output,
        sections=["certifications"],
        max_certifications=1,
    )
    html = docx_to_html(docx_bytes)
    first_cert = sample_background["certifications"][0]["name"]
    assert first_cert in html
    if len(sample_background["certifications"]) > 1:
        second_cert = sample_background["certifications"][1]["name"]
        assert second_cert not in html
