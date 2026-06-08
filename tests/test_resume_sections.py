"""Tests for configurable resume section order."""

from __future__ import annotations

from pathlib import Path

import pytest

from resume import (
    DEFAULT_RESUME_SECTIONS,
    build_resume,
    docx_to_html,
    load_background,
    resolve_resume_sections,
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


def _heading_positions(html: str) -> dict[str, int]:
    headings = {
        "PROFESSIONAL SUMMARY": html.find("PROFESSIONAL SUMMARY"),
        "SKILLS": html.find("SKILLS"),
        "EXPERIENCE": html.find("EXPERIENCE"),
        "EDUCATION": html.find("EDUCATION"),
        "PROJECTS": html.find("PROJECTS"),
        "CERTIFICATIONS": html.find("CERTIFICATIONS"),
    }
    return {name: pos for name, pos in headings.items() if pos >= 0}


def test_default_order_unchanged(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    docx_bytes = build_resume(sample_background, sample_ai_output)
    html = docx_to_html(docx_bytes)
    positions = _heading_positions(html)

    assert list(positions) == [
        "PROFESSIONAL SUMMARY",
        "SKILLS",
        "EXPERIENCE",
        "EDUCATION",
        "PROJECTS",
        "CERTIFICATIONS",
    ]
    ordered = [positions[name] for name in positions]
    assert ordered == sorted(ordered)


def test_exclude_summary(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    sections = [section for section in DEFAULT_RESUME_SECTIONS if section != "summary"]
    docx_bytes = build_resume(sample_background, sample_ai_output, sections=sections)
    html = docx_to_html(docx_bytes)

    assert "PROFESSIONAL SUMMARY" not in html
    assert "SKILLS" in html


def test_reorder_certifications_before_education(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    sections = [
        "summary",
        "skills",
        "experience",
        "certifications",
        "education",
        "projects",
    ]
    docx_bytes = build_resume(sample_background, sample_ai_output, sections=sections)
    html = docx_to_html(docx_bytes)
    positions = _heading_positions(html)

    assert positions["CERTIFICATIONS"] < positions["EDUCATION"]
    assert positions["EDUCATION"] < positions["PROJECTS"]


def test_resolve_resume_sections_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="Unknown resume section"):
        resolve_resume_sections(["summary", "bad_id"])


def test_resolve_resume_sections_falls_back_to_default_when_empty() -> None:
    assert resolve_resume_sections(None) == DEFAULT_RESUME_SECTIONS
    assert resolve_resume_sections([]) == DEFAULT_RESUME_SECTIONS
