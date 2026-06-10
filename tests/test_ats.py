"""Tests for deterministic ATS compatibility analysis."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai import EMPTY_AI_OUTPUT
from ats import (
    analyze_ats_compatibility,
    compare_pdf_fidelity,
    detect_sections,
    extract_jd_keywords,
    match_keywords,
    parse_contact_info,
)
from resume import build_resume_artifacts, flatten_resume_text, load_background

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_JD = """
Senior Full Stack Developer

Requirements:
- 3+ years experience with Python, FastAPI, and React
- Strong Docker and CI/CD experience
- AWS or GCP cloud deployment
- PostgreSQL and Redis
- Kubernetes experience preferred
"""

SAMPLE_AI_OUTPUT = {
    "summary": "Full-stack developer with Python and React experience.",
    "skill_categories": [
        {
            "name": "Backend",
            "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
        },
        {
            "name": "Frontend",
            "skills": ["React", "TypeScript"],
        },
    ],
}


@pytest.fixture
def sample_background() -> dict:
    return load_background(ROOT / "background.example.md")


def test_extract_jd_keywords_finds_tech_terms() -> None:
    keywords = extract_jd_keywords(SAMPLE_JD)
    lowered = {k.lower() for k in keywords}
    assert "fastapi" in lowered
    assert "react" in lowered
    assert "docker" in lowered
    assert "postgresql" in lowered
    assert "redis" in lowered
    assert "kubernetes" in lowered


def test_extract_jd_keywords_deduplicates_aliases() -> None:
    jd = "We need k8s and Kubernetes experts with JS and JavaScript skills."
    keywords = extract_jd_keywords(jd)
    lowered = [k.lower() for k in keywords]
    assert lowered.count("kubernetes") == 1
    assert lowered.count("javascript") == 1


def test_match_keywords_case_insensitive() -> None:
    matched, missing = match_keywords(["Python", "React"], "Built apps with python and REACT.")
    assert matched == ["Python", "React"]
    assert missing == []


def test_match_keywords_word_boundary() -> None:
    matched, missing = match_keywords(["Go"], "I worked on algorithms and data pipelines.")
    assert matched == []
    assert missing == ["Go"]

    matched_go, _ = match_keywords(["Go"], "Experience with Go and microservices.")
    assert matched_go == ["Go"]


def test_detect_sections_finds_headings() -> None:
    text = "Tej Bal\nSKILLS\nPython\nEXPERIENCE\nSoftware Engineer"
    found, missing = detect_sections(text, ["skills", "experience", "education"])
    assert "skills" in found
    assert "experience" in found
    assert "education" in missing


def test_parse_contact_info() -> None:
    text = (
        "Your Name\n"
        "you@example.com | +1 (555) 123-4567 | LinkedIn | GitHub\n"
        "EXPERIENCE"
    )
    header = {
        "name": "Your Name",
        "email": "you@example.com",
        "phone": "+1 (555) 123-4567",
        "links": [
            {"label": "LinkedIn", "url": "https://linkedin.com/in/you"},
            {"label": "GitHub", "url": "https://github.com/you"},
        ],
    }
    contact = parse_contact_info(text, header)
    assert contact.email_found
    assert contact.phone_found
    assert contact.linkedin_found
    assert contact.github_found
    assert contact.email_matches_expected
    assert contact.phone_matches_expected


def test_compare_pdf_fidelity() -> None:
    header = {"name": "Jane Doe", "email": "jane@example.com"}
    pdf_text = "Jane Doe\njane@example.com\nSKILLS\nPython\nEXPERIENCE\nEngineer"
    flattened = "Jane Doe\njane@example.com\nSKILLS\nPython\nEXPERIENCE\nEngineer"

    fidelity = compare_pdf_fidelity(
        pdf_text=pdf_text,
        flattened_text=flattened,
        header=header,
        expected_sections=["skills", "experience"],
        matched_keywords=["Python"],
    )
    assert fidelity.name_in_pdf
    assert fidelity.email_in_pdf
    assert fidelity.sections_in_pdf["skills"]
    assert fidelity.sections_in_pdf["experience"]
    assert fidelity.fidelity_score >= 80


def test_compare_pdf_fidelity_missing_name() -> None:
    fidelity = compare_pdf_fidelity(
        pdf_text="Unknown\nSKILLS\nPython",
        flattened_text="Jane Doe\nSKILLS\nPython",
        header={"name": "Jane Doe", "email": "jane@example.com"},
        expected_sections=["skills"],
        matched_keywords=[],
    )
    assert not fidelity.name_in_pdf


def test_analyze_ats_end_to_end(sample_background: dict) -> None:
    flattened = flatten_resume_text(
        sample_background,
        SAMPLE_AI_OUTPUT,
        sections=["skills", "experience", "education", "projects", "certifications"],
    )
    report = analyze_ats_compatibility(
        job_description=SAMPLE_JD,
        background_data=sample_background,
        ai_output=SAMPLE_AI_OUTPUT,
        pdf_bytes=None,
        sections=["skills", "experience", "education", "projects", "certifications"],
        content_selection=None,
        max_certifications=1,
    )
    assert report is not None
    assert report.resume_text_source == "flattened"
    assert report.pdf_fidelity is None
    assert report.keyword_match_pct > 0
    assert "Python" in report.matched_keywords or "FastAPI" in report.matched_keywords
    assert "skills" in report.sections_found
    assert report.contact.email_found
    assert len(flattened) > 100


def test_analyze_ats_missing_keywords(sample_background: dict) -> None:
    jd = "Must have extensive Kubernetes, GraphQL, and Redis experience."
    report = analyze_ats_compatibility(
        job_description=jd,
        background_data=sample_background,
        ai_output=EMPTY_AI_OUTPUT,
        pdf_bytes=None,
        sections=["experience", "education", "projects"],
        content_selection=None,
        max_certifications=None,
    )
    assert report is not None
    missing_lower = {k.lower() for k in report.missing_keywords}
    assert "kubernetes" in missing_lower or "graphql" in missing_lower or "redis" in missing_lower


def test_build_resume_artifacts_does_not_auto_run_ats(sample_background: dict) -> None:
    artifacts = build_resume_artifacts(
        sample_background,
        SAMPLE_AI_OUTPUT,
        max_pages=2,
    )
    assert "ats" not in artifacts["diagnostics"]


def test_run_ats_check_on_demand(sample_background: dict) -> None:
    from app import run_ats_check

    artifacts = build_resume_artifacts(
        sample_background,
        SAMPLE_AI_OUTPUT,
        max_pages=2,
    )
    result = {
        "summary": artifacts["ai_output"]["summary"],
        "skill_categories": artifacts["ai_output"]["skill_categories"],
        "content_selection": artifacts["content_selection"],
        "pdf_bytes": artifacts["pdf_bytes"],
    }
    ats = run_ats_check(
        sample_background,
        result,
        SAMPLE_JD,
        sections=["skills", "experience", "education", "projects", "certifications"],
        max_certifications=1,
    )
    assert ats is not None
    assert ats["keyword_match_pct"] > 0
    assert "matched_keywords" in ats


def test_analyze_ats_returns_none_for_empty_jd(sample_background: dict) -> None:
    assert (
        analyze_ats_compatibility(
            job_description="",
            background_data=sample_background,
            ai_output=SAMPLE_AI_OUTPUT,
            pdf_bytes=None,
            sections=None,
            content_selection=None,
            max_certifications=None,
        )
        is None
    )
