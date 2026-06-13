"""Tests for no-LLM background preview."""

from __future__ import annotations

from pathlib import Path

from resume_tune.llm.ai import EMPTY_AI_OUTPUT
from resume_tune.render.resume import (
    build_resume_artifacts,
    docx_to_html,
    load_background,
    static_preview_sections,
)
from resume_tune.llm.selection import full_selection

ROOT = Path(__file__).resolve().parent.parent


def test_static_preview_sections_drops_ai_sections_preserves_order() -> None:
    sections = ["skills", "experience", "education", "projects", "certifications"]
    assert static_preview_sections(sections) == [
        "experience",
        "education",
        "projects",
        "certifications",
    ]


def test_static_preview_sections_drops_summary_and_skills() -> None:
    sections = ["summary", "skills", "experience"]
    assert static_preview_sections(sections) == ["experience"]


def test_background_preview_artifacts_without_llm() -> None:
    background_data = load_background(ROOT / "background.example.md")
    config_sections = ["skills", "experience", "education", "projects", "certifications"]
    preview_sections = static_preview_sections(config_sections)
    content_selection = full_selection(background_data)

    artifacts = build_resume_artifacts(
        background_data,
        dict(EMPTY_AI_OUTPUT),
        sections=preview_sections,
        content_selection=content_selection,
    )

    html = artifacts["html"]
    assert "EXPERIENCE" in html
    assert "EDUCATION" in html
    assert "SKILLS" not in html
    assert "PROFESSIONAL SUMMARY" not in html
    assert artifacts["docx_bytes"]
    assert docx_to_html(artifacts["docx_bytes"]) == html
