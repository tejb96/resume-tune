"""Tests for job-aware content selection and page-fit trimming."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from resume import _prevent_orphan_word, build_resume, fit_resume_to_page_budget, load_background
from selection import (
    apply_content_selection,
    default_selection,
    enforce_project_floor,
    full_selection,
    parse_selection_response,
    trim_selection_one_step,
    validate_selection,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_background() -> dict:
    return load_background(ROOT / "background.example.md")


def test_validate_selection_rejects_out_of_range_role_index(sample_background: dict) -> None:
    with pytest.raises(ValueError, match="role_index 99 out of range"):
        validate_selection(
            {
                "experience_selections": [{"role_index": 99, "bullet_indices": [0]}],
                "project_selections": [],
                "education_indices": [0],
            },
            sample_background,
        )


def test_validate_selection_rejects_out_of_range_bullet_index(sample_background: dict) -> None:
    with pytest.raises(ValueError, match="bullet_index 99 out of range"):
        validate_selection(
            {
                "experience_selections": [{"role_index": 0, "bullet_indices": [99]}],
                "project_selections": [],
                "education_indices": [0],
            },
            sample_background,
        )


def test_validate_selection_rejects_too_many_bullets_per_role(sample_background: dict) -> None:
    with pytest.raises(ValueError, match="too many bullet_indices"):
        validate_selection(
            {
                "experience_selections": [{"role_index": 0, "bullet_indices": [0, 1, 2, 3]}],
                "project_selections": [],
                "education_indices": [0],
            },
            sample_background,
            max_bullets_per_role=3,
        )


def test_parse_selection_response_valid_json(sample_background: dict) -> None:
    raw = """
    {
      "experience_selections": [{"role_index": 0, "bullet_indices": [0, 1]}],
      "project_selections": [{"project_index": 0, "bullet_indices": [0]}],
      "education_indices": [0]
    }
    """
    selection = parse_selection_response(raw, sample_background)
    assert selection["experience_selections"][0]["role_index"] == 0
    assert selection["experience_selections"][0]["bullet_indices"] == [0, 1]
    assert selection["education_indices"] == [0]


def test_apply_content_selection_copies_bullets_verbatim(sample_background: dict) -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [1, 0]}],
        "project_selections": [],
        "education_indices": [0],
    }
    render_data = apply_content_selection(sample_background, selection)
    original_bullets = sample_background["experience"][0]["bullets"]
    assert render_data["experience"][0]["bullets"] == [original_bullets[1], original_bullets[0]]


def test_apply_content_selection_filters_education(sample_background: dict) -> None:
    selection = {
        "experience_selections": [],
        "project_selections": [],
        "education_indices": [0],
    }
    render_data = apply_content_selection(sample_background, selection)
    assert len(render_data["education"]) == 1
    assert render_data["education"][0]["degree"] == sample_background["education"][0]["degree"]


def test_default_selection_respects_limits(sample_background: dict) -> None:
    selection = default_selection(
        sample_background,
        max_experience_entries=1,
        max_bullets_per_role=2,
        max_project_entries=1,
    )
    assert len(selection["experience_selections"]) == 1
    assert len(selection["experience_selections"][0]["bullet_indices"]) <= 2
    assert len(selection["project_selections"]) <= 1
    assert selection["project_selections"]


def test_enforce_project_floor_injects_project_when_empty() -> None:
    background = {
        "experience": [
            {
                "company": "Co",
                "title": "Dev",
                "start": "2020",
                "end": "present",
                "bullets": ["Did work"],
            }
        ],
        "projects": [{"name": "P", "bullets": ["b1", "b2"]}],
        "education": [],
    }
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0]}],
        "project_selections": [],
        "education_indices": [],
    }
    result = enforce_project_floor(background, selection)
    assert len(result["project_selections"]) == 1
    assert result["project_selections"][0]["project_index"] == 0
    assert result["project_selections"][0]["bullet_indices"] == [0]


def test_trim_selection_one_step_drops_experience_before_projects() -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0, 1]}],
        "project_selections": [{"project_index": 0, "bullet_indices": [0, 1]}],
        "education_indices": [0, 1],
    }
    trimmed = trim_selection_one_step(selection)
    assert trimmed["experience_selections"][0]["bullet_indices"] == [0]
    assert trimmed["project_selections"][0]["bullet_indices"] == [0, 1]


def test_trim_selection_one_step_drops_project_after_experience_exhausted() -> None:
    selection = {
        "experience_selections": [],
        "project_selections": [{"project_index": 0, "bullet_indices": [0, 1]}],
        "education_indices": [0],
    }
    trimmed = trim_selection_one_step(selection)
    assert trimmed["project_selections"][0]["bullet_indices"] == [0]


def test_trim_selection_one_step_respects_project_floor() -> None:
    selection = {
        "experience_selections": [],
        "project_selections": [{"project_index": 0, "bullet_indices": [0]}],
        "education_indices": [0, 1],
    }
    trimmed = trim_selection_one_step(selection)
    assert trimmed["project_selections"][0]["bullet_indices"] == [0]
    assert trimmed["education_indices"] == [0]


def test_trim_selection_one_step_drops_experience_when_no_projects() -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0, 1]}],
        "project_selections": [],
        "education_indices": [],
    }
    trimmed = trim_selection_one_step(selection)
    assert trimmed["experience_selections"][0]["bullet_indices"] == [0]


def test_prevent_orphan_word_joins_last_two_words() -> None:
    result = _prevent_orphan_word(
        "Resolved 100+ high-priority HubSpot support tickets, keeping stakeholders informed throughout"
    )
    assert "\u00a0" in result
    assert result.endswith("informed\u00a0throughout")


def test_fit_resume_trims_static_when_ai_exhausted() -> None:
    heavy_background = {
        "header": {"name": "Test User", "email": "test@example.com"},
        "experience": [
            {
                "company": "Co",
                "title": "Engineer",
                "start": "2020-01",
                "end": "present",
                "bullets": [f"Bullet {i} with enough text to consume vertical space." for i in range(8)],
            }
        ],
        "education": [],
        "projects": [],
        "certifications": [],
    }
    ai_output = {"summary": "", "skill_categories": []}
    selection = full_selection(heavy_background)

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
                    sections=["experience"],
                    content_selection=selection,
                )
            )

    assert fitted_ai == ai_output
    assert page_count == 1
    assert len(fitted_selection["experience_selections"][0]["bullet_indices"]) == 7
    assert diagnostics["trim_stalled"] is False


def test_fit_resume_keeps_minimum_project_bullet() -> None:
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
        "projects": [{"name": "Side Project", "bullets": ["Built a MERN app with TensorFlow.js"]}],
        "certifications": [],
    }
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": list(range(6))}],
        "project_selections": [{"project_index": 0, "bullet_indices": [0]}],
        "education_indices": [],
    }
    call_count = {"n": 0}

    def fake_page_count(_pdf: bytes) -> int:
        call_count["n"] += 1
        return 2 if call_count["n"] <= 6 else 1

    with patch("resume.docx_to_pdf", return_value=b"%PDF-fake"):
        with patch("resume.pdf_page_count", side_effect=fake_page_count):
            _fitted_ai, fitted_selection, _docx, _pdf, page_count, diagnostics = (
                fit_resume_to_page_budget(
                    heavy_background,
                    {"summary": "", "skill_categories": []},
                    max_pages=1,
                    sections=["experience", "projects"],
                    content_selection=selection,
                    min_project_entries=1,
                    min_project_bullets=1,
                )
            )

    assert page_count == 1
    assert fitted_selection["project_selections"]
    assert fitted_selection["project_selections"][0]["bullet_indices"] == [0]
    assert diagnostics["trim_stalled"] is False


def test_build_resume_with_selection_excludes_unselected_bullets(sample_background: dict) -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0]}],
        "project_selections": [],
        "education_indices": [0],
    }
    render_data = apply_content_selection(sample_background, selection)
    docx_bytes = build_resume(
        sample_background,
        {"summary": "", "skill_categories": []},
        sections=["experience"],
        content_selection=selection,
    )
    assert docx_bytes
    assert len(render_data["experience"][0]["bullets"]) == 1
