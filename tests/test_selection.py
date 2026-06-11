"""Tests for job-aware content selection and page-fit trimming."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ai import AIResponseError
from resume import build_resume, fit_resume_to_page_budget, load_background
from selection import (
    apply_content_selection,
    build_ratings_system_prompt,
    default_selection,
    enforce_project_floor,
    full_selection,
    parse_ratings_response,
    selection_policy_for_background,
    trim_selection_one_step,
)
from scoring import SelectionPolicy, build_selection_from_scores

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_background() -> dict:
    return load_background(ROOT / "background.example.md")


def test_parse_ratings_response_valid_json(sample_background: dict) -> None:
    raw = """
    {
      "roles": [5, 3],
      "experience_bullets": [[5, 4, 2], [4, 3]],
      "projects": [4],
      "project_bullets": [[4]],
      "education": [5]
    }
    """
    ratings = parse_ratings_response(raw, sample_background)
    assert ratings.roles == {0: 5, 1: 3}
    assert ratings.experience_bullets[(0, 2)] == 2
    assert ratings.projects == {0: 4}
    assert ratings.education == {0: 5}


def test_parse_ratings_response_clamps_out_of_range(sample_background: dict) -> None:
    raw = """
    {
      "roles": [9, 0],
      "experience_bullets": [[5, 4, 2], [4, 3]],
      "projects": [4],
      "project_bullets": [[4]],
      "education": [5]
    }
    """
    ratings = parse_ratings_response(raw, sample_background)
    assert ratings.roles == {0: 5, 1: 1}


def test_parse_ratings_response_pads_short_ratings(sample_background: dict) -> None:
    raw = """
    {
      "roles": [5],
      "experience_bullets": [[5, 4], [4, 3]],
      "projects": [4],
      "project_bullets": [[4]],
      "education": [5]
    }
    """
    ratings = parse_ratings_response(raw, sample_background)
    assert ratings.roles == {0: 5, 1: 3}
    assert ratings.experience_bullets[(0, 0)] == 5
    assert ratings.experience_bullets[(0, 1)] == 4
    assert ratings.experience_bullets[(0, 2)] == 3


def test_parse_ratings_response_rejects_invalid_json(sample_background: dict) -> None:
    with pytest.raises(AIResponseError, match="invalid JSON"):
        parse_ratings_response("not json", sample_background)


def test_ratings_build_selects_relevant_role_and_bullets(sample_background: dict) -> None:
    raw = """
    {
      "roles": [2, 5],
      "experience_bullets": [[3, 2, 1], [5, 4]],
      "projects": [5],
      "project_bullets": [[5]],
      "education": [4]
    }
    """
    ratings = parse_ratings_response(raw, sample_background)
    selection = build_selection_from_scores(
        sample_background,
        ratings,
        policy=SelectionPolicy(max_experience_entries=2, max_bullets_per_role=2),
    )
    role_indices = [e["role_index"] for e in selection["experience_selections"]]
    assert role_indices == [0, 1]
    exp_by_role = {e["role_index"]: e for e in selection["experience_selections"]}
    assert exp_by_role[1]["bullet_indices"] == [0, 1]
    assert selection["project_selections"][0]["project_index"] == 0
    assert "content_composites" in selection


def test_ratings_prompt_lists_every_item_and_counts(sample_background: dict) -> None:
    prompt = build_ratings_system_prompt(sample_background)
    assert '"roles": [2 integers]' in prompt
    assert '"experience_bullets": [[3 integers], [2 integers]]' in prompt
    assert "Senior Software Engineer" in prompt
    assert "B.S. Computer Science" in prompt


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


def test_default_selection_includes_scores(sample_background: dict) -> None:
    selection = default_selection(sample_background)
    assert selection["experience_selections"]
    assert "content_composites" in selection
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


def test_fit_resume_trims_lowest_composite_not_position() -> None:
    """With score metadata, page fitting drops the least relevant bullet, not the last one."""
    background = {
        "header": {"name": "Test User", "email": "test@example.com"},
        "experience": [
            {
                "company": "Co",
                "title": "Engineer",
                "start": "2020-01",
                "end": "present",
                "bullets": ["Most relevant bullet", "Weak bullet", "Strong closing bullet"],
            }
        ],
        "education": [],
        "projects": [{"name": "P", "bullets": ["Relevant project bullet"]}],
        "certifications": [],
    }
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0, 1, 2]}],
        "project_selections": [{"project_index": 0, "bullet_indices": [0]}],
        "education_indices": [],
        "content_composites": {
            "exp:0:0": 92.5,
            "exp:0:1": 40.0,
            "exp:0:2": 85.0,
            "proj:0:0": 80.0,
        },
    }
    call_count = {"n": 0}

    def fake_page_count(_pdf: bytes) -> int:
        call_count["n"] += 1
        return 2 if call_count["n"] == 1 else 1

    with patch("resume.docx_to_pdf", return_value=b"%PDF-fake"):
        with patch("resume.pdf_page_count", side_effect=fake_page_count):
            _ai, fitted_selection, _docx, _pdf, page_count, diagnostics = (
                fit_resume_to_page_budget(
                    background,
                    {"summary": "", "skill_categories": []},
                    max_pages=1,
                    sections=["experience", "projects"],
                    content_selection=selection,
                    min_project_entries=1,
                    min_project_bullets=1,
                )
            )

    assert page_count == 1
    # The mid-position low-composite bullet is removed; positional trim would drop index 2.
    assert fitted_selection["experience_selections"][0]["bullet_indices"] == [0, 2]
    # The project is at its floor and protected even though scored below experience.
    assert fitted_selection["project_selections"][0]["bullet_indices"] == [0]
    assert diagnostics["trim_log"]
    assert diagnostics["trim_log"][0]["removed"] == "exp:0:1"


def test_fit_resume_expands_until_page_overflow() -> None:
    """Expand loop adds highest-scored bullets while trial PDF still fits."""
    background = {
        "header": {"name": "Test User", "email": "test@example.com"},
        "experience": [
            {
                "company": "Co",
                "title": "Engineer",
                "start": "2020-01",
                "end": "present",
                "bullets": ["Strong bullet", "Weak bullet", "Mid bullet"],
            }
        ],
        "education": [],
        "projects": [],
        "certifications": [],
    }
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0]}],
        "project_selections": [],
        "education_indices": [],
        "content_composites": {
            "exp:0:0": 92.5,
            "exp:0:1": 40.0,
            "exp:0:2": 85.0,
        },
    }
    call_count = {"n": 0}

    def fake_page_count(_pdf: bytes) -> int:
        call_count["n"] += 1
        # Initial + first expand fits; second expand would overflow
        return 1 if call_count["n"] <= 2 else 2

    with patch("resume.docx_to_pdf", return_value=b"%PDF-fake"):
        with patch("resume.pdf_page_count", side_effect=fake_page_count):
            _ai, fitted_selection, _docx, _pdf, page_count, diagnostics = (
                fit_resume_to_page_budget(
                    background,
                    {"summary": "", "skill_categories": []},
                    max_pages=1,
                    sections=["experience"],
                    content_selection=selection,
                )
            )

    assert page_count == 1
    assert fitted_selection["experience_selections"][0]["bullet_indices"] == [0, 2]
    assert diagnostics["expand_log"]
    assert diagnostics["expand_log"][0]["added"] == "exp:0:2"


def test_fit_resume_overflow_warning_when_strong_content_remains() -> None:
    from resume import _page_fit_diagnostics

    background = {
        "header": {"name": "Test User", "email": "test@example.com"},
        "experience": [
            {
                "company": "Co",
                "title": "Engineer",
                "start": "2020-01",
                "end": "present",
                "bullets": ["On page", "Extra strong"],
            },
            {
                "company": "Old",
                "title": "Intern",
                "start": "2018-01",
                "end": "2019-12",
                "bullets": ["More strong content for page two"],
            },
        ],
        "education": [],
        "projects": [],
        "certifications": [],
    }
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0]}],
        "project_selections": [],
        "education_indices": [],
        "content_composites": {
            "exp:0:0": 92.5,
            "exp:0:1": 80.0,
            "exp:1:0": 85.0,
        },
    }
    ai_output = {"summary": "", "skill_categories": []}

    def mock_render(sel: dict, _ai: dict) -> tuple[bytes, bytes, int]:
        bullets = sum(len(e["bullet_indices"]) for e in sel["experience_selections"])
        pages = 2 if bullets >= 3 else 1
        return b"docx", b"%PDF", pages

    diagnostics = _page_fit_diagnostics(
        background,
        ai_output,
        selection,
        page_count=1,
        max_pages=1,
        max_chars=None,
        include_summary=False,
        include_skills=False,
        trim_stalled=False,
        render_page_count=mock_render,
        ai_output_for_render=ai_output,
    )

    assert diagnostics["overflow_warning"] is True
    assert diagnostics["omitted_high_quality_count"] == 2


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
