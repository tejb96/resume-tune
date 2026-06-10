"""Tests for skipping LLM generation of excluded AI sections."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai import (
    EMPTY_AI_OUTPUT,
    ai_output_char_count,
    enforce_output_budget,
    generate_tailored_content,
    normalize_ai_output,
    pack_skill_lines,
    parse_response,
    revise_tailored_content,
    skills_subject_to_char_budget,
)
from resume import trim_summary_one_step

ROOT = Path(__file__).resolve().parent.parent


def test_parse_response_accepts_skills_only() -> None:
    payload = {
        "skill_categories": [
            {"name": "Languages", "skills": ["Python", "Go"]},
        ],
    }
    parsed = parse_response(json.dumps(payload), include_summary=False, include_skills=True)
    assert parsed["summary"] == ""
    assert parsed["skill_categories"][0]["name"] == "Languages"


def test_parse_response_accepts_summary_only() -> None:
    payload = {"summary": "Backend engineer with Python experience."}
    parsed = parse_response(json.dumps(payload), include_summary=True, include_skills=False)
    assert parsed["summary"] == "Backend engineer with Python experience."
    assert parsed["skill_categories"] == []


def test_normalize_ai_output_allows_empty_excluded_fields() -> None:
    assert normalize_ai_output(
        {"summary": "", "skill_categories": []},
        include_summary=False,
        include_skills=False,
    ) == EMPTY_AI_OUTPUT

    normalized = normalize_ai_output(
        {"summary": "", "skill_categories": [{"name": "Languages", "skills": ["Python"]}]},
        include_summary=False,
        include_skills=True,
    )
    assert normalized["summary"] == ""
    assert normalized["skill_categories"][0]["skills"] == ["Python"]

    normalized = normalize_ai_output(
        {"summary": "Engineer.", "skill_categories": []},
        include_summary=True,
        include_skills=False,
    )
    assert normalized["summary"] == "Engineer."
    assert normalized["skill_categories"] == []


def test_enforce_output_budget_skips_excluded_summary() -> None:
    trimmed = enforce_output_budget(
        "A" * 200,
        [{"name": "Languages", "skills": ["Python"]}],
        50,
        include_summary=False,
        include_skills=True,
    )
    assert trimmed["summary"] == ""
    assert trimmed["skill_categories"][0]["skills"] == ["Python"]


def test_skills_only_excluded_from_char_budget() -> None:
    assert not skills_subject_to_char_budget(include_summary=False, include_skills=True)
    categories = [{"name": "Languages", "skills": ["Python", "Go", "Java", "Rust", "C"]}]
    assert ai_output_char_count("", categories, include_summary=False, include_skills=True) == 0
    trimmed = enforce_output_budget(
        "",
        categories,
        10,
        include_summary=False,
        include_skills=True,
    )
    assert trimmed["skill_categories"] == categories


def test_pack_skill_lines_fills_missing_git() -> None:
    background = "Other tools: Git, Linux, Jira. Experience with Python and Docker."
    categories = [
        {"name": "Backend", "skills": ["Python", "Docker"]},
    ]
    jd = "Looking for Python, Docker, and Git experience."
    packed, info = pack_skill_lines(
        categories,
        background,
        jd,
        max_chars_per_line=88,
    )
    all_skills = [s for cat in packed for s in cat["skills"]]
    assert "Git" in all_skills
    assert "Git" in info["added_skills"]


def test_pack_skill_lines_prefers_tailwind_css_label() -> None:
    background = "Full stack: React, Tailwind, Tailwind CSS, TypeScript."
    categories = [{"name": "Frontend", "skills": ["React", "TypeScript"]}]
    jd = "Requires React, TypeScript, and Tailwind CSS."
    packed, info = pack_skill_lines(
        categories,
        background,
        jd,
        max_chars_per_line=88,
    )
    all_skills = [s for cat in packed for s in cat["skills"]]
    assert "Tailwind CSS" in all_skills


def test_trim_summary_one_step_skips_skills() -> None:
    from resume import trim_summary_one_step

    ai_output = {
        "summary": "One sentence. Second sentence. Third sentence.",
        "skill_categories": [{"name": "Languages", "skills": ["Python", "Go"]}],
    }
    trimmed = trim_summary_one_step(
        ai_output,
        sections=["summary", "skills", "experience"],
    )
    assert trimmed["summary"] != ai_output["summary"]
    assert trimmed["skill_categories"] == ai_output["skill_categories"]


def test_generate_tailored_content_skips_llm_when_both_excluded() -> None:
    background_path = ROOT / "background.example.md"
    with patch("ai.OpenAI") as mock_openai:
        ai_output, warnings, packer = generate_tailored_content(
            "",
            background_path,
            endpoint_url="http://localhost:13305/api/v1",
            model_name="test-model",
            include_summary=False,
            include_skills=False,
        )
    mock_openai.assert_not_called()
    assert ai_output == EMPTY_AI_OUTPUT
    assert warnings == []
    assert packer == {"added_skills": [], "line_utilization": []}


def test_generate_tailored_content_skills_only_uses_skills_prompt() -> None:
    background_path = ROOT / "background.example.md"
    payload = {
        "skill_categories": [
            {"name": "Languages", "skills": ["Python", "Go"]},
        ],
    }

    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = json.dumps(payload)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    passthrough_packer = lambda cats, *_a, **_k: (
        cats,
        {"added_skills": [], "line_utilization": []},
    )
    with (
        patch("ai.OpenAI", return_value=mock_client),
        patch("ai.pack_skill_lines", side_effect=passthrough_packer),
    ):
        ai_output, _warnings, _packer = generate_tailored_content(
            "Backend role",
            background_path,
            endpoint_url="http://localhost:13305/api/v1",
            model_name="test-model",
            include_summary=False,
            include_skills=True,
        )

    system_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert '"summary"' not in system_prompt
    assert "LINE CAPACITY" in system_prompt
    assert "skill_categories" in system_prompt
    assert ai_output["summary"] == ""
    assert ai_output["skill_categories"][0]["skills"] == ["Python", "Go"]


def test_generate_tailored_content_summary_only_uses_summary_prompt() -> None:
    background_path = ROOT / "background.example.md"
    payload = {"summary": "Backend engineer focused on Python APIs."}

    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = json.dumps(payload)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("ai.OpenAI", return_value=mock_client):
        ai_output, _warnings, _packer = generate_tailored_content(
            "Backend role",
            background_path,
            endpoint_url="http://localhost:13305/api/v1",
            model_name="test-model",
            include_summary=True,
            include_skills=False,
        )

    system_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0][
        "content"
    ]
    assert "skill_categories" not in system_prompt
    assert ai_output["summary"] == "Backend engineer focused on Python APIs."
    assert ai_output["skill_categories"] == []


def test_revise_tailored_content_skills_only_preserves_summary() -> None:
    background_path = ROOT / "background.example.md"
    current_output = {
        "summary": "Keep this summary.",
        "skill_categories": [{"name": "Languages", "skills": ["Python", "Go"]}],
    }
    revised = {
        "skill_categories": [
            {"name": "Languages", "skills": ["Python", "Go"]},
            {"name": "Data", "skills": ["PostgreSQL"]},
        ],
    }

    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = json.dumps(revised)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    passthrough_packer = lambda cats, *_a, **_k: (
        cats,
        {"added_skills": [], "line_utilization": []},
    )
    with (
        patch("ai.OpenAI", return_value=mock_client),
        patch("ai.pack_skill_lines", side_effect=passthrough_packer),
    ):
        result, _warnings, _packer = revise_tailored_content(
            "Backend engineer role",
            current_output,
            "Add a data category.",
            background_path,
            endpoint_url="http://localhost:13305/api/v1",
            model_name="test-model",
            include_summary=False,
            include_skills=True,
        )

    assert result["summary"] == "Keep this summary."
    assert len(result["skill_categories"]) == 2


def test_revise_tailored_content_raises_when_both_excluded() -> None:
    background_path = ROOT / "background.example.md"
    with pytest.raises(ValueError, match="No AI sections"):
        revise_tailored_content(
            "Role",
            EMPTY_AI_OUTPUT,
            "Change something",
            background_path,
            endpoint_url="http://localhost:13305/api/v1",
            model_name="test-model",
            include_summary=False,
            include_skills=False,
        )
