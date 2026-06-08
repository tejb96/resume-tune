"""Tests for skipping LLM generation of excluded AI sections."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai import (
    EMPTY_AI_OUTPUT,
    enforce_output_budget,
    generate_tailored_content,
    normalize_ai_output,
    parse_response,
    revise_tailored_content,
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
    assert trimmed["skill_categories"]


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
        ai_output, warnings = generate_tailored_content(
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

    with patch("ai.OpenAI", return_value=mock_client):
        ai_output, _warnings = generate_tailored_content(
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
        ai_output, _warnings = generate_tailored_content(
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

    with patch("ai.OpenAI", return_value=mock_client):
        result, _warnings = revise_tailored_content(
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
