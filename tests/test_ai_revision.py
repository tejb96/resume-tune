"""Tests for AI revision helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from resume_tune.llm.ai import (
    AIResponseError,
    filter_skill_categories,
    format_revision_user_message,
    format_skill_categories,
    parse_skill_categories,
    parse_response,
    revise_tailored_content,
    _call_with_retries,
)

ROOT = Path(__file__).resolve().parent.parent


def test_format_revision_user_message_includes_feedback() -> None:
    categories = [
        {"name": "Languages", "skills": ["Python", "Go"]},
    ]
    message = format_revision_user_message(
        job_description="Backend role",
        summary="Current summary.",
        skill_categories=categories,
        feedback="Make skills more backend-focused.",
    )

    assert "Backend role" in message
    assert "Current summary." in message
    assert "Languages: Python, Go" in message
    assert "Make skills more backend-focused." in message


def test_parse_skill_categories_round_trip() -> None:
    text = "Languages: Python, Go\nCloud: AWS, Docker"
    parsed = parse_skill_categories(text)
    assert parsed == [
        {"name": "Languages", "skills": ["Python", "Go"]},
        {"name": "Cloud", "skills": ["AWS", "Docker"]},
    ]
    assert format_skill_categories(parsed) == text


def test_parse_response_accepts_skill_categories() -> None:
    payload = {
        "summary": "Backend engineer.",
        "skill_categories": [
            {"name": "Languages", "skills": ["Python", "Go"]},
        ],
    }
    parsed = parse_response(json.dumps(payload))
    assert parsed["summary"] == "Backend engineer."
    assert parsed["skill_categories"][0]["name"] == ""


def test_parse_response_migrates_legacy_skills() -> None:
    payload = {"summary": "Engineer.", "skills": ["Python", "Go"]}
    parsed = parse_response(json.dumps(payload))
    assert parsed["skill_categories"] == [{"name": "Skills", "skills": ["Python", "Go"]}]


def test_filter_skill_categories_drops_unknown() -> None:
    categories = [
        {"name": "Testing", "skills": ["Jest", "Python"]},
    ]
    skills_map = {"languages": ["Python", "Go"]}
    filtered, dropped = filter_skill_categories(categories, skills_map)
    assert dropped == ["Jest"]
    assert filtered == [{"name": "Testing", "skills": ["Python"]}]


def test_revise_tailored_content_calls_model_and_returns_json() -> None:
    background_path = ROOT / "background.example.md"
    current_output = {
        "summary": "Engineer with Python experience.",
        "skill_categories": [{"name": "Languages", "skills": ["Python", "Go"]}],
    }
    revised = {
        "summary": "Backend engineer focused on Python APIs.",
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

    passthrough_guard = lambda cats, *_a, **_k: (
        cats,
        {
            "removed_skills": [],
            "deduped_skills": [],
            "added_skills": [],
            "line_utilization": [],
        },
    )
    with (
        patch("resume_tune.llm.ai.OpenAI", return_value=mock_client),
        patch("resume_tune.llm.ai.apply_skills_guardrails", side_effect=passthrough_guard),
    ):
        result, warnings, _packer = revise_tailored_content(
            "Backend engineer role",
            current_output,
            "Focus on backend Python work.",
            background_path,
            endpoint_url="http://localhost:13305/api/v1",
            model_name="test-model",
        )

    assert result["summary"] == revised["summary"]
    assert result["skill_categories"][0]["skills"] == ["Python", "Go"]
    assert result["skill_categories"][1]["skills"] == ["PostgreSQL"]
    assert all(cat["name"] == "" for cat in result["skill_categories"])
    assert isinstance(warnings, list)
    mock_client.chat.completions.create.assert_called_once()


def test_revise_tailored_content_requires_feedback() -> None:
    background_path = ROOT / "background.example.md"
    current_output = {
        "summary": "Summary.",
        "skill_categories": [{"name": "Languages", "skills": ["Python"]}],
    }

    with pytest.raises(ValueError, match="Revision feedback cannot be empty"):
        revise_tailored_content(
            "Job description",
            current_output,
            "   ",
            background_path,
            endpoint_url="http://localhost:13305/api/v1",
            model_name="test-model",
        )


def test_revise_tailored_content_retries_on_invalid_json() -> None:
    background_path = ROOT / "background.example.md"
    current_output = {
        "summary": "Summary.",
        "skill_categories": [{"name": "Languages", "skills": ["Python"]}],
    }
    revised = {
        "summary": "Better summary.",
        "skill_categories": [{"name": "Languages", "skills": ["Python", "Go"]}],
    }

    bad_choice = MagicMock()
    bad_choice.finish_reason = "stop"
    bad_choice.message.content = "not json"

    good_choice = MagicMock()
    good_choice.finish_reason = "stop"
    good_choice.message.content = json.dumps(revised)

    mock_response_bad = MagicMock()
    mock_response_bad.choices = [bad_choice]
    mock_response_good = MagicMock()
    mock_response_good.choices = [good_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        mock_response_bad,
        mock_response_good,
    ]

    passthrough_guard = lambda cats, *_a, **_k: (
        cats,
        {
            "removed_skills": [],
            "deduped_skills": [],
            "added_skills": [],
            "line_utilization": [],
        },
    )
    with (
        patch("resume_tune.llm.ai.OpenAI", return_value=mock_client),
        patch("resume_tune.llm.ai.apply_skills_guardrails", side_effect=passthrough_guard),
    ):
        result, _warnings, _packer = revise_tailored_content(
            "Job description",
            current_output,
            "Improve the summary.",
            background_path,
            endpoint_url="http://localhost:13305/api/v1",
            model_name="test-model",
        )

    assert result["summary"] == revised["summary"]
    assert result["skill_categories"][0]["name"] == ""
    assert result["skill_categories"][0]["skills"] == ["Python", "Go"]
    assert mock_client.chat.completions.create.call_count == 2


def test_revise_tailored_content_raises_after_invalid_json_exhausted() -> None:
    background_path = ROOT / "background.example.md"
    current_output = {
        "summary": "Summary.",
        "skill_categories": [{"name": "Languages", "skills": ["Python"]}],
    }

    bad_choice = MagicMock()
    bad_choice.finish_reason = "stop"
    bad_choice.message.content = "not json"

    mock_response = MagicMock()
    mock_response.choices = [bad_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("resume_tune.llm.ai.OpenAI", return_value=mock_client):
        with pytest.raises(AIResponseError):
            revise_tailored_content(
                "Job description",
                current_output,
                "Improve the summary.",
                background_path,
                endpoint_url="http://localhost:13305/api/v1",
                model_name="test-model",
            )

    assert mock_client.chat.completions.create.call_count == 3


def test_call_with_retries_accepts_partial_dropped_skills_without_retry() -> None:
    payload = {
        "skill_categories": [
            {"name": "", "skills": ["Python", "FakeTool", "Go"]},
        ],
    }
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = json.dumps(payload)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    skills_map = {"languages": ["Python", "Go"]}
    passthrough_guard = lambda cats, *_a, **_k: (
        cats or [{"name": "", "skills": ["Python", "Go"]}],
        {
            "removed_skills": [],
            "deduped_skills": [],
            "added_skills": [],
            "line_utilization": [],
        },
    )
    with patch("resume_tune.llm.ai.apply_skills_guardrails", side_effect=passthrough_guard):
        parsed, dropped, _packer = _call_with_retries(
            mock_client,
            model_name="test-model",
            messages=[{"role": "user", "content": "JD"}],
            background_text="",
            job_description="Python role",
            skills_map=skills_map,
            ai_output_max_chars=600,
            include_summary=False,
            include_skills=True,
            max_skill_categories=3,
        )

    assert "FakeTool" in dropped
    mock_client.chat.completions.create.assert_called_once()
    assert parsed["skill_categories"]
