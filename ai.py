"""OpenAI-compatible local LLM client for tailored summary and skills."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer. Given the candidate background below and a job description, produce a tailored professional summary and a curated skills list.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"summary": "<string>", "skills": ["<skill>", ...]}}
- "summary": 3-5 sentences, third person or implied first person, targeted to the job description. Use only facts from the background.
- "skills": 12-18 items max, ordered by relevance to the job description. Only include skills evidenced in the background. No invented skills.
- Do not include any keys other than "summary" and "skills".

CANDIDATE BACKGROUND:
{background_text}
"""


class AIResponseError(Exception):
    """Raised when the model response cannot be parsed or validated."""


def read_background_text(path: Path) -> str:
    """Read full background file for the system prompt."""
    if not path.exists():
        raise FileNotFoundError(f"Background file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_system_prompt(background_text: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(background_text=background_text)


def strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_response(raw: str) -> dict[str, Any]:
    """Parse and validate model JSON response."""
    cleaned = strip_json_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        snippet = cleaned[:500] + ("..." if len(cleaned) > 500 else "")
        raise AIResponseError(
            f"Model returned invalid JSON: {exc}. Response snippet: {snippet!r}"
        ) from exc

    if not isinstance(data, dict):
        raise AIResponseError("Model response must be a JSON object")

    if set(data.keys()) != {"summary", "skills"}:
        raise AIResponseError(
            f"Model response must contain only 'summary' and 'skills', got keys: {sorted(data.keys())}"
        )

    summary = data["summary"]
    skills = data["skills"]

    if not isinstance(summary, str) or not summary.strip():
        raise AIResponseError("'summary' must be a non-empty string")

    if not isinstance(skills, list) or not skills:
        raise AIResponseError("'skills' must be a non-empty list")

    if not all(isinstance(s, str) and s.strip() for s in skills):
        raise AIResponseError("'skills' must be a non-empty list of strings")

    return {"summary": summary.strip(), "skills": [s.strip() for s in skills]}


def check_skills_against_background(skills: list[str], background_text: str) -> list[str]:
    """Return skills that do not appear (case-insensitive) anywhere in background text."""
    bg_lower = background_text.lower()
    return [skill for skill in skills if skill.lower() not in bg_lower]


def generate_tailored_content(
    job_description: str,
    background_path: Path,
    *,
    endpoint_url: str,
    model_name: str,
    api_key: str = "ollama",
) -> tuple[dict[str, Any], list[str]]:
    """
    Call local LLM and return (parsed JSON, skill warnings).

    skill warnings lists skills not found verbatim in background text.
    Retries once on JSON parse failure.
    """
    if not job_description.strip():
        raise ValueError("Job description cannot be empty")

    background_text = read_background_text(background_path)
    system_prompt = build_system_prompt(background_text)
    client = OpenAI(base_url=endpoint_url, api_key=api_key)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": job_description.strip()},
    ]

    last_error: AIResponseError | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
            )
        except APIConnectionError as exc:
            raise AIResponseError(
                f"Cannot connect to local model API at {endpoint_url}. "
                "Is your local OpenAI-compatible API running?"
            ) from exc
        except APIStatusError as exc:
            raise AIResponseError(f"Model API error ({exc.status_code}): {exc.message}") from exc

        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message or not choice.message.content:
            raise AIResponseError("Model returned an empty response")

        try:
            parsed = parse_response(choice.message.content)
            warnings = check_skills_against_background(parsed["skills"], background_text)
            return parsed, warnings
        except AIResponseError as exc:
            last_error = exc
            if attempt == 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON. "
                            "Reply again with ONLY the JSON object, no markdown fences."
                        ),
                    }
                ]

    assert last_error is not None
    raise last_error
