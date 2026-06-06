"""OpenAI-compatible local LLM client for tailored summary and skills."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

# Reference fit that keeps AI summary + skills on page 2 of the DOCX template.
DEFAULT_AI_OUTPUT_MAX_CHARS = 967

SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer. Given the candidate background below and a job description, produce a tailored professional summary and a curated skills list.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"summary": "<string>", "skills": ["<skill>", ...]}}
- TOTAL LENGTH (critical): Count every character in "summary" plus every skill label in "skills". The combined total must be at most {max_chars} characters. A good fit is ~780 characters for the summary and ~185 across ~11 concise skill labels, but you may shift length between summary and skills as needed.
- "summary": 3-5 sentences, third person or implied first person, targeted to the job description. Use only facts from the background. Prefer tight phrasing over extra detail when near the limit.
- "skills": Include as many relevant items as fit within the total character budget (typically 10-14). Order by relevance to the job description. Only include skills evidenced in the background. No invented skills. Prefer concise labels (1-4 words when possible; standard technology names over sentence-length phrases).
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


def ai_output_char_count(summary: str, skills: list[str]) -> int:
    """Characters in summary plus all skill labels (DOCX body text for AI sections)."""
    return len(summary.strip()) + sum(len(skill) for skill in skills)


def enforce_output_budget(
    summary: str,
    skills: list[str],
    max_chars: int,
) -> dict[str, Any]:
    """Trim least-important skills, then summary sentences, to fit the page budget."""
    summary = summary.strip()
    skills = list(skills)

    while skills and ai_output_char_count(summary, skills) > max_chars:
        skills.pop()

    if ai_output_char_count(summary, skills) <= max_chars:
        return {"summary": summary, "skills": skills}

    sentences = re.split(r"(?<=[.!?])\s+", summary)
    while len(sentences) > 1 and ai_output_char_count(" ".join(sentences), skills) > max_chars:
        sentences.pop()
    summary = " ".join(sentences).strip()

    if ai_output_char_count(summary, skills) > max_chars:
        skill_chars = sum(len(s) for s in skills)
        summary_room = max_chars - skill_chars
        if summary_room > 0 and len(summary) > summary_room:
            summary = summary[:summary_room].rsplit(" ", 1)[0].rstrip(".,;")
            if summary and summary[-1] not in ".!?":
                summary += "."
        elif len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip(".,;")
            if summary and summary[-1] not in ".!?":
                summary += "."
            skills = []

    return {"summary": summary, "skills": skills}


def build_system_prompt(background_text: str, *, max_chars: int = DEFAULT_AI_OUTPUT_MAX_CHARS) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        background_text=background_text,
        max_chars=max_chars,
    )


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


def log_ai_response(choice: Any) -> None:
    """Print model fields to the console for debugging."""
    if not choice or not choice.message:
        print("[ai] response: (no message)")
        return
    msg = choice.message
    print(f"[ai] finish_reason: {choice.finish_reason!r}")
    print(f"[ai] content: {msg.content!r}")
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        print(f"[ai] reasoning_content: {reasoning!r}")


def check_skills_against_background(skills: list[str], background_text: str) -> list[str]:
    """Return skills that do not appear (case-insensitive) anywhere in background text."""
    bg_lower = background_text.lower()
    return [skill for skill in skills if skill.lower() not in bg_lower]


def _shorten_output_message(char_count: int, max_chars: int) -> str:
    return (
        f"Your previous JSON used {char_count} characters total "
        f"(summary + skill labels). Shorten it to at most {max_chars} "
        "characters combined while keeping the most job-relevant facts. "
        "Use tighter summary phrasing and/or fewer, shorter skill labels. "
        "Reply again with ONLY the JSON object, no markdown fences."
    )


def generate_tailored_content(
    job_description: str,
    background_path: Path,
    *,
    endpoint_url: str,
    model_name: str,
    api_key: str = "ollama",
    ai_output_max_chars: int = DEFAULT_AI_OUTPUT_MAX_CHARS,
) -> tuple[dict[str, Any], list[str]]:
    """
    Call local LLM and return (parsed JSON, skill warnings).

    skill warnings lists skills not found verbatim in background text.
    Retries on JSON parse failure and when output exceeds the character budget.
    """
    if not job_description.strip():
        raise ValueError("Job description cannot be empty")
    if ai_output_max_chars < 1:
        raise ValueError("ai_output_max_chars must be positive")

    background_text = read_background_text(background_path)
    system_prompt = build_system_prompt(background_text, max_chars=ai_output_max_chars)
    client = OpenAI(base_url=endpoint_url, api_key=api_key)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": job_description.strip()},
    ]

    last_error: AIResponseError | None = None
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=512,
                extra_body={"enable_thinking": False},
            )
        except APIConnectionError as exc:
            raise AIResponseError(
                f"Cannot connect to local model API at {endpoint_url}. "
                "Is your local OpenAI-compatible API running?"
            ) from exc
        except APIStatusError as exc:
            raise AIResponseError(f"Model API error ({exc.status_code}): {exc.message}") from exc

        choice = response.choices[0] if response.choices else None
        log_ai_response(choice)
        if not choice or not choice.message or not choice.message.content:
            raise AIResponseError("Model returned an empty response")

        try:
            parsed = parse_response(choice.message.content)
        except AIResponseError as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON. "
                            "Reply again with ONLY the JSON object, no markdown fences."
                        ),
                    }
                ]
            continue

        char_count = ai_output_char_count(parsed["summary"], parsed["skills"])
        print(f"[ai] output chars: {char_count}/{ai_output_max_chars}")

        if char_count <= ai_output_max_chars:
            warnings = check_skills_against_background(parsed["skills"], background_text)
            return parsed, warnings

        last_error = AIResponseError(
            f"Model output exceeded character budget ({char_count} > {ai_output_max_chars})"
        )
        if attempt < max_attempts - 1:
            messages = messages + [
                {"role": "user", "content": _shorten_output_message(char_count, ai_output_max_chars)},
            ]
            continue

        trimmed = enforce_output_budget(
            parsed["summary"],
            parsed["skills"],
            ai_output_max_chars,
        )
        print(
            "[ai] trimmed output chars: "
            f"{ai_output_char_count(trimmed['summary'], trimmed['skills'])}/{ai_output_max_chars}"
        )
        warnings = check_skills_against_background(trimmed["skills"], background_text)
        return trimmed, warnings

    assert last_error is not None
    raise last_error
