"""OpenAI-compatible local LLM client for tailored summary and skills."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

# Reference fit that keeps AI summary + skills on page 2 of the DOCX template.
DEFAULT_AI_OUTPUT_MAX_CHARS = 967

SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer. Given the candidate background below and a job description, produce a tailored professional summary and categorized skills.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"summary": "<string>", "skill_categories": [{{"name": "<category>", "skills": ["<skill>", ...]}}, ...]}}
- TOTAL LENGTH (critical): Count every character in "summary", every category "name", and every skill string. The combined total must be at most {max_chars} characters.
- "summary": 3-5 sentences, third person or implied first person, targeted to the job description. Use only facts from the background. Prefer tight phrasing over extra detail when near the limit.
- "skill_categories": 4-6 categories max. Each category has a short label (e.g. "Languages", "Backend", "Cloud & DevOps", "Databases") and 3-8 concise skills comma-joinable in one line. Order categories by relevance to the job description.
- Skills must be evidenced in CANDIDATE BACKGROUND only. Do NOT invent skills. Do NOT add testing frameworks, tools, or technologies mentioned only in the job description unless they appear in CANDIDATE BACKGROUND.
- Do not include any keys other than "summary" and "skill_categories".

CANDIDATE BACKGROUND:
{background_text}
"""

REVISION_SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer revising a tailored professional summary and categorized skills list based on user feedback.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"summary": "<string>", "skill_categories": [{{"name": "<category>", "skills": ["<skill>", ...]}}, ...]}}
- TOTAL LENGTH (critical): Count every character in "summary", every category "name", and every skill string. The combined total must be at most {max_chars} characters.
- Apply the user's revision request while keeping content targeted to the job description.
- Use only facts from the background. Skills must be evidenced in CANDIDATE BACKGROUND only. Do NOT invent skills. Do NOT add skills from the job description unless they appear in CANDIDATE BACKGROUND.
- Do not include any keys other than "summary" and "skill_categories".

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


def normalize_ai_output(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize AI output to skill_categories schema (migrate legacy flat skills)."""
    summary = data.get("summary", "").strip()
    if "skill_categories" in data:
        categories = _validate_skill_categories(data["skill_categories"])
        return {"summary": summary, "skill_categories": categories}

    skills = data.get("skills", [])
    if isinstance(skills, list) and skills:
        return {
            "summary": summary,
            "skill_categories": [{"name": "Skills", "skills": [s.strip() for s in skills if isinstance(s, str) and s.strip()]}],
        }

    raise ValueError("AI output missing non-empty 'skill_categories' or 'skills'")


def _validate_skill_categories(categories: Any) -> list[dict[str, Any]]:
    if not isinstance(categories, list) or not categories:
        raise ValueError("'skill_categories' must be a non-empty list")

    validated: list[dict[str, Any]] = []
    for i, cat in enumerate(categories):
        if not isinstance(cat, dict):
            raise ValueError(f"skill_categories[{i}] must be a mapping")
        name = cat.get("name", "")
        skills = cat.get("skills", [])
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"skill_categories[{i}].name must be a non-empty string")
        if not isinstance(skills, list) or not skills:
            raise ValueError(f"skill_categories[{i}].skills must be a non-empty list")
        if not all(isinstance(s, str) and s.strip() for s in skills):
            raise ValueError(f"skill_categories[{i}].skills must be non-empty strings")
        validated.append({"name": name.strip(), "skills": [s.strip() for s in skills]})
    return validated


def ai_output_char_count(summary: str, skill_categories: list[dict[str, Any]]) -> int:
    """Characters in summary plus category names and skill labels."""
    total = len(summary.strip())
    for cat in skill_categories:
        total += len(cat["name"])
        total += sum(len(skill) for skill in cat["skills"])
    return total


def enforce_output_budget(
    summary: str,
    skill_categories: list[dict[str, Any]],
    max_chars: int,
) -> dict[str, Any]:
    """Trim least-important skills/categories, then summary sentences, to fit the char budget."""
    summary = summary.strip()
    categories = [
        {"name": cat["name"], "skills": list(cat["skills"])}
        for cat in skill_categories
    ]

    while categories and ai_output_char_count(summary, categories) > max_chars:
        last = categories[-1]
        if last["skills"]:
            last["skills"].pop()
        if not last["skills"]:
            categories.pop()

    if ai_output_char_count(summary, categories) <= max_chars:
        return {"summary": summary, "skill_categories": categories}

    sentences = re.split(r"(?<=[.!?])\s+", summary)
    while len(sentences) > 1 and ai_output_char_count(" ".join(sentences), categories) > max_chars:
        sentences.pop()
    summary = " ".join(sentences).strip()

    if ai_output_char_count(summary, categories) > max_chars:
        cat_chars = sum(
            len(c["name"]) + sum(len(s) for s in c["skills"])
            for c in categories
        )
        summary_room = max_chars - cat_chars
        if summary_room > 0 and len(summary) > summary_room:
            summary = summary[:summary_room].rsplit(" ", 1)[0].rstrip(".,;")
            if summary and summary[-1] not in ".!?":
                summary += "."
        elif len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip(".,;")
            if summary and summary[-1] not in ".!?":
                summary += "."
            categories = []

    return {"summary": summary, "skill_categories": categories}


def build_system_prompt(background_text: str, *, max_chars: int = DEFAULT_AI_OUTPUT_MAX_CHARS) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        background_text=background_text,
        max_chars=max_chars,
    )


def build_revision_system_prompt(
    background_text: str,
    *,
    max_chars: int = DEFAULT_AI_OUTPUT_MAX_CHARS,
) -> str:
    return REVISION_SYSTEM_PROMPT_TEMPLATE.format(
        background_text=background_text,
        max_chars=max_chars,
    )


def format_skill_categories(skill_categories: list[dict[str, Any]]) -> str:
    """Format categories for display in revision messages and manual edit."""
    return "\n".join(
        f"{cat['name']}: {', '.join(cat['skills'])}" for cat in skill_categories
    )


def parse_skill_categories(text: str) -> list[dict[str, Any]]:
    """Parse 'Category: skill1, skill2' lines from manual edit input."""
    categories: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, skills_part = line.split(":", 1)
        skills = [s.strip() for s in skills_part.split(",") if s.strip()]
        if name.strip() and skills:
            categories.append({"name": name.strip(), "skills": skills})
    return categories


def format_revision_user_message(
    *,
    job_description: str,
    summary: str,
    skill_categories: list[dict[str, Any]],
    feedback: str,
) -> str:
    skills_block = format_skill_categories(skill_categories)
    return (
        "Revise the professional summary and skills below based on the user's request.\n\n"
        f"JOB DESCRIPTION:\n{job_description.strip()}\n\n"
        f"CURRENT SUMMARY:\n{summary.strip()}\n\n"
        f"CURRENT SKILLS:\n{skills_block}\n\n"
        f"USER REQUEST:\n{feedback.strip()}"
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

    allowed_keys = {"summary", "skill_categories"}
    legacy_keys = {"summary", "skills"}
    if set(data.keys()) == legacy_keys:
        return normalize_ai_output(data)

    if set(data.keys()) != allowed_keys:
        raise AIResponseError(
            "Model response must contain only 'summary' and 'skill_categories', "
            f"got keys: {sorted(data.keys())}"
        )

    summary = data["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise AIResponseError("'summary' must be a non-empty string")

    try:
        categories = _validate_skill_categories(data["skill_categories"])
    except ValueError as exc:
        raise AIResponseError(str(exc)) from exc

    return {"summary": summary.strip(), "skill_categories": categories}


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


def skill_in_background(skill: str, background_text: str) -> bool:
    """Return True when skill appears as a case-insensitive substring in background."""
    return skill.lower() in background_text.lower()


def filter_skill_categories(
    skill_categories: list[dict[str, Any]],
    background_text: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop skills not evidenced in background; return filtered categories and dropped skills."""
    dropped: list[str] = []
    filtered: list[dict[str, Any]] = []
    for cat in skill_categories:
        kept = [s for s in cat["skills"] if skill_in_background(s, background_text)]
        dropped.extend(s for s in cat["skills"] if s not in kept)
        if kept:
            filtered.append({"name": cat["name"], "skills": kept})
    return filtered, dropped


def check_skills_against_background(
    skill_categories: list[dict[str, Any]],
    background_text: str,
) -> list[str]:
    """Return skills that do not appear (case-insensitive) anywhere in background text."""
    _, dropped = filter_skill_categories(skill_categories, background_text)
    return dropped


def _shorten_output_message(char_count: int, max_chars: int) -> str:
    return (
        f"Your previous JSON used {char_count} characters total "
        f"(summary + category names + skill labels). Shorten it to at most {max_chars} "
        "characters combined while keeping the most job-relevant facts. "
        "Use tighter summary phrasing and/or fewer, shorter skills per category. "
        "Reply again with ONLY the JSON object, no markdown fences."
    )


def _rejected_skills_message(rejected: list[str]) -> str:
    skills_list = ", ".join(rejected)
    return (
        f"The following skills are not evidenced in CANDIDATE BACKGROUND and were removed: "
        f"{skills_list}. Replace them with skills that appear verbatim in the background. "
        "Do not add skills from the job description unless they appear in CANDIDATE BACKGROUND. "
        "Reply again with ONLY the JSON object, no markdown fences."
    )


def _call_with_retries(
    client: OpenAI,
    *,
    model_name: str,
    messages: list[dict[str, str]],
    background_text: str,
    ai_output_max_chars: int,
) -> tuple[dict[str, Any], list[str]]:
    """Call the model with JSON/budget/skill-grounding retries."""
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
                f"Cannot connect to local model API. "
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

        filtered, dropped = filter_skill_categories(parsed["skill_categories"], background_text)
        if not filtered:
            last_error = AIResponseError("All skills were rejected as not evidenced in background")
            if attempt < max_attempts - 1:
                messages = messages + [
                    {"role": "user", "content": _rejected_skills_message(dropped)},
                ]
            continue

        parsed = {"summary": parsed["summary"], "skill_categories": filtered}
        char_count = ai_output_char_count(parsed["summary"], parsed["skill_categories"])
        print(f"[ai] output chars: {char_count}/{ai_output_max_chars}")

        if dropped and attempt < max_attempts - 1:
            messages = messages + [
                {"role": "user", "content": _rejected_skills_message(dropped)},
            ]
            continue

        if char_count <= ai_output_max_chars:
            return parsed, dropped

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
            parsed["skill_categories"],
            ai_output_max_chars,
        )
        print(
            "[ai] trimmed output chars: "
            f"{ai_output_char_count(trimmed['summary'], trimmed['skill_categories'])}/{ai_output_max_chars}"
        )
        return trimmed, dropped

    assert last_error is not None
    raise last_error


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
    Call local LLM and return (parsed JSON, dropped skill warnings).

    Retries on JSON parse failure, hallucinated skills, and character budget overflow.
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

    try:
        return _call_with_retries(
            client,
            model_name=model_name,
            messages=messages,
            background_text=background_text,
            ai_output_max_chars=ai_output_max_chars,
        )
    except AIResponseError as exc:
        if "Cannot connect to local model API" in str(exc):
            raise AIResponseError(
                f"Cannot connect to local model API at {endpoint_url}. "
                "Is your local OpenAI-compatible API running?"
            ) from exc
        raise


def revise_tailored_content(
    job_description: str,
    current_output: dict[str, Any],
    feedback: str,
    background_path: Path,
    *,
    endpoint_url: str,
    model_name: str,
    api_key: str = "ollama",
    ai_output_max_chars: int = DEFAULT_AI_OUTPUT_MAX_CHARS,
    revision_history: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Revise summary/skills based on user feedback.

    revision_history holds prior chat turns as {"role": "user"|"assistant", "content": "..."}.
    """
    if not job_description.strip():
        raise ValueError("Job description cannot be empty")
    if not feedback.strip():
        raise ValueError("Revision feedback cannot be empty")
    if ai_output_max_chars < 1:
        raise ValueError("ai_output_max_chars must be positive")

    normalized = normalize_ai_output(current_output)
    summary = normalized["summary"]
    skill_categories = normalized["skill_categories"]
    if not summary:
        raise ValueError("Current summary cannot be empty")
    if not skill_categories:
        raise ValueError("Current skills cannot be empty")

    background_text = read_background_text(background_path)
    system_prompt = build_revision_system_prompt(background_text, max_chars=ai_output_max_chars)
    client = OpenAI(base_url=endpoint_url, api_key=api_key)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if revision_history:
        messages.extend(revision_history)

    messages.append(
        {
            "role": "user",
            "content": format_revision_user_message(
                job_description=job_description,
                summary=summary,
                skill_categories=skill_categories,
                feedback=feedback,
            ),
        }
    )

    try:
        return _call_with_retries(
            client,
            model_name=model_name,
            messages=messages,
            background_text=background_text,
            ai_output_max_chars=ai_output_max_chars,
        )
    except AIResponseError as exc:
        if "Cannot connect to local model API" in str(exc):
            raise AIResponseError(
                f"Cannot connect to local model API at {endpoint_url}. "
                "Is your local OpenAI-compatible API running?"
            ) from exc
        raise
