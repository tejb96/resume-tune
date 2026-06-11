"""OpenAI-compatible local LLM client for tailored summary and skills."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

# Reference fit that keeps AI summary + skills on page 2 of the DOCX template.
DEFAULT_AI_OUTPUT_MAX_CHARS = 967
DEFAULT_MAX_SKILL_CATEGORIES = 4
DEFAULT_MAX_SKILLS_PER_CATEGORY = 5
DEFAULT_MAX_CHARS_PER_SKILL_LINE = 88

SKILLS_LAYOUT_RULES = (
    '- "skill_categories": exactly {max_skill_categories} lines when possible. '
    "Each line must fit on one row (about {max_chars_per_skill_line} characters). "
    'Use "name": "" on every line — no category labels; group similar skills per line. '
    "Line 1: languages/frameworks · Line 2: AI/ML · Line 3: cloud/devops · Line 4: tools. "
    "Use exact spellings from ALLOWED SKILLS. Order skills by job relevance within each line. "
    "Do not list redundant pairs (TensorFlow.js OR TensorFlow, not both; Tailwind CSS OR Tailwind). "
    "Do not add skills merely to fill space."
)

SKILLS_MAP_RULES = (
    "- Select skills ONLY from ALLOWED SKILLS below (exact spelling). "
    "Do not add skills outside the map. "
    "Include STRONG job matches first; add evidenced supporting skills only when they fit the role. "
    "Skip niche AI/CV tools (e.g. YOLOv8, Ollama) when the job is full-stack web unless the JD names them. "
    'Use "name": "" on every line.'
)

SKILLS_LINE_CAPACITY_RULES = (
    "- LINE CAPACITY: Up to {max_skill_categories} lines, all with empty \"name\". "
    "Each line must fit on one row (about {max_chars_per_skill_line} characters). "
    "Pack job-relevant skills only; guardrails will rebuild lines from the skills map."
)

SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer. Given the candidate background below and a job description, produce a tailored professional summary and categorized skills.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"summary": "<string>", "skill_categories": [{{"name": "<category>", "skills": ["<skill>", ...]}}, ...]}}
- TOTAL LENGTH (critical): Count every character in "summary", every category "name", and every skill string. The combined total must be at most {max_chars} characters.
- "summary": 3-5 sentences, third person or implied first person, targeted to the job description. Use only facts from the background. Prefer tight phrasing over extra detail when near the limit.
- {skills_layout_rules}
- {skills_map_rules}
- Do not include any keys other than "summary" and "skill_categories".

{skill_hints}{allowed_skills}

CANDIDATE BACKGROUND (narrative context):
{background_text}
"""

REVISION_SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer revising a tailored professional summary and categorized skills list based on user feedback.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"summary": "<string>", "skill_categories": [{{"name": "<category>", "skills": ["<skill>", ...]}}, ...]}}
- TOTAL LENGTH (critical): Count every character in "summary", every category "name", and every skill string. The combined total must be at most {max_chars} characters.
- Apply the user's revision request while keeping content targeted to the job description.
- {skills_layout_rules}
- {skills_map_rules}
- Do not include any keys other than "summary" and "skill_categories".

{skill_hints}{allowed_skills}

CANDIDATE BACKGROUND (narrative context):
{background_text}
"""

SKILLS_ONLY_SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer. Given the allowed skills map and job description, produce tailored categorized skills.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"skill_categories": [{{"name": "<category>", "skills": ["<skill>", ...]}}, ...]}}
- {skills_line_capacity_rules}
- {skills_layout_rules}
- {skills_map_rules}
- Do not include any keys other than "skill_categories".

{skill_hints}{allowed_skills}

CANDIDATE BACKGROUND (narrative context):
{background_text}
"""

SUMMARY_ONLY_SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer. Given the candidate background below and a job description, produce a tailored professional summary.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"summary": "<string>"}}
- TOTAL LENGTH (critical): Count every character in "summary". The total must be at most {max_chars} characters.
- "summary": 3-5 sentences, third person or implied first person, targeted to the job description. Use only facts from the background. Prefer tight phrasing over extra detail when near the limit.
- Do not include any keys other than "summary".

CANDIDATE BACKGROUND:
{background_text}
"""

SKILLS_ONLY_REVISION_SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer revising a tailored categorized skills list based on user feedback.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"skill_categories": [{{"name": "<category>", "skills": ["<skill>", ...]}}, ...]}}
- {skills_line_capacity_rules}
- Apply the user's revision request while keeping content targeted to the job description.
- {skills_layout_rules}
- {skills_map_rules}
- Do not include any keys other than "skill_categories".

{skill_hints}{allowed_skills}

CANDIDATE BACKGROUND (narrative context):
{background_text}
"""

SUMMARY_ONLY_REVISION_SYSTEM_PROMPT_TEMPLATE = """You are a professional resume writer revising a tailored professional summary based on user feedback.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema: {{"summary": "<string>"}}
- TOTAL LENGTH (critical): Count every character in "summary". The total must be at most {max_chars} characters.
- Apply the user's revision request while keeping content targeted to the job description.
- Use only facts from the background.
- Do not include any keys other than "summary".

CANDIDATE BACKGROUND:
{background_text}
"""

EMPTY_AI_OUTPUT: dict[str, Any] = {"summary": "", "skill_categories": []}


class AIResponseError(Exception):
    """Raised when the model response cannot be parsed or validated."""


def read_background_text(path: Path) -> str:
    """Read full background file for the system prompt."""
    if not path.exists():
        raise FileNotFoundError(f"Background file not found: {path}")
    return path.read_text(encoding="utf-8")


def normalize_ai_output(
    data: dict[str, Any],
    *,
    include_summary: bool = True,
    include_skills: bool = True,
) -> dict[str, Any]:
    """Normalize AI output to skill_categories schema (migrate legacy flat skills)."""
    if not include_summary and not include_skills:
        return dict(EMPTY_AI_OUTPUT)

    summary = data.get("summary", "").strip()
    categories: list[dict[str, Any]] = []

    if include_skills:
        if "skill_categories" in data:
            categories = _validate_skill_categories(data["skill_categories"])
        else:
            skills = data.get("skills", [])
            if isinstance(skills, list) and skills:
                categories = [
                    {
                        "name": "Skills",
                        "skills": [s.strip() for s in skills if isinstance(s, str) and s.strip()],
                    }
                ]
    else:
        raw_categories = data.get("skill_categories", [])
        categories = raw_categories if isinstance(raw_categories, list) else []

    if include_summary and not summary:
        raise ValueError("AI output missing non-empty 'summary'")
    if include_skills and not categories:
        raise ValueError("AI output missing non-empty 'skill_categories' or 'skills'")

    return {"summary": summary, "skill_categories": categories}


def _coerce_skill_category(cat: Any) -> dict[str, Any]:
    if isinstance(cat, dict):
        return cat
    if isinstance(cat, list):
        if not cat or not all(isinstance(s, str) for s in cat):
            raise ValueError("skill_categories entry must be a mapping or string list")
        if cat[0] == "":
            return {"name": "", "skills": [s.strip() for s in cat[1:] if s.strip()]}
        return {"name": "", "skills": [s.strip() for s in cat if s.strip()]}
    raise ValueError("skill_categories entry must be a mapping")


def _validate_skill_categories(categories: Any) -> list[dict[str, Any]]:
    if not isinstance(categories, list) or not categories:
        raise ValueError("'skill_categories' must be a non-empty list")

    validated: list[dict[str, Any]] = []
    for i, cat in enumerate(categories):
        try:
            coerced = _coerce_skill_category(cat)
        except ValueError as exc:
            raise ValueError(f"skill_categories[{i}] {exc}") from exc
        name = coerced.get("name", "")
        skills = coerced.get("skills", [])
        if not isinstance(name, str):
            raise ValueError(f"skill_categories[{i}].name must be a string (may be empty)")
        if not isinstance(skills, list) or not skills:
            raise ValueError(f"skill_categories[{i}].skills must be a non-empty list")
        if not all(isinstance(s, str) and s.strip() for s in skills):
            raise ValueError(f"skill_categories[{i}].skills must be non-empty strings")
        validated.append({"name": name.strip(), "skills": [s.strip() for s in skills]})
    return validated


def skills_subject_to_char_budget(*, include_summary: bool, include_skills: bool) -> bool:
    """Skills count toward ai_output_max_chars only when summary is also generated."""
    return include_summary and include_skills


def ai_output_char_count(
    summary: str,
    skill_categories: list[dict[str, Any]],
    *,
    include_summary: bool = True,
    include_skills: bool = True,
) -> int:
    """Characters in included AI fields only."""
    total = 0
    if include_summary:
        total += len(summary.strip())
    if include_skills and skills_subject_to_char_budget(
        include_summary=include_summary,
        include_skills=include_skills,
    ):
        for cat in skill_categories:
            total += len(cat["name"])
            total += sum(len(skill) for skill in cat["skills"])
    return total


def enforce_output_budget(
    summary: str,
    skill_categories: list[dict[str, Any]],
    max_chars: int,
    *,
    include_summary: bool = True,
    include_skills: bool = True,
) -> dict[str, Any]:
    """Trim least-important included content to fit the char budget."""
    if not include_summary and not include_skills:
        return dict(EMPTY_AI_OUTPUT)

    summary = summary.strip() if include_summary else ""
    categories = (
        [{"name": cat["name"], "skills": list(cat["skills"])} for cat in skill_categories]
        if include_skills
        else []
    )

    if include_skills and skills_subject_to_char_budget(
        include_summary=include_summary,
        include_skills=include_skills,
    ):
        while categories and ai_output_char_count(
            summary,
            categories,
            include_summary=include_summary,
            include_skills=include_skills,
        ) > max_chars:
            last = categories[-1]
            if last["skills"]:
                last["skills"].pop()
            if not last["skills"]:
                categories.pop()

    if (
        ai_output_char_count(
            summary,
            categories,
            include_summary=include_summary,
            include_skills=include_skills,
        )
        <= max_chars
    ):
        return {"summary": summary, "skill_categories": categories}

    if include_summary:
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        while len(sentences) > 1 and ai_output_char_count(
            " ".join(sentences),
            categories,
            include_summary=True,
            include_skills=include_skills,
        ) > max_chars:
            sentences.pop()
        summary = " ".join(sentences).strip()

        if ai_output_char_count(
            summary,
            categories,
            include_summary=True,
            include_skills=include_skills,
        ) > max_chars:
            if include_skills:
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
            elif len(summary) > max_chars:
                summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip(".,;")
                if summary and summary[-1] not in ".!?":
                    summary += "."

    return {"summary": summary, "skill_categories": categories}


def skill_category_line_length(name: str, skills: list[str]) -> int:
    """Character length of a rendered skills category line."""
    label = name.strip()
    if not skills:
        return len(label) + (2 if label else 0)
    joined = ", ".join(skills)
    if not label:
        return len(joined)
    return len(label) + 2 + len(joined)


def enforce_skills_layout(
    skill_categories: list[dict[str, Any]],
    *,
    max_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_skills_per_category: int = DEFAULT_MAX_SKILLS_PER_CATEGORY,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
    cap_skills_per_category: bool = False,
) -> list[dict[str, Any]]:
    """Cap categories and trim each line to fit on one row at generation time."""
    result: list[dict[str, Any]] = []
    for cat in skill_categories[:max_categories]:
        name = cat["name"].strip()
        skills = [s.strip() for s in cat["skills"] if s.strip()]
        if cap_skills_per_category:
            skills = skills[:max_skills_per_category]
        while skills and skill_category_line_length(name, skills) > max_chars_per_line:
            skills.pop()
        if skills:
            result.append({"name": name, "skills": skills})
    return result


def _skills_on_resume(categories: list[dict[str, Any]]) -> set[str]:
    return {skill.lower() for cat in categories for skill in cat["skills"]}


def _skill_already_listed(skill: str, categories: list[dict[str, Any]]) -> bool:
    skill_lower = skill.lower()
    present = _skills_on_resume(categories)
    if skill_lower in present:
        return True
    if skill_lower == "tailwind css" and "tailwind" in present:
        return True
    if skill_lower == "tailwind" and "tailwind css" in present:
        return True
    return False


def _skill_fits_line(name: str, skills: list[str], new_skill: str, max_chars: int) -> bool:
    return skill_category_line_length(name, skills + [new_skill]) <= max_chars


def _rank_skills_for_packing(
    pool: list[str],
    categories: list[dict[str, Any]],
    *,
    priority_keywords: list[str],
) -> list[str]:
    """Rank evidenced skills for line packing; ATS-missing JD keywords first."""
    present = _skills_on_resume(categories)
    remaining = [s for s in pool if not _skill_already_listed(s, categories)]

    def priority(skill: str) -> tuple[int, int, str]:
        skill_lower = skill.lower()
        for idx, keyword in enumerate(priority_keywords):
            if skill_lower == keyword.lower():
                return (0, idx, skill_lower)
            if keyword.lower() == "tailwind css" and skill_lower == "tailwind":
                return (1, idx, skill_lower)
            if keyword.lower() == "tailwind" and skill_lower == "tailwind css":
                return (0, idx, skill_lower)
        if skill_lower in present:
            return (3, 0, skill_lower)
        return (2, 0, skill_lower)

    return sorted(remaining, key=priority)


def skill_line_utilization(
    skill_categories: list[dict[str, Any]],
    *,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
) -> list[dict[str, Any]]:
    """Per-line character utilization for diagnostics."""
    lines: list[dict[str, Any]] = []
    for cat in skill_categories:
        chars = skill_category_line_length(cat["name"], cat["skills"])
        lines.append(
            {
                "name": cat["name"],
                "chars": chars,
                "max": max_chars_per_line,
            }
        )
    return lines


def dedupe_skill_redundancies(
    skill_categories: list[dict[str, Any]],
    *,
    job_description: str = "",
    jd_keywords: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove redundant skill pairs within each line."""
    from skills_selection import dedupe_redundant_skills

    if jd_keywords is None and job_description.strip():
        from ats import extract_jd_keywords

        jd_keywords = extract_jd_keywords(job_description)
    keywords = jd_keywords or []

    deduped: list[str] = []
    result: list[dict[str, Any]] = []
    for cat in skill_categories:
        skills, removed = dedupe_redundant_skills(
            list(cat["skills"]),
            jd_keywords=keywords,
            job_description=job_description,
        )
        deduped.extend(removed)
        if skills:
            result.append({"name": cat["name"], "skills": skills})
    return result, deduped


def _buckets_on_line(skills: list[str], skills_map: dict[str, list[str]]) -> set[str]:
    from skills_map import buckets_for_skill

    buckets: set[str] = set()
    for skill in skills:
        buckets.update(buckets_for_skill(skill, skills_map))
    return buckets


def _same_bucket_candidates_for_line(
    cat: dict[str, Any],
    skills_map: dict[str, list[str]],
    categories: list[dict[str, Any]],
) -> list[str]:
    """Skills from buckets already represented on this line, not yet on the resume."""
    buckets = _buckets_on_line(cat["skills"], skills_map)
    if not buckets:
        return []

    present = _skills_on_resume(categories)
    candidates: list[str] = []
    seen: set[str] = set()
    for bucket in sorted(buckets):
        for skill in skills_map.get(bucket, []):
            key = skill.lower()
            if key in present or key in seen:
                continue
            seen.add(key)
            candidates.append(skill)
    return candidates


def apply_skills_guardrails(
    skill_categories: list[dict[str, Any]],
    skills_map: dict[str, list[str]],
    job_description: str = "",
    *,
    evidence_text: str = "",
    max_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Map-grounded guardrails: filter, dedupe, relevance-aware prune/pack.

    Returns (final categories, diagnostics).
    """
    from skills_selection import select_relevant_skill_categories

    categories = [
        {"name": cat["name"], "skills": list(cat["skills"])} for cat in skill_categories
    ]
    filtered, removed = filter_skill_categories(categories, skills_map)
    categories = filtered

    categories, deduped = dedupe_skill_redundancies(categories, job_description=job_description)

    packed, selection_info = select_relevant_skill_categories(
        categories,
        skills_map,
        job_description,
        evidence_text,
        max_categories=max_categories,
        max_chars_per_line=max_chars_per_line,
    )

    diagnostics = {
        "removed_skills": removed,
        "deduped_skills": deduped,
        "added_skills": selection_info.get("added_skills", []),
        "removed_irrelevant": selection_info.get("removed_irrelevant", []),
        "dropped_categories": selection_info.get("dropped_categories", []),
        "skill_tiers": selection_info.get("skill_tiers", {}),
        "line_utilization": selection_info.get("line_utilization", []),
    }
    return packed, diagnostics


def pack_skill_lines(
    skill_categories: list[dict[str, Any]],
    background_text: str,
    job_description: str = "",
    *,
    max_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_chars_per_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
    min_tools_line_items: int = 2,
    skills_map: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible alias; prefers skills_map over background_text scan."""
    from skills_map import load_skills_map, parse_core_strengths_markdown

    resolved_map = skills_map or {}
    if not resolved_map and background_text:
        resolved_map = parse_core_strengths_markdown(background_text)
    return apply_skills_guardrails(
        skill_categories,
        resolved_map,
        job_description,
        max_categories=max_categories,
        max_chars_per_line=max_chars_per_line,
    )


def _format_skills_layout_rules(
    *,
    max_skill_categories: int,
    max_skills_per_category: int,
    max_chars_per_skill_line: int,
) -> str:
    return SKILLS_LAYOUT_RULES.format(
        max_skill_categories=max_skill_categories,
        max_skills_per_category=max_skills_per_category,
        max_chars_per_skill_line=max_chars_per_skill_line,
    )


def _format_skills_line_capacity_rules(
    *,
    max_skill_categories: int,
    max_chars_per_skill_line: int,
) -> str:
    return SKILLS_LINE_CAPACITY_RULES.format(
        max_skill_categories=max_skill_categories,
        max_chars_per_skill_line=max_chars_per_skill_line,
    )


def build_evidence_text(
    background_text: str,
    background_data: dict[str, Any] | None = None,
    content_selection: dict[str, Any] | None = None,
) -> str:
    """Combine static resume YAML and narrative text for skills evidence."""
    from skills_selection import extract_narrative_text, flatten_static_evidence_text

    narrative = extract_narrative_text(background_text)
    if background_data is None:
        return narrative
    return flatten_static_evidence_text(
        background_data,
        content_selection=content_selection,
        narrative_text=narrative,
    )


def build_skill_hints_for_prompt(
    skills_map: dict[str, list[str]],
    job_description: str,
    evidence_text: str,
) -> str:
    """Build deterministic skill hint block for the LLM system prompt."""
    from skills_selection import format_skill_hints_for_prompt, score_skills_for_job

    if not skills_map or not job_description.strip():
        return ""
    scores = score_skills_for_job(skills_map, job_description, evidence_text)
    hints = format_skill_hints_for_prompt(scores, skills_map)
    return f"{hints}\n" if hints else ""


def build_system_prompt(
    background_text: str,
    *,
    skills_map: dict[str, list[str]] | None = None,
    max_chars: int = DEFAULT_AI_OUTPUT_MAX_CHARS,
    include_summary: bool = True,
    include_skills: bool = True,
    max_skill_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_skills_per_category: int = DEFAULT_MAX_SKILLS_PER_CATEGORY,
    max_chars_per_skill_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
    job_description: str = "",
    evidence_text: str = "",
) -> str:
    from skills_map import format_skills_map_for_prompt

    skills_layout_rules = _format_skills_layout_rules(
        max_skill_categories=max_skill_categories,
        max_skills_per_category=max_skills_per_category,
        max_chars_per_skill_line=max_chars_per_skill_line,
    )
    skills_line_capacity_rules = _format_skills_line_capacity_rules(
        max_skill_categories=max_skill_categories,
        max_chars_per_skill_line=max_chars_per_skill_line,
    )
    if include_summary and include_skills:
        template = SYSTEM_PROMPT_TEMPLATE
    elif include_skills:
        template = SKILLS_ONLY_SYSTEM_PROMPT_TEMPLATE
    elif include_summary:
        template = SUMMARY_ONLY_SYSTEM_PROMPT_TEMPLATE
    else:
        raise ValueError("At least one AI section must be included to build a system prompt")
    format_kwargs: dict[str, Any] = {
        "background_text": background_text,
        "max_chars": max_chars,
        "skills_layout_rules": skills_layout_rules,
    }
    if include_skills:
        format_kwargs["skills_map_rules"] = SKILLS_MAP_RULES
        format_kwargs["allowed_skills"] = format_skills_map_for_prompt(skills_map or {})
        format_kwargs["skill_hints"] = build_skill_hints_for_prompt(
            skills_map or {},
            job_description,
            evidence_text,
        )
        if not include_summary:
            format_kwargs["skills_line_capacity_rules"] = skills_line_capacity_rules
    else:
        format_kwargs["skill_hints"] = ""
        format_kwargs["allowed_skills"] = ""
    return template.format(**format_kwargs)


def build_revision_system_prompt(
    background_text: str,
    *,
    skills_map: dict[str, list[str]] | None = None,
    max_chars: int = DEFAULT_AI_OUTPUT_MAX_CHARS,
    include_summary: bool = True,
    include_skills: bool = True,
    max_skill_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_skills_per_category: int = DEFAULT_MAX_SKILLS_PER_CATEGORY,
    max_chars_per_skill_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
    job_description: str = "",
    evidence_text: str = "",
) -> str:
    from skills_map import format_skills_map_for_prompt

    skills_layout_rules = _format_skills_layout_rules(
        max_skill_categories=max_skill_categories,
        max_skills_per_category=max_skills_per_category,
        max_chars_per_skill_line=max_chars_per_skill_line,
    )
    skills_line_capacity_rules = _format_skills_line_capacity_rules(
        max_skill_categories=max_skill_categories,
        max_chars_per_skill_line=max_chars_per_skill_line,
    )
    if include_summary and include_skills:
        template = REVISION_SYSTEM_PROMPT_TEMPLATE
    elif include_skills:
        template = SKILLS_ONLY_REVISION_SYSTEM_PROMPT_TEMPLATE
    elif include_summary:
        template = SUMMARY_ONLY_REVISION_SYSTEM_PROMPT_TEMPLATE
    else:
        raise ValueError("At least one AI section must be included to build a revision prompt")
    format_kwargs: dict[str, Any] = {
        "background_text": background_text,
        "max_chars": max_chars,
        "skills_layout_rules": skills_layout_rules,
    }
    if include_skills:
        format_kwargs["skills_map_rules"] = SKILLS_MAP_RULES
        format_kwargs["allowed_skills"] = format_skills_map_for_prompt(skills_map or {})
        format_kwargs["skill_hints"] = build_skill_hints_for_prompt(
            skills_map or {},
            job_description,
            evidence_text,
        )
        if not include_summary:
            format_kwargs["skills_line_capacity_rules"] = skills_line_capacity_rules
    else:
        format_kwargs["skill_hints"] = ""
        format_kwargs["allowed_skills"] = ""
    return template.format(**format_kwargs)


def format_skill_categories(skill_categories: list[dict[str, Any]]) -> str:
    """Format categories for display in revision messages and manual edit."""
    lines: list[str] = []
    for cat in skill_categories:
        name = (cat.get("name") or "").strip()
        skills_text = ", ".join(cat["skills"])
        if name:
            lines.append(f"{name}: {skills_text}")
        else:
            lines.append(skills_text)
    return "\n".join(lines)


def parse_skill_categories(text: str) -> list[dict[str, Any]]:
    """Parse 'Category: skill1, skill2' or plain 'skill1, skill2' lines from manual edit."""
    categories: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            name, skills_part = line.split(":", 1)
            skills = [s.strip() for s in skills_part.split(",") if s.strip()]
            if skills:
                categories.append({"name": name.strip(), "skills": skills})
        else:
            skills = [s.strip() for s in line.split(",") if s.strip()]
            if skills:
                categories.append({"name": "", "skills": skills})
    return categories


def format_revision_user_message(
    *,
    job_description: str,
    summary: str,
    skill_categories: list[dict[str, Any]],
    feedback: str,
    include_summary: bool = True,
    include_skills: bool = True,
) -> str:
    parts = ["Revise the tailored resume content below based on the user's request.", ""]
    parts.append(f"JOB DESCRIPTION:\n{job_description.strip()}")
    if include_summary:
        parts.extend(["", f"CURRENT SUMMARY:\n{summary.strip()}"])
    if include_skills:
        skills_block = format_skill_categories(skill_categories)
        parts.extend(["", f"CURRENT SKILLS:\n{skills_block}"])
    parts.extend(["", f"USER REQUEST:\n{feedback.strip()}"])
    return "\n\n".join(parts)


def strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_response(
    raw: str,
    *,
    include_summary: bool = True,
    include_skills: bool = True,
) -> dict[str, Any]:
    """Parse and validate model JSON response for the active AI sections."""
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

    if include_summary and include_skills:
        allowed_keys = {"summary", "skill_categories"}
        legacy_keys = {"summary", "skills"}
        if set(data.keys()) == legacy_keys:
            return normalize_ai_output(data, include_summary=True, include_skills=True)
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

    if include_skills:
        if set(data.keys()) != {"skill_categories"}:
            raise AIResponseError(
                "Model response must contain only 'skill_categories', "
                f"got keys: {sorted(data.keys())}"
            )
        try:
            categories = _validate_skill_categories(data["skill_categories"])
        except ValueError as exc:
            raise AIResponseError(str(exc)) from exc
        return {"summary": "", "skill_categories": categories}

    if set(data.keys()) != {"summary"}:
        raise AIResponseError(
            "Model response must contain only 'summary', "
            f"got keys: {sorted(data.keys())}"
        )
    summary = data["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise AIResponseError("'summary' must be a non-empty string")
    return {"summary": summary.strip(), "skill_categories": []}


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
    skills_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop skills not in skills_map; return filtered categories and dropped skills."""
    from skills_map import canonical_skill_label

    dropped: list[str] = []
    filtered: list[dict[str, Any]] = []
    for cat in skill_categories:
        kept: list[str] = []
        for skill in cat["skills"]:
            canonical = canonical_skill_label(skill, skills_map)
            if canonical and canonical.lower() not in {k.lower() for k in kept}:
                kept.append(canonical)
            else:
                dropped.append(skill)
        if kept:
            filtered.append({"name": cat["name"], "skills": kept})
    return filtered, dropped


def check_skills_against_map(
    skill_categories: list[dict[str, Any]],
    skills_map: dict[str, list[str]],
) -> list[str]:
    """Return skills that are not in skills_map."""
    _, dropped = filter_skill_categories(skill_categories, skills_map)
    return dropped


def check_skills_against_background(
    skill_categories: list[dict[str, Any]],
    background_text: str,
) -> list[str]:
    """Return skills not in skills_map (parses Core strengths fallback from body)."""
    from skills_map import parse_core_strengths_markdown

    skills_map = parse_core_strengths_markdown(background_text)
    return check_skills_against_map(skill_categories, skills_map)


def _shorten_output_message(
    char_count: int,
    max_chars: int,
    *,
    include_summary: bool = True,
    include_skills: bool = True,
) -> str:
    if include_summary and include_skills:
        scope = "summary + category names + skill labels"
        hint = "Use tighter summary phrasing and/or fewer, shorter skills per category."
    elif include_skills:
        scope = "category names + skill labels"
        hint = "Use fewer categories and/or shorter skills per category."
    else:
        scope = "summary"
        hint = "Use tighter summary phrasing."
    return (
        f"Your previous JSON used {char_count} characters total ({scope}). "
        f"Shorten it to at most {max_chars} characters combined while keeping the most "
        f"job-relevant facts. {hint} Reply again with ONLY the JSON object, no markdown fences."
    )


def _merge_parsed_with_preserved(
    parsed: dict[str, Any],
    preserved: dict[str, Any],
    *,
    include_summary: bool,
    include_skills: bool,
) -> dict[str, Any]:
    return {
        "summary": parsed["summary"] if include_summary else preserved.get("summary", ""),
        "skill_categories": (
            parsed["skill_categories"] if include_skills else preserved.get("skill_categories", [])
        ),
    }


def _rejected_skills_message(rejected: list[str]) -> str:
    skills_list = ", ".join(rejected)
    return (
        f"The following skills are not in ALLOWED SKILLS and were removed: "
        f"{skills_list}. Replace them with exact labels from ALLOWED SKILLS only. "
        "Reply again with ONLY the JSON object, no markdown fences."
    )


def _build_user_message_with_jd_hints(
    job_description: str,
    *,
    include_skills: bool,
) -> str:
    """User message for generation; optional JD keyword hints for skills packing."""
    from ats import extract_jd_keywords

    jd = job_description.strip()
    if not jd or not include_skills:
        return jd
    keywords = extract_jd_keywords(jd)
    if not keywords:
        return jd
    hinted = ", ".join(keywords[:20])
    suffix = (
        f"\n\nPrioritize ALLOWED SKILLS that match these job keywords when they fit "
        f"on a line: {hinted}."
    )
    return jd + suffix


def _call_with_retries(
    client: OpenAI,
    *,
    model_name: str,
    messages: list[dict[str, str]],
    background_text: str,
    job_description: str,
    skills_map: dict[str, list[str]],
    evidence_text: str = "",
    ai_output_max_chars: int,
    include_summary: bool = True,
    include_skills: bool = True,
    preserved_output: dict[str, Any] | None = None,
    max_skill_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_skills_per_category: int = DEFAULT_MAX_SKILLS_PER_CATEGORY,
    max_chars_per_skill_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Call the model with JSON/budget/skill-grounding retries."""
    preserved = preserved_output or dict(EMPTY_AI_OUTPUT)
    packer_info: dict[str, Any] = {
        "removed_skills": [],
        "deduped_skills": [],
        "added_skills": [],
        "line_utilization": [],
    }
    apply_char_budget = include_summary or skills_subject_to_char_budget(
        include_summary=include_summary,
        include_skills=include_skills,
    )
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
            parsed = parse_response(
                choice.message.content,
                include_summary=include_summary,
                include_skills=include_skills,
            )
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

        dropped: list[str] = []
        if include_skills:
            filtered, dropped = filter_skill_categories(parsed["skill_categories"], skills_map)
            if not filtered:
                last_error = AIResponseError(
                    "All skills were rejected as not in skills_map"
                )
                if attempt < max_attempts - 1:
                    messages = messages + [
                        {"role": "user", "content": _rejected_skills_message(dropped)},
                    ]
                continue
            parsed = {"summary": parsed["summary"], "skill_categories": filtered}

        if include_skills:
            guarded, packer_info = apply_skills_guardrails(
                parsed["skill_categories"],
                skills_map,
                job_description,
                evidence_text=evidence_text,
                max_categories=max_skill_categories,
                max_chars_per_line=max_chars_per_skill_line,
            )
            parsed["skill_categories"] = guarded
            if not parsed["skill_categories"]:
                last_error = AIResponseError(
                    "Skills could not be fitted to the configured layout limits"
                )
                if attempt < max_attempts - 1:
                    messages = messages + [
                        {
                            "role": "user",
                            "content": (
                                "Use fewer categories and shorter skill names so each category "
                                "fits on one line. Reply again with ONLY the JSON object."
                            ),
                        }
                    ]
                continue

        parsed = _merge_parsed_with_preserved(
            parsed,
            preserved,
            include_summary=include_summary,
            include_skills=include_skills,
        )
        char_count = ai_output_char_count(
            parsed["summary"],
            parsed["skill_categories"],
            include_summary=include_summary,
            include_skills=include_skills,
        )
        if apply_char_budget:
            print(f"[ai] output chars: {char_count}/{ai_output_max_chars}")
        else:
            print("[ai] skills-only mode: line-capacity layout (no combined char budget)")

        if dropped and attempt < max_attempts - 1:
            messages = messages + [
                {"role": "user", "content": _rejected_skills_message(dropped)},
            ]
            continue

        if not apply_char_budget or char_count <= ai_output_max_chars:
            return parsed, dropped, packer_info

        last_error = AIResponseError(
            f"Model output exceeded character budget ({char_count} > {ai_output_max_chars})"
        )
        if attempt < max_attempts - 1:
            messages = messages + [
                {
                    "role": "user",
                    "content": _shorten_output_message(
                        char_count,
                        ai_output_max_chars,
                        include_summary=include_summary,
                        include_skills=include_skills,
                    ),
                },
            ]
            continue

        trimmed = enforce_output_budget(
            parsed["summary"],
            parsed["skill_categories"],
            ai_output_max_chars,
            include_summary=include_summary,
            include_skills=include_skills,
        )
        if include_skills:
            trimmed["skill_categories"] = enforce_skills_layout(
                trimmed["skill_categories"],
                max_categories=max_skill_categories,
                max_chars_per_line=max_chars_per_skill_line,
            )
            guarded, packer_info = apply_skills_guardrails(
                trimmed["skill_categories"],
                skills_map,
                job_description,
                max_categories=max_skill_categories,
                max_chars_per_line=max_chars_per_skill_line,
            )
            trimmed["skill_categories"] = guarded
        print(
            "[ai] trimmed output chars: "
            f"{ai_output_char_count(trimmed['summary'], trimmed['skill_categories'], include_summary=include_summary, include_skills=include_skills)}"
            f"/{ai_output_max_chars}"
        )
        return trimmed, dropped, packer_info

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
    include_summary: bool = True,
    include_skills: bool = True,
    max_skill_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_skills_per_category: int = DEFAULT_MAX_SKILLS_PER_CATEGORY,
    max_chars_per_skill_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
    background_data: dict[str, Any] | None = None,
    content_selection: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """
    Call local LLM and return (parsed JSON, dropped skill warnings, packer diagnostics).

    Retries on JSON parse failure, hallucinated skills, and character budget overflow.
    """
    if not include_summary and not include_skills:
        return dict(EMPTY_AI_OUTPUT), [], {
            "removed_skills": [],
            "deduped_skills": [],
            "added_skills": [],
            "line_utilization": [],
        }

    if not job_description.strip():
        raise ValueError("Job description cannot be empty")
    if ai_output_max_chars < 1:
        raise ValueError("ai_output_max_chars must be positive")

    from skills_map import load_skills_map

    background_text = read_background_text(background_path)
    skills_map = load_skills_map(background_path)
    if include_skills and not skills_map:
        raise ValueError(
            "background.md must define skills_map (YAML) or ## Core strengths bullets for skills"
        )

    evidence_text = build_evidence_text(
        background_text,
        background_data,
        content_selection,
    )

    system_prompt = build_system_prompt(
        background_text,
        skills_map=skills_map,
        max_chars=ai_output_max_chars,
        include_summary=include_summary,
        include_skills=include_skills,
        max_skill_categories=max_skill_categories,
        max_skills_per_category=max_skills_per_category,
        max_chars_per_skill_line=max_chars_per_skill_line,
        job_description=job_description.strip(),
        evidence_text=evidence_text,
    )
    client = OpenAI(base_url=endpoint_url, api_key=api_key)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _build_user_message_with_jd_hints(
                job_description,
                include_skills=include_skills,
            ),
        },
    ]

    try:
        return _call_with_retries(
            client,
            model_name=model_name,
            messages=messages,
            background_text=background_text,
            job_description=job_description.strip(),
            skills_map=skills_map,
            evidence_text=evidence_text,
            ai_output_max_chars=ai_output_max_chars,
            include_summary=include_summary,
            include_skills=include_skills,
            max_skill_categories=max_skill_categories,
            max_skills_per_category=max_skills_per_category,
            max_chars_per_skill_line=max_chars_per_skill_line,
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
    include_summary: bool = True,
    include_skills: bool = True,
    max_skill_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_skills_per_category: int = DEFAULT_MAX_SKILLS_PER_CATEGORY,
    max_chars_per_skill_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
    background_data: dict[str, Any] | None = None,
    content_selection: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """
    Revise summary/skills based on user feedback.

    revision_history holds prior chat turns as {"role": "user"|"assistant", "content": "..."}.
    """
    if not include_summary and not include_skills:
        raise ValueError("No AI sections are configured for revision")
    if not job_description.strip():
        raise ValueError("Job description cannot be empty")
    if not feedback.strip():
        raise ValueError("Revision feedback cannot be empty")
    if ai_output_max_chars < 1:
        raise ValueError("ai_output_max_chars must be positive")

    normalized = normalize_ai_output(
        current_output,
        include_summary=include_summary,
        include_skills=include_skills,
    )
    summary = normalized["summary"]
    skill_categories = normalized["skill_categories"]
    if include_summary and not summary:
        raise ValueError("Current summary cannot be empty")
    if include_skills and not skill_categories:
        raise ValueError("Current skills cannot be empty")

    from skills_map import load_skills_map

    background_text = read_background_text(background_path)
    skills_map = load_skills_map(background_path)
    if include_skills and not skills_map:
        raise ValueError(
            "background.md must define skills_map (YAML) or ## Core strengths bullets for skills"
        )

    evidence_text = build_evidence_text(
        background_text,
        background_data,
        content_selection,
    )

    system_prompt = build_revision_system_prompt(
        background_text,
        skills_map=skills_map,
        max_chars=ai_output_max_chars,
        include_summary=include_summary,
        include_skills=include_skills,
        max_skill_categories=max_skill_categories,
        max_skills_per_category=max_skills_per_category,
        max_chars_per_skill_line=max_chars_per_skill_line,
        job_description=job_description.strip(),
        evidence_text=evidence_text,
    )
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
                include_summary=include_summary,
                include_skills=include_skills,
            ),
        }
    )

    try:
        return _call_with_retries(
            client,
            model_name=model_name,
            messages=messages,
            background_text=background_text,
            job_description=job_description.strip(),
            skills_map=skills_map,
            evidence_text=evidence_text,
            ai_output_max_chars=ai_output_max_chars,
            include_summary=include_summary,
            include_skills=include_skills,
            preserved_output=normalized,
            max_skill_categories=max_skill_categories,
            max_skills_per_category=max_skills_per_category,
            max_chars_per_skill_line=max_chars_per_skill_line,
        )
    except AIResponseError as exc:
        if "Cannot connect to local model API" in str(exc):
            raise AIResponseError(
                f"Cannot connect to local model API at {endpoint_url}. "
                "Is your local OpenAI-compatible API running?"
            ) from exc
        raise


def finalize_manual_skills(
    skill_categories: list[dict[str, Any]],
    skills_map: dict[str, list[str]],
    job_description: str = "",
    *,
    evidence_text: str = "",
    max_skill_categories: int = DEFAULT_MAX_SKILL_CATEGORIES,
    max_chars_per_skill_line: int = DEFAULT_MAX_CHARS_PER_SKILL_LINE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply map-grounded guardrails to manually edited skills."""
    return apply_skills_guardrails(
        skill_categories,
        skills_map,
        job_description,
        evidence_text=evidence_text,
        max_categories=max_skill_categories,
        max_chars_per_line=max_chars_per_skill_line,
    )
