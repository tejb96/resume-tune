"""Structured skills_map from background.md for LLM selection and guardrails."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import frontmatter

_SECTION_KEY_ALIASES: dict[str, str] = {
    "full-stack delivery": "full_stack",
    "full stack delivery": "full_stack",
    "full stack": "full_stack",
    "ai / ml integration": "ai_ml",
    "ai/ml integration": "ai_ml",
    "ai ml integration": "ai_ml",
    "cloud & devops": "cloud_devops",
    "cloud and devops": "cloud_devops",
    "languages": "languages",
    "other tools": "tools",
}

_CORE_STRENGTHS_HEADER = re.compile(r"^##\s+Core\s+strengths\s*$", re.IGNORECASE | re.MULTILINE)
_BULLET_SECTION = re.compile(
    r"^-\s+\*\*(.+?)\*\*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_section_key(label: str) -> str:
    cleaned = label.strip().lower()
    if cleaned in _SECTION_KEY_ALIASES:
        return _SECTION_KEY_ALIASES[cleaned]
    slug = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    return slug or "other"


def _split_skill_tokens(text: str) -> list[str]:
    parts = re.split(r",|\s{2,}|\n", text)
    skills: list[str] = []
    seen: set[str] = set()
    for part in parts:
        skill = part.strip().strip(".")
        if not skill:
            continue
        key = skill.lower()
        if key not in seen:
            seen.add(key)
            skills.append(skill)
    return skills


def parse_core_strengths_markdown(body: str) -> dict[str, list[str]]:
    """Parse ## Core strengths bullets into a skills_map."""
    match = _CORE_STRENGTHS_HEADER.search(body)
    if not match:
        return {}

    section_text = body[match.end() :]
    next_header = re.search(r"^##\s+", section_text, re.MULTILINE)
    if next_header:
        section_text = section_text[: next_header.start()]

    result: dict[str, list[str]] = {}
    for section_match in _BULLET_SECTION.finditer(section_text):
        label = section_match.group(1)
        skills_blob = section_match.group(2)
        key = _normalize_section_key(label)
        skills = _split_skill_tokens(skills_blob)
        if skills:
            existing = result.setdefault(key, [])
            for skill in skills:
                if skill.lower() not in {s.lower() for s in existing}:
                    existing.append(skill)
    return result


def validate_skills_map(skills_map: Any) -> dict[str, list[str]]:
    """Validate and normalize skills_map from YAML frontmatter."""
    if skills_map is None:
        return {}
    if not isinstance(skills_map, dict):
        raise ValueError("background.md: 'skills_map' must be a mapping of bucket -> skill list")

    validated: dict[str, list[str]] = {}
    for bucket, skills in skills_map.items():
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("background.md: skills_map keys must be non-empty strings")
        if not isinstance(skills, list) or not skills:
            raise ValueError(f"background.md: skills_map.{bucket} must be a non-empty list")
        cleaned: list[str] = []
        for i, skill in enumerate(skills):
            if not isinstance(skill, str) or not skill.strip():
                raise ValueError(
                    f"background.md: skills_map.{bucket}[{i}] must be a non-empty string"
                )
            if skill.strip().lower() not in {s.lower() for s in cleaned}:
                cleaned.append(skill.strip())
        validated[bucket.strip()] = cleaned
    return validated


def load_skills_map(background_path: Path) -> dict[str, list[str]]:
    """Load skills_map from YAML frontmatter, falling back to Core strengths markdown."""
    if not background_path.exists():
        raise FileNotFoundError(f"Background file not found: {background_path}")

    post = frontmatter.load(background_path)
    raw_map = post.metadata.get("skills_map")
    if raw_map:
        return validate_skills_map(raw_map)

    parsed = parse_core_strengths_markdown(post.content)
    if parsed:
        return parsed

    return {}


def flatten_skills_map(skills_map: dict[str, list[str]]) -> dict[str, str]:
    """Map lowercase skill label -> canonical label from skills_map."""
    flat: dict[str, str] = {}
    for skills in skills_map.values():
        for skill in skills:
            flat[skill.lower()] = skill
    return flat


def skill_in_skills_map(skill: str, skills_map: dict[str, list[str]]) -> bool:
    """Return True when skill matches a skills_map entry (case-insensitive)."""
    return skill.strip().lower() in flatten_skills_map(skills_map)


def canonical_skill_label(skill: str, skills_map: dict[str, list[str]]) -> str | None:
    """Return canonical map label for skill, or None if not in map."""
    return flatten_skills_map(skills_map).get(skill.strip().lower())


def buckets_for_skill(skill: str, skills_map: dict[str, list[str]]) -> list[str]:
    """Return bucket keys that contain this skill."""
    needle = skill.strip().lower()
    return [bucket for bucket, skills in skills_map.items() if needle in {s.lower() for s in skills}]


def format_skills_map_for_prompt(skills_map: dict[str, list[str]]) -> str:
    """Format skills_map for the LLM system prompt."""
    if not skills_map:
        return "ALLOWED SKILLS: (none configured — add skills_map to background.md)"

    lines = ["ALLOWED SKILLS (pick only from these lists — exact spelling):"]
    for bucket, skills in skills_map.items():
        lines.append(f"- {bucket}: {', '.join(skills)}")
    return "\n".join(lines)
