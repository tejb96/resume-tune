"""Load settings from .env / environment, with config.toml fallbacks."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from resume_tune.llm.ai import (
    DEFAULT_AI_OUTPUT_MAX_CHARS,
    DEFAULT_MAX_CHARS_PER_SKILL_LINE,
    DEFAULT_MAX_SKILL_CATEGORIES,
    DEFAULT_MAX_SKILLS_PER_CATEGORY,
)
from resume_tune.llm.selection import (
    DEFAULT_MIN_PROJECT_BULLETS,
    DEFAULT_MIN_PROJECT_ENTRIES,
)
from resume_tune.render.resume import resolve_resume_sections

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config.toml"
CONFIG_LOCAL_PATH = ROOT / "config.local.toml"
ENV_PATH = ROOT / ".env"

ENV_SETTING_KEYS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "AI_OUTPUT_MAX_CHARS",
)

LOCAL_CONFIG_KEYS = (
    "output_dir",
    "background_file",
    "tracker_file",
    "max_resume_pages",
    "auto_fill_page_budget",
    "overflow_warning_min_composite",
    "enable_job_aware_selection",
    "enable_ats_compat",
    "min_project_entries",
    "min_project_bullets",
    "max_skill_categories",
    "max_skills_per_category",
    "max_chars_per_skill_line",
    "max_completion_tokens",
    "max_certifications",
    "resume_sections",
)


def _load_dotenv() -> None:
    env_path = ENV_PATH
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_merged_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)
    if CONFIG_LOCAL_PATH.is_file():
        with CONFIG_LOCAL_PATH.open("rb") as f:
            local = tomllib.load(f)
        config = _deep_merge(config, local)
    return config


def _resolve_ai_output_max_chars(config: dict) -> int:
    raw = os.getenv("AI_OUTPUT_MAX_CHARS")
    if raw is not None and raw.strip():
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"AI_OUTPUT_MAX_CHARS must be an integer, got {raw!r}") from exc
    else:
        value = int(config.get("ai_output_max_chars", DEFAULT_AI_OUTPUT_MAX_CHARS))
    if value < 1:
        raise ValueError(f"AI_OUTPUT_MAX_CHARS must be positive, got {value}")
    return value


def load_settings() -> dict:
    _load_dotenv()
    config = _load_merged_config()

    endpoint_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or config.get("endpoint_url", "")
    ).strip()
    api_key = (os.getenv("OPENAI_API_KEY") or config.get("api_key") or "ollama").strip()
    model_name = (os.getenv("OPENAI_MODEL") or config.get("model_name") or "").strip()

    return {
        **config,
        "endpoint_url": endpoint_url,
        "api_key": api_key,
        "model_name": model_name,
        "ai_output_max_chars": _resolve_ai_output_max_chars(config),
        "resume_sections": resolve_resume_sections(config.get("resume_sections")),
        "max_resume_pages": int(config.get("max_resume_pages", 2)),
        "enable_job_aware_selection": bool(config.get("enable_job_aware_selection", False)),
        "enable_ats_compat": bool(config.get("enable_ats_compat", True)),
        "auto_fill_page_budget": bool(config.get("auto_fill_page_budget", True)),
        "overflow_warning_min_composite": float(
            config.get("overflow_warning_min_composite", 75.0)
        ),
        "min_project_entries": int(config.get("min_project_entries", DEFAULT_MIN_PROJECT_ENTRIES)),
        "min_project_bullets": int(config.get("min_project_bullets", DEFAULT_MIN_PROJECT_BULLETS)),
        "max_skill_categories": int(
            config.get("max_skill_categories", DEFAULT_MAX_SKILL_CATEGORIES)
        ),
        "max_skills_per_category": int(
            config.get("max_skills_per_category", DEFAULT_MAX_SKILLS_PER_CATEGORY)
        ),
        "max_chars_per_skill_line": int(
            config.get("max_chars_per_skill_line", DEFAULT_MAX_CHARS_PER_SKILL_LINE)
        ),
        "max_certifications": int(config.get("max_certifications", 1)),
        "max_completion_tokens": _optional_int(config.get("max_completion_tokens")),
        "tracker_file": str(config.get("tracker_file", "./Applications/applications.xlsx")),
    }


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse KEY=VALUE pairs from .env, preserving only recognized keys' values."""
    env_path = path or ENV_PATH
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        if key:
            values[key] = raw_value.strip()
    return values


def save_env_settings(
    *,
    endpoint_url: str,
    api_key: str,
    model_name: str,
    ai_output_max_chars: int,
    path: Path | None = None,
) -> None:
    """Update known LLM keys in .env without removing unrelated lines or comments."""
    env_path = path or ENV_PATH
    updates = {
        "OPENAI_BASE_URL": endpoint_url.strip(),
        "OPENAI_API_KEY": api_key.strip(),
        "OPENAI_MODEL": model_name.strip(),
        "AI_OUTPUT_MAX_CHARS": str(ai_output_max_chars),
    }

    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = [
            "# OpenAI-compatible local LLM (Lemonade, Ollama, etc.)",
            "# Copy from .env.example and edit in Settings or here.",
        ]

    seen: set[str] = set()
    new_lines: list[str] = []
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")

    for line in lines:
        match = pattern.match(line)
        if match and match.group(1) in updates:
            key = match.group(1)
            new_lines.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            new_lines.append(line)

    for key in ENV_SETTING_KEYS:
        if key not in seen:
            new_lines.append(f"{key}={updates[key]}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def save_local_config(updates: dict[str, Any], path: Path | None = None) -> None:
    """Write user overrides to config.local.toml."""
    local_path = path or CONFIG_LOCAL_PATH
    payload = {key: updates[key] for key in LOCAL_CONFIG_KEYS if key in updates}
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as f:
        tomli_w.dump(payload, f)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(text)
