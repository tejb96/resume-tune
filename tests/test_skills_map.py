"""Tests for structured skills_map loading and guardrails."""

from __future__ import annotations

from pathlib import Path

from resume_tune.llm.ai import (
    apply_skills_guardrails,
    dedupe_skill_redundancies,
    filter_skill_categories,
)
from resume_tune.render.resume import build_resume, flatten_resume_text, load_background
from resume_tune.skills.skills_map import load_skills_map, parse_core_strengths_markdown

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_MAP = {
    "full_stack": ["React", "TypeScript", "Tailwind CSS"],
    "ai_ml": ["TensorFlow.js", "OpenAI API"],
    "cloud_devops": ["AWS", "Docker", "GitHub Actions"],
    "languages": ["Python", "JavaScript", "SQL"],
    "tools": ["Git", "Agile", "Jira"],
}


def test_load_skills_map_from_background_example() -> None:
    skills_map = load_skills_map(ROOT / "background.example.md")
    assert "languages" in skills_map
    assert "Python" in skills_map["languages"]


def test_load_skills_map_user_background_has_no_azure() -> None:
    path = ROOT / "background.md"
    if not path.exists():
        return
    skills_map = load_skills_map(path)
    all_skills = [s for skills in skills_map.values() for s in skills]
    assert "Azure" not in all_skills


def test_parse_core_strengths_markdown() -> None:
    body = """
## Core strengths

- **Full-stack delivery**: React, Next.js, TypeScript
- **Other tools**: Git, Agile, Jira

## Tailoring guidance

- do not invent Azure
"""
    parsed = parse_core_strengths_markdown(body)
    assert "React" in parsed["full_stack"]
    assert "Git" in parsed["tools"]
    assert "Azure" not in {s for vals in parsed.values() for s in vals}


def test_filter_skill_categories_uses_map() -> None:
    categories = [{"name": "Stack", "skills": ["React", "Azure", "FakeTool"]}]
    filtered, dropped = filter_skill_categories(categories, SAMPLE_MAP)
    assert filtered[0]["skills"] == ["React"]
    assert "Azure" in dropped
    assert "FakeTool" in dropped


def test_dedupe_tensorflow_and_git() -> None:
    categories = [
        {"name": "AI", "skills": ["TensorFlow.js", "TensorFlow", "OpenAI API"]},
        {"name": "Tools", "skills": ["Git", "GitHub"]},
    ]
    deduped_cats, removed = dedupe_skill_redundancies(categories)
    assert "TensorFlow" not in deduped_cats[0]["skills"]
    assert "GitHub" not in deduped_cats[1]["skills"]
    assert "TensorFlow" in removed
    assert "GitHub" in removed


def test_dedupe_aws_saa_when_aws_present() -> None:
    categories = [{"name": "Cloud", "skills": ["AWS", "AWS SAA", "Docker"]}]
    deduped_cats, removed = dedupe_skill_redundancies(categories)
    assert deduped_cats[0]["skills"] == ["AWS", "Docker"]
    assert "AWS SAA" in removed


def test_apply_skills_guardrails_same_bucket_topup() -> None:
    categories = [{"name": "Languages", "skills": ["Python"]}]
    jd = "Looking for Python and JavaScript experience."
    guarded, info = apply_skills_guardrails(
        categories,
        SAMPLE_MAP,
        jd,
        max_chars_per_line=88,
    )
    skills = guarded[0]["skills"]
    assert "Python" in skills
    assert "JavaScript" in skills
    assert "JavaScript" in info.get("added_skills", [])


def test_apply_skills_guardrails_packs_git_sequentially() -> None:
    categories = [{"name": "Languages", "skills": ["Python", "JavaScript", "SQL"]}]
    guarded, _info = apply_skills_guardrails(
        categories,
        SAMPLE_MAP,
        "Need Python and Git",
        max_chars_per_line=88,
    )
    all_skills = [s for cat in guarded for s in cat["skills"]]
    assert "Git" in all_skills
    assert "Python" in all_skills


def test_unnamed_category_renders_without_label() -> None:
    data = load_background(ROOT / "background.example.md")
    ai_output = {
        "summary": "",
        "skill_categories": [{"name": "", "skills": ["Python", "Go"]}],
    }
    text = flatten_resume_text(data, ai_output, sections=["skills"])
    assert "▪ Python, Go" in text
    assert ": Python" not in text

    docx = build_resume(data, ai_output, sections=["skills"])
    assert docx
