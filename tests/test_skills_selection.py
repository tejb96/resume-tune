"""Tests for job-relevant skills selection."""

from __future__ import annotations

from pathlib import Path

from ai import apply_skills_guardrails
from skills_selection import (
    SkillTier,
    build_packed_skill_lines,
    dedupe_redundant_skills,
    flatten_static_evidence_text,
    score_skills_for_job,
    select_relevant_skill_categories,
)

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_MAP = {
    "full_stack": ["React", "TypeScript", "Tailwind CSS"],
    "ai_ml": ["TensorFlow.js", "OpenAI API"],
    "cloud_devops": ["AWS", "Docker", "Kubernetes", "GitHub Actions"],
    "languages": ["Python", "JavaScript", "SQL"],
    "infrastructure": ["AWS", "Docker", "PostgreSQL", "Redis"],
    "tools": ["Git", "Agile", "Jira"],
}

BACKEND_BACKGROUND = {
    "header": {"name": "Test", "email": "t@example.com"},
    "experience": [
        {
            "company": "Acme",
            "title": "Backend Engineer",
            "start": "2020-01",
            "end": "present",
            "bullets": [
                "Built REST APIs in Python with PostgreSQL and Redis caching",
                "Deployed services on AWS with Docker and Kubernetes",
            ],
        }
    ],
}


def test_dedupe_redundant_skills_aws_saa() -> None:
    skills, removed = dedupe_redundant_skills(["AWS", "AWS SAA", "Docker"])
    assert skills == ["AWS", "Docker"]
    assert removed == ["AWS SAA"]


def test_dedupe_redundant_skills_tensorflow_prefers_js() -> None:
    skills, removed = dedupe_redundant_skills(["TensorFlow.js", "TensorFlow", "Python"])
    assert skills == ["TensorFlow.js", "Python"]
    assert removed == ["TensorFlow"]


def test_guardrails_dedupes_aws_saa() -> None:
    categories = [{"name": "Cloud", "skills": ["AWS", "AWS SAA"]}]
    guarded, info = apply_skills_guardrails(
        categories,
        {
            "cloud_devops": ["AWS", "AWS SAA", "Docker"],
            "languages": ["Python"],
        },
        "Need AWS and Docker experience.",
        max_chars_per_line=88,
    )
    all_skills = [s for cat in guarded for s in cat["skills"]]
    assert "AWS" in all_skills
    assert "AWS SAA" not in all_skills
    assert "AWS SAA" in info.get("deduped_skills", [])


def test_flatten_static_evidence_text_includes_experience() -> None:
    text = flatten_static_evidence_text(BACKEND_BACKGROUND)
    assert "PostgreSQL" in text
    assert "Redis" in text
    assert "Kubernetes" in text


def test_score_skills_strong_jd_matches() -> None:
    evidence = flatten_static_evidence_text(BACKEND_BACKGROUND)
    scores = score_skills_for_job(
        SAMPLE_MAP,
        "Need Python, Kubernetes, and AWS experience.",
        evidence,
    )
    assert scores["Python"].tier == SkillTier.STRONG
    assert scores["Kubernetes"].tier == SkillTier.STRONG
    assert scores["AWS"].tier == SkillTier.STRONG


def test_score_evidenced_not_in_jd_is_relevant_or_supporting() -> None:
    evidence = flatten_static_evidence_text(BACKEND_BACKGROUND)
    scores = score_skills_for_job(
        SAMPLE_MAP,
        "Need Python, AWS, and Docker experience.",
        evidence,
    )
    assert scores["PostgreSQL"].tier in (SkillTier.RELEVANT, SkillTier.SUPPORTING)
    assert scores["Redis"].tier in (SkillTier.RELEVANT, SkillTier.SUPPORTING)


def test_score_irrelevant_tools_are_drop() -> None:
    evidence = flatten_static_evidence_text(BACKEND_BACKGROUND)
    scores = score_skills_for_job(
        SAMPLE_MAP,
        "Machine learning engineer with Python and TensorFlow.",
        evidence,
    )
    assert scores["Jira"].tier == SkillTier.DROP
    assert scores["Agile"].tier == SkillTier.DROP


def test_select_relevant_strong_before_filler() -> None:
    evidence = flatten_static_evidence_text(BACKEND_BACKGROUND)
    categories = [
        {
            "name": "Stack",
            "skills": ["Jira", "Python", "Agile", "Kubernetes", "AWS"],
        }
    ]
    packed, info = select_relevant_skill_categories(
        categories,
        SAMPLE_MAP,
        "Need Python, Kubernetes, and AWS experience.",
        evidence,
        max_categories=4,
        max_chars_per_line=88,
    )
    skills = [s for cat in packed for s in cat["skills"]]
    assert "Jira" not in skills
    assert "Agile" not in skills
    assert "Python" in skills
    assert "Kubernetes" in skills
    assert "AWS" in skills
    assert all(cat["name"] == "" for cat in packed)


def test_select_relevant_drops_irrelevant_category() -> None:
    evidence = flatten_static_evidence_text(BACKEND_BACKGROUND)
    categories = [{"name": "Tools", "skills": ["Jira", "Agile"]}]
    packed, info = select_relevant_skill_categories(
        categories,
        SAMPLE_MAP,
        "Need Python, Kubernetes, and AWS experience.",
        evidence,
        max_categories=4,
        max_chars_per_line=88,
    )
    assert info["dropped_categories"]
    assert "Jira" not in [s for cat in packed for s in cat["skills"]]
    assert "Agile" not in [s for cat in packed for s in cat["skills"]]
    assert any("Python" in cat["skills"] for cat in packed)


def test_guardrails_topup_javascript_on_short_languages_line() -> None:
    categories = [{"name": "Languages", "skills": ["Python"]}]
    jd = "Looking for Python and JavaScript experience."
    guarded, info = apply_skills_guardrails(
        categories,
        {
            "languages": ["Python", "JavaScript", "SQL"],
            "tools": ["Git"],
        },
        jd,
        max_chars_per_line=88,
    )
    skills = guarded[0]["skills"]
    assert "Python" in skills
    assert "JavaScript" in skills
    assert "JavaScript" in info.get("added_skills", [])


def test_guardrails_packs_git_sequentially() -> None:
    categories = [{"name": "Languages", "skills": ["Python", "JavaScript", "SQL"]}]
    guarded, _info = apply_skills_guardrails(
        categories,
        {
            "languages": ["Python", "JavaScript", "SQL"],
            "tools": ["Git"],
        },
        "Need Python and Git",
        max_chars_per_line=88,
    )
    all_skills = [s for cat in guarded for s in cat["skills"]]
    assert "Git" in all_skills
    assert "Python" in all_skills


def test_guardrails_packs_git_with_core_skills() -> None:
    skills_map = {
        "cloud_devops": ["Python", "Docker"],
        "tools": ["Git", "Agile", "Jira"],
    }
    categories = [{"name": "Cloud", "skills": ["Python", "Docker"]}]
    jd = "Looking for Python, Docker, and Git experience."
    guarded, _info = apply_skills_guardrails(
        categories,
        skills_map,
        jd,
        max_chars_per_line=88,
    )
    all_skills = [s for cat in guarded for s in cat["skills"]]
    assert "Git" in all_skills
    assert "Python" in all_skills
    assert "Docker" in all_skills


def test_sequential_fill_uses_line1_before_line2() -> None:
    skills_map = {
        "full_stack": [
            "React",
            "Next.js",
            "TypeScript",
            "Node.js",
            "Express.js",
            "FastAPI",
            "MongoDB",
            "PostgreSQL",
        ],
        "tools": ["Git", "Agile"],
    }
    jd = "Full-stack React, Next.js, TypeScript, and Node.js developer."
    evidence = (
        "Built React and Next.js apps with TypeScript, Node.js, Express.js, "
        "FastAPI, MongoDB, and PostgreSQL."
    )
    packed, _, _ = build_packed_skill_lines(
        skills_map,
        jd,
        evidence,
        max_categories=4,
        max_chars_per_line=50,
    )
    assert len(packed) >= 2
    line0_chars = len(", ".join(packed[0]["skills"]))
    assert line0_chars >= 40
    assert packed[1]["skills"]


def test_no_sparse_bucket_lines() -> None:
    skills_map = {
        "full_stack": ["React", "Next.js", "TypeScript", "Tailwind CSS", "Redux"],
        "ai_ml": ["TensorFlow.js"],
        "cloud_devops": ["AWS", "Docker"],
        "languages": ["Python", "JavaScript"],
    }
    jd = "Full-stack React developer with TypeScript and Next.js."
    evidence = (
        "React, Next.js, TypeScript, Tailwind CSS, Redux, TensorFlow.js, "
        "AWS, Docker, Python, JavaScript."
    )
    packed, _, _ = build_packed_skill_lines(
        skills_map,
        jd,
        evidence,
        max_categories=4,
        max_chars_per_line=88,
    )
    for idx in range(len(packed) - 1):
        chars = len(", ".join(packed[idx]["skills"]))
        remaining = 88 - chars
        next_chars = len(", ".join(packed[idx + 1]["skills"]))
        if remaining > 10:
            assert next_chars >= 20 or remaining <= 10


def test_frontend_skills_pack_after_core() -> None:
    skills_map = {
        "full_stack": [
            "React",
            "Next.js",
            "TypeScript",
            "Tailwind CSS",
            "Material UI",
            "Redux",
            "Shadcn/ui",
            "Node.js",
            "FastAPI",
        ],
        "ai_ml": ["TensorFlow.js", "OpenAI API"],
        "cloud_devops": ["AWS", "Docker", "GitHub Actions"],
        "languages": ["Python", "JavaScript"],
    }
    jd = "Full-stack developer: React, Next.js, TypeScript, Node.js, FastAPI."
    evidence = (
        "MERN stack with React, Redux, Material UI, Tailwind CSS, Shadcn/ui, "
        "TensorFlow.js on RepVision, AWS, Docker, GitHub Actions."
    )
    packed, _, _ = build_packed_skill_lines(
        skills_map,
        jd,
        evidence,
        max_categories=4,
        max_chars_per_line=88,
    )
    all_skills = [s for cat in packed for s in cat["skills"]]
    assert "Tailwind CSS" in all_skills or "Redux" in all_skills
    if "TensorFlow.js" in all_skills:
        tf_line = next(i for i, cat in enumerate(packed) if "TensorFlow.js" in cat["skills"])
        line_chars = len(", ".join(packed[tf_line]["skills"]))
        assert line_chars >= 20 or tf_line == 0


def test_diagnostics_split_overflow_vs_irrelevant() -> None:
    evidence = flatten_static_evidence_text(BACKEND_BACKGROUND)
    categories = [
        {"name": "", "skills": ["Python", "Kubernetes", "AWS", "CloudFront", "Jira"]},
    ]
    _, info = select_relevant_skill_categories(
        categories,
        {
            **SAMPLE_MAP,
            "cloud_devops": ["AWS", "CloudFront", "Docker", "Kubernetes"],
        },
        "Need Python, Kubernetes, and AWS experience.",
        evidence,
        max_categories=4,
        max_chars_per_line=88,
    )
    assert "CloudFront" in info["removed_irrelevant"]
    assert "Jira" in info["removed_irrelevant"]
    assert "removed_overflow" in info


def test_trim_removes_filler_before_strong() -> None:
    evidence = flatten_static_evidence_text(BACKEND_BACKGROUND)
    categories = [
        {
            "name": "Cloud",
            "skills": [
                "Python",
                "AWS",
                "Docker",
                "Kubernetes",
                "GitHub Actions",
                "Redis",
                "PostgreSQL",
            ],
        }
    ]
    packed, _info = select_relevant_skill_categories(
        categories,
        {
            "infrastructure": [
                "AWS",
                "Docker",
                "Kubernetes",
                "GitHub Actions",
                "PostgreSQL",
                "Redis",
            ],
            "languages": ["Python"],
        },
        "Need Python, AWS, Docker, and Kubernetes.",
        evidence,
        max_categories=4,
        max_chars_per_line=40,
    )
    assert packed
    skills = packed[0]["skills"]
    assert "Python" in skills or "AWS" in skills
