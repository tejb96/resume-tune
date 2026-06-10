"""Deterministic job-relevant skills selection from skills_map, JD, and resume evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from selection import apply_content_selection
from skills_map import buckets_for_skill, flatten_skills_map


class SkillTier(IntEnum):
    STRONG = 0
    RELEVANT = 1
    SUPPORTING = 2
    FILLER = 3
    DROP = 4


@dataclass(frozen=True)
class SkillScore:
    skill: str
    tier: SkillTier
    buckets: tuple[str, ...]


FILLER_LINE_UTILIZATION_THRESHOLD = 0.70

# Four unnamed lines grouped by bucket similarity (order = pack priority for full-stack roles).
LINE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"full_stack", "languages"}),
    frozenset({"ai_ml"}),
    frozenset({"cloud_devops"}),
    frozenset({"tools"}),
)

FULL_STACK_JD_SIGNALS: tuple[str, ...] = (
    "full-stack",
    "full stack",
    "fullstack",
    "web application",
    "frontend",
    "front-end",
    "react",
    "javascript/typescript",
    "javascript / typescript",
)

# Buckets whose STRONG skills may share a rendered line (e.g. Python on a Full Stack line).
BUCKET_AFFINITY: dict[str, frozenset[str]] = {
    "full_stack": frozenset({"full_stack", "languages"}),
    "languages": frozenset({"full_stack", "languages"}),
    "ai_ml": frozenset({"ai_ml", "full_stack"}),
    "cloud_devops": frozenset({"cloud_devops"}),
    "tools": frozenset({"tools"}),
}


def _affinity_buckets(bucket: str) -> frozenset[str]:
    return BUCKET_AFFINITY.get(bucket, frozenset({bucket}))


def _skill_buckets_affinity(skill: str, skills_map: dict[str, list[str]]) -> set[str]:
    affinity: set[str] = set()
    for bucket in buckets_for_skill(skill, skills_map):
        affinity.update(_affinity_buckets(bucket))
    return affinity


def _is_full_stack_jd(job_description: str) -> bool:
    text = job_description.lower()
    return any(signal in text for signal in FULL_STACK_JD_SIGNALS)


def _bucket_role_rank(bucket: str, job_description: str) -> int:
    """Lower rank = higher priority when the JD reads full-stack / web."""
    if _is_full_stack_jd(job_description):
        return {
            "full_stack": 0,
            "languages": 1,
            "tools": 2,
            "cloud_devops": 3,
            "ai_ml": 4,
        }.get(bucket, 10)
    return {
        "ai_ml": 0,
        "full_stack": 1,
        "languages": 2,
        "cloud_devops": 3,
        "tools": 4,
    }.get(bucket, 10)


def _primary_bucket(skill: str, skills_map: dict[str, list[str]]) -> str:
    buckets = buckets_for_skill(skill, skills_map)
    return buckets[0] if buckets else ""


def _line_group_index(skill: str, skills_map: dict[str, list[str]]) -> int:
    buckets = set(buckets_for_skill(skill, skills_map))
    for idx, group in enumerate(LINE_GROUPS):
        if buckets & group:
            return idx
    return len(LINE_GROUPS) - 1


def _web_stack_penalty(skill: str, job_description: str) -> int:
    """Deprioritize legacy/CMS skills on full-stack web roles."""
    if not _is_full_stack_jd(job_description):
        return 0
    if skill.lower() in {"php", "wordpress", "hubspot", "jira", "agile"}:
        return 50
    return 0


def _eligible_for_pack(
    skill: str,
    score: SkillScore,
    job_description: str,
    skills_map: dict[str, list[str]],
) -> bool:
    if score.tier == SkillTier.DROP:
        return False
    primary = _primary_bucket(skill, skills_map)
    if primary == "ai_ml":
        if score.tier <= SkillTier.RELEVANT:
            return True
        return _skill_in_job_text(skill, job_description)
    if score.tier <= SkillTier.RELEVANT:
        return True
    if score.tier == SkillTier.SUPPORTING:
        if _skill_in_job_text(skill, job_description):
            return True
        if _is_full_stack_jd(job_description) and primary in {"full_stack", "languages"}:
            return True
        return False
    if score.tier == SkillTier.FILLER:
        return _skill_in_job_text(skill, job_description)
    return False


def _skill_pack_key(
    skill: str,
    scores: dict[str, SkillScore],
    skills_map: dict[str, list[str]],
    jd_keywords: list[str],
    job_description: str,
) -> tuple[int, int, int, int, str]:
    primary = _primary_bucket(skill, skills_map)
    return (
        scores[skill].tier,
        _jd_priority(skill, jd_keywords, job_description),
        _web_stack_penalty(skill, job_description),
        _bucket_role_rank(primary, job_description),
        skill.lower(),
    )


def _dedupe_skills_for_pack(
    skills: list[str],
    scores: dict[str, SkillScore],
    skills_map: dict[str, list[str]],
    jd_keywords: list[str],
    job_description: str,
) -> list[str]:
    """Drop redundant pairs; keep the higher-priority label."""
    lowers = {s.lower() for s in skills}
    drop: set[str] = set()
    if "tailwind css" in lowers and "tailwind" in lowers:
        prefer_css = "tailwind css" in job_description.lower() or any(
            k.lower() == "tailwind css" for k in jd_keywords
        )
        drop.add("tailwind" if prefer_css else "tailwind css")
    if "tensorflow.js" in lowers and "tensorflow" in lowers:
        drop.add("tensorflow")
    if "git" in lowers and "github" in lowers:
        drop.add("github")
    return [s for s in skills if s.lower() not in drop]


def build_packed_skill_lines(
    skills_map: dict[str, list[str]],
    job_description: str,
    evidence_text: str,
    *,
    max_categories: int,
    max_chars_per_line: int,
) -> tuple[list[dict[str, Any]], dict[str, SkillScore], list[str]]:
    """Build up to four unnamed lines grouped by bucket similarity."""
    from ats import extract_jd_keywords

    scores = score_skills_for_job(skills_map, job_description, evidence_text)
    jd_keywords = extract_jd_keywords(job_description) if job_description.strip() else []

    eligible = _dedupe_skills_for_pack(
        [
            skill
            for skill, score in scores.items()
            if _eligible_for_pack(skill, score, job_description, skills_map)
        ],
        scores,
        skills_map,
        jd_keywords,
        job_description,
    )
    eligible.sort(
        key=lambda s: _skill_pack_key(s, scores, skills_map, jd_keywords, job_description)
    )

    line_skills: list[list[str]] = [[] for _ in range(min(max_categories, len(LINE_GROUPS)))]

    for skill in eligible:
        group_idx = _line_group_index(skill, skills_map)
        if group_idx >= len(line_skills):
            continue
        name = ""
        current = line_skills[group_idx]
        skill_key = _skill_pack_key(skill, scores, skills_map, jd_keywords, job_description)

        if _skill_category_line_length(name, current + [skill]) <= max_chars_per_line:
            current.append(skill)
            continue

        while True:
            droppable = [
                i
                for i, listed in enumerate(current)
                if _skill_pack_key(listed, scores, skills_map, jd_keywords, job_description)
                > skill_key
            ]
            if not droppable:
                break
            drop_idx = max(
                droppable,
                key=lambda i: _skill_pack_key(
                    current[i], scores, skills_map, jd_keywords, job_description
                ),
            )
            current.pop(drop_idx)
            if _skill_category_line_length(name, current + [skill]) <= max_chars_per_line:
                current.append(skill)
                break

    packed: list[dict[str, Any]] = []
    for skills in line_skills:
        if not skills:
            continue
        ordered = sorted(
            skills,
            key=lambda s: _skill_pack_key(s, scores, skills_map, jd_keywords, job_description),
        )
        packed.append({"name": "", "skills": ordered})

    return packed[:max_categories], scores, jd_keywords


def flatten_static_evidence_text(
    background_data: dict[str, Any],
    *,
    content_selection: dict[str, Any] | None = None,
    narrative_text: str = "",
) -> str:
    """Build text corpus from static resume YAML and optional narrative."""
    render_data = (
        apply_content_selection(background_data, content_selection)
        if content_selection is not None
        else background_data
    )
    parts: list[str] = []

    for job in render_data.get("experience", []):
        parts.append(str(job.get("title", "")))
        parts.append(str(job.get("company", "")))
        for bullet in job.get("bullets", []):
            parts.append(str(bullet))

    for project in render_data.get("projects", []):
        parts.append(str(project.get("name", "")))
        parts.append(str(project.get("tech", "")))
        for bullet in project.get("bullets", []):
            parts.append(str(bullet))

    for cert in render_data.get("certifications", []):
        parts.append(str(cert.get("name", "")))
        parts.append(str(cert.get("issuer", "")))

    if narrative_text.strip():
        parts.append(narrative_text.strip())

    return "\n".join(part for part in parts if part)


def extract_narrative_text(background_text: str) -> str:
    """Return markdown body after YAML frontmatter."""
    import frontmatter

    post = frontmatter.loads(background_text)
    return post.content or ""


def evidenced_skills_in_map(
    skills_map: dict[str, list[str]],
    evidence_text: str,
    jd_keywords: list[str] | None = None,
) -> set[str]:
    """Map free-text evidence hits to canonical skills_map labels."""
    from ats import extract_evidenced_skills

    if not evidence_text.strip():
        return set()

    flat = flatten_skills_map(skills_map)
    evidenced: set[str] = set()
    for term in extract_evidenced_skills(evidence_text, jd_keywords=jd_keywords):
        canonical = flat.get(term.lower())
        if canonical:
            evidenced.add(canonical)
    return evidenced


def _skill_in_job_text(skill: str, job_description: str) -> bool:
    from ats import _keyword_in_text

    if not job_description.strip():
        return False
    return _keyword_in_text(skill, job_description)


def _jd_priority(skill: str, jd_keywords: list[str], job_description: str) -> int:
    """Lower rank = higher priority in the job description."""
    skill_lower = skill.lower()
    for idx, keyword in enumerate(jd_keywords):
        if skill_lower == keyword.lower():
            return idx
    if _skill_in_job_text(skill, job_description):
        return len(jd_keywords)
    return len(jd_keywords) + 100


def _skill_matches_jd(skill: str, jd_keywords: list[str], job_description: str) -> bool:
    from ats import _keyword_in_text

    if _skill_in_job_text(skill, job_description):
        return True
    skill_lower = skill.lower()
    if skill_lower == "git" and any(k.lower() == "github" for k in jd_keywords):
        return True
    for keyword in jd_keywords:
        if skill_lower == keyword.lower():
            return True
        if _keyword_in_text(keyword, skill):
            return True
        if _keyword_in_text(skill, keyword):
            return True
    return False


def score_skills_for_job(
    skills_map: dict[str, list[str]],
    job_description: str,
    evidence_text: str,
) -> dict[str, SkillScore]:
    """Score every skills_map skill by job and resume relevance."""
    from ats import extract_jd_keywords

    jd_keywords = extract_jd_keywords(job_description) if job_description.strip() else []
    evidenced = evidenced_skills_in_map(skills_map, evidence_text, jd_keywords=jd_keywords)

    all_skills: list[str] = []
    seen: set[str] = set()
    for skills in skills_map.values():
        for skill in skills:
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                all_skills.append(skill)

    tiers: dict[str, SkillTier] = {}
    for skill in all_skills:
        if _skill_matches_jd(skill, jd_keywords, job_description):
            tiers[skill] = SkillTier.STRONG

    strong_buckets = {
        bucket
        for skill, tier in tiers.items()
        if tier == SkillTier.STRONG
        for bucket in buckets_for_skill(skill, skills_map)
    }

    for skill in all_skills:
        if skill in tiers:
            continue
        skill_buckets = set(buckets_for_skill(skill, skills_map))
        if (
            skill in evidenced
            and skill_buckets & strong_buckets
            and _skill_in_job_text(skill, job_description)
        ):
            tiers[skill] = SkillTier.RELEVANT

    relevant_buckets = {
        bucket
        for skill, tier in tiers.items()
        if tier in (SkillTier.STRONG, SkillTier.RELEVANT)
        for bucket in buckets_for_skill(skill, skills_map)
    }

    for skill in all_skills:
        if skill in tiers:
            continue
        skill_buckets = set(buckets_for_skill(skill, skills_map))
        if skill in evidenced and skill_buckets & relevant_buckets:
            tiers[skill] = SkillTier.SUPPORTING

    active_buckets = {
        bucket
        for skill, tier in tiers.items()
        if tier in (SkillTier.STRONG, SkillTier.RELEVANT, SkillTier.SUPPORTING)
        for bucket in buckets_for_skill(skill, skills_map)
    }

    for skill in all_skills:
        if skill in tiers:
            continue
        skill_buckets = set(buckets_for_skill(skill, skills_map))
        if skill_buckets & active_buckets:
            tiers[skill] = SkillTier.FILLER

    for skill in all_skills:
        if skill not in tiers:
            tiers[skill] = SkillTier.DROP

    return {
        skill: SkillScore(
            skill=skill,
            tier=tiers[skill],
            buckets=tuple(buckets_for_skill(skill, skills_map)),
        )
        for skill in all_skills
    }


def eligible_buckets(scores: dict[str, SkillScore]) -> set[str]:
    """Buckets with at least one strong, relevant, or supporting skill."""
    eligible: set[str] = set()
    for score in scores.values():
        if score.tier <= SkillTier.SUPPORTING:
            eligible.update(score.buckets)
    return eligible


def format_skill_hints_for_prompt(
    scores: dict[str, SkillScore],
    skills_map: dict[str, list[str]],
    *,
    max_per_bucket: int = 6,
) -> str:
    """Build JOB-MATCHING / EVIDENCED hint block for the LLM system prompt."""
    job_matching: list[str] = []
    evidenced_relevant: list[str] = []

    for bucket, skills in skills_map.items():
        bucket_strong = [
            s for s in skills if scores.get(s) and scores[s].tier == SkillTier.STRONG
        ]
        bucket_relevant = [
            s
            for s in skills
            if scores.get(s) and scores[s].tier in (SkillTier.RELEVANT, SkillTier.SUPPORTING)
        ]
        job_matching.extend(bucket_strong[:max_per_bucket])
        evidenced_relevant.extend(bucket_relevant[:max_per_bucket])

    job_matching = sorted(set(job_matching), key=str.lower)
    evidenced_relevant = sorted(
        {s for s in evidenced_relevant if s not in job_matching},
        key=str.lower,
    )

    lines: list[str] = []
    if job_matching:
        lines.append(f"JOB-MATCHING SKILLS (prioritize): {', '.join(job_matching)}")
    if evidenced_relevant:
        lines.append(
            f"EVIDENCED IN RESUME (use when relevant): {', '.join(evidenced_relevant)}"
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _skill_category_line_length(name: str, skills: list[str]) -> int:
    label = name.strip()
    if not skills:
        return len(label) + (2 if label else 0)
    joined = ", ".join(skills)
    if not label:
        return len(joined)
    return len(label) + 2 + len(joined)


def _buckets_on_line(skills: list[str], skills_map: dict[str, list[str]]) -> set[str]:
    buckets: set[str] = set()
    for skill in skills:
        buckets.update(buckets_for_skill(skill, skills_map))
    return buckets


def _line_utilization(name: str, skills: list[str], max_chars: int) -> float:
    if max_chars <= 0:
        return 1.0
    return _skill_category_line_length(name, skills) / max_chars


def _sort_skills_by_tier(
    skills: list[str],
    scores: dict[str, SkillScore],
    *,
    jd_keywords: list[str] | None = None,
    job_description: str = "",
) -> list[str]:
    return sorted(
        skills,
        key=lambda s: (
            scores.get(s, SkillScore(s, SkillTier.DROP, ())).tier,
            _jd_priority(s, jd_keywords or [], job_description),
            s.lower(),
        ),
    )


def enforce_skills_layout_by_tier(
    skill_categories: list[dict[str, Any]],
    scores: dict[str, SkillScore],
    *,
    max_categories: int,
    max_chars_per_line: int,
    jd_keywords: list[str] | None = None,
    job_description: str = "",
) -> list[dict[str, Any]]:
    """Cap categories and trim lines, removing lowest-tier skills first."""
    result: list[dict[str, Any]] = []
    for cat in skill_categories[:max_categories]:
        name = cat["name"].strip()
        skills = _sort_skills_by_tier(
            [s.strip() for s in cat["skills"] if s.strip()],
            scores,
            jd_keywords=jd_keywords,
            job_description=job_description,
        )
        while skills and _skill_category_line_length(name, skills) > max_chars_per_line:
            drop_idx = max(
                range(len(skills)),
                key=lambda i: (
                    scores.get(skills[i], SkillScore(skills[i], SkillTier.DROP, ())).tier,
                    _jd_priority(skills[i], jd_keywords or [], job_description),
                    -i,
                ),
            )
            skills.pop(drop_idx)
        if skills:
            result.append({"name": name, "skills": skills})
    return result


def _direct_bucket_overlap(
    cat: dict[str, Any],
    skill: str,
    skills_map: dict[str, list[str]],
) -> bool:
    return bool(_buckets_on_line(cat["skills"], skills_map) & set(buckets_for_skill(skill, skills_map)))


def _line_compatible_with_skill(
    cat: dict[str, Any],
    skill: str,
    skills_map: dict[str, list[str]],
) -> bool:
    if not cat["skills"]:
        return set(buckets_for_skill(skill, skills_map)) != set()
    line_buckets = _buckets_on_line(cat["skills"], skills_map)
    return bool(line_buckets & _skill_buckets_affinity(skill, skills_map))


def _category_name_for_skill(skill: str, skills_map: dict[str, list[str]]) -> str:
    bucket = buckets_for_skill(skill, skills_map)[0] if buckets_for_skill(skill, skills_map) else ""
    labels = {
        "full_stack": "Full Stack",
        "languages": "Languages",
        "ai_ml": "AI / ML",
        "cloud_devops": "Cloud & DevOps",
        "tools": "Tools",
    }
    return labels.get(bucket, bucket.replace("_", " ").title())


def _make_room_for_skill(
    name: str,
    skills: list[str],
    new_skill: str,
    scores: dict[str, SkillScore],
    max_chars: int,
    *,
    jd_keywords: list[str],
    job_description: str,
) -> list[str] | None:
    from ai import _skill_fits_line

    current = list(skills)
    new_priority = _jd_priority(new_skill, jd_keywords, job_description)

    if _skill_fits_line(name, current, new_skill, max_chars):
        current.append(new_skill)
        return current

    while current:
        droppable = [
            i
            for i, skill in enumerate(current)
            if scores.get(skill, SkillScore(skill, SkillTier.DROP, ())).tier > SkillTier.STRONG
            or (
                scores.get(skill, SkillScore(skill, SkillTier.DROP, ())).tier == SkillTier.STRONG
                and _jd_priority(skill, jd_keywords, job_description) > new_priority
            )
        ]
        if not droppable:
            return None
        drop_idx = max(
            droppable,
            key=lambda i: (
                scores.get(current[i], SkillScore(current[i], SkillTier.DROP, ())).tier,
                _jd_priority(current[i], jd_keywords, job_description),
                -i,
            ),
        )
        current.pop(drop_idx)
        if _skill_fits_line(name, current, new_skill, max_chars):
            current.append(new_skill)
            return current
    return None


def ensure_strong_skills_present(
    categories: list[dict[str, Any]],
    skills_map: dict[str, list[str]],
    scores: dict[str, SkillScore],
    *,
    max_chars_per_line: int,
    max_categories: int,
    jd_keywords: list[str],
    job_description: str,
) -> list[str]:
    """Ensure every JD-strong skill appears, displacing lower-priority skills if needed."""
    from ai import _skill_already_listed

    added: list[str] = []
    missing_strong = sorted(
        [skill for skill, score in scores.items() if score.tier == SkillTier.STRONG],
        key=lambda s: (_jd_priority(s, jd_keywords, job_description), s.lower()),
    )
    for skill in missing_strong:
        if _skill_already_listed(skill, categories):
            continue

        direct = [
            cat for cat in categories if _direct_bucket_overlap(cat, skill, skills_map)
        ]
        affinity_only = [
            cat
            for cat in categories
            if _line_compatible_with_skill(cat, skill, skills_map)
            and cat not in direct
        ]

        placed = False
        if direct or affinity_only:
            compatible = direct or affinity_only
            compatible.sort(
                key=lambda cat: _line_utilization(cat["name"], cat["skills"], max_chars_per_line)
            )
            for cat in compatible:
                updated = _make_room_for_skill(
                    cat["name"],
                    cat["skills"],
                    skill,
                    scores,
                    max_chars_per_line,
                    jd_keywords=jd_keywords,
                    job_description=job_description,
                )
                if updated is not None:
                    cat["skills"] = _sort_skills_by_tier(
                        updated,
                        scores,
                        jd_keywords=jd_keywords,
                        job_description=job_description,
                    )
                    added.append(skill)
                    placed = True
                    break

        if not placed and len(categories) < max_categories:
            categories.append(
                {"name": _category_name_for_skill(skill, skills_map), "skills": [skill]}
            )
            added.append(skill)

    return added


def select_relevant_skill_categories(
    skill_categories: list[dict[str, Any]],
    skills_map: dict[str, list[str]],
    job_description: str,
    evidence_text: str,
    *,
    max_categories: int,
    max_chars_per_line: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild skills into unnamed grouped lines packed by JD relevance."""
    from ai import skill_line_utilization

    llm_skills = [skill for cat in skill_categories for skill in cat["skills"]]
    packed, scores, jd_keywords = build_packed_skill_lines(
        skills_map,
        job_description,
        evidence_text,
        max_categories=max_categories,
        max_chars_per_line=max_chars_per_line,
    )
    packed_skills = {s.lower() for cat in packed for s in cat["skills"]}

    removed_irrelevant = sorted(
        {s for s in llm_skills if s.lower() not in packed_skills},
        key=str.lower,
    )
    added_skills = sorted(
        {s for cat in packed for s in cat["skills"] if s not in llm_skills},
        key=str.lower,
    )

    skill_tiers = {
        skill: scores[skill].tier.name
        for cat in packed
        for skill in cat["skills"]
        if skill in scores
    }

    diagnostics = {
        "removed_irrelevant": removed_irrelevant,
        "dropped_categories": [
            (cat.get("name") or ", ".join(cat["skills"][:3])).strip()
            for cat in skill_categories
            if cat.get("name")
        ],
        "added_skills": added_skills,
        "skill_tiers": skill_tiers,
        "line_utilization": skill_line_utilization(
            packed,
            max_chars_per_line=max_chars_per_line,
        ),
    }
    return packed, diagnostics
