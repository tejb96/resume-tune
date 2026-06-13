"""Score-driven selection: coarse 1-5 ratings, role-first composites, policy-based build/trim."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

RATING_TO_SCORE: dict[int, float] = {5: 100.0, 4: 75.0, 3: 50.0, 2: 25.0, 1: 10.0}

DEFAULT_EXPERIENCE_ROLE_WEIGHT = 0.7
DEFAULT_EXPERIENCE_BULLET_WEIGHT = 0.3
DEFAULT_PROJECT_RELEVANCE_WEIGHT = 0.8
DEFAULT_PROJECT_BULLET_WEIGHT = 0.2

DEFAULT_MIN_ROLE_RELEVANCE_RATING = 2
DEFAULT_MIN_EXPERIENCE_ROLES = 1
DEFAULT_MIN_BULLETS_PER_ROLE = 1
DEFAULT_MIN_PROJECT_COMPOSITE = 50.0
DEFAULT_PROJECT_BEATS_EXPERIENCE_GAP = 15.0
DEFAULT_PREFER_EXPERIENCE_WITHIN_GAP = 10.0
DEFAULT_MIN_EDUCATION_COMPOSITE = 25.0
DEFAULT_MIN_EDUCATION_RELEVANCE_RATING = 3
DEFAULT_MAX_EDUCATION_ENTRIES = 1
DEFAULT_OVERFLOW_WARNING_MIN_COMPOSITE = 75.0

TrimKind = Literal["experience_bullet", "project_bullet", "education"]
ExpandKind = TrimKind


@dataclass(frozen=True)
class CompositeWeights:
    experience_role_weight: float = DEFAULT_EXPERIENCE_ROLE_WEIGHT
    experience_bullet_weight: float = DEFAULT_EXPERIENCE_BULLET_WEIGHT
    project_relevance_weight: float = DEFAULT_PROJECT_RELEVANCE_WEIGHT
    project_bullet_weight: float = DEFAULT_PROJECT_BULLET_WEIGHT


@dataclass(frozen=True)
class SelectionPolicy:
    min_role_relevance_rating: int = DEFAULT_MIN_ROLE_RELEVANCE_RATING
    max_experience_entries: int = 4
    max_bullets_per_role: int = 3
    min_experience_roles: int = DEFAULT_MIN_EXPERIENCE_ROLES
    min_bullets_per_role: int = DEFAULT_MIN_BULLETS_PER_ROLE
    max_project_entries: int = 2
    # Floors are 0 by default; the app passes config values (usually 1/1).
    min_project_entries: int = 0
    min_project_bullets: int = 0
    min_project_composite: float = DEFAULT_MIN_PROJECT_COMPOSITE
    project_beats_experience_gap: float = DEFAULT_PROJECT_BEATS_EXPERIENCE_GAP
    prefer_experience_within_gap: float = DEFAULT_PREFER_EXPERIENCE_WITHIN_GAP
    min_education_composite: float = DEFAULT_MIN_EDUCATION_COMPOSITE
    min_education_relevance_rating: int = DEFAULT_MIN_EDUCATION_RELEVANCE_RATING
    max_education_entries: int = DEFAULT_MAX_EDUCATION_ENTRIES


@dataclass
class ContentRatings:
    roles: dict[int, int] = field(default_factory=dict)
    experience_bullets: dict[tuple[int, int], int] = field(default_factory=dict)
    projects: dict[int, int] = field(default_factory=dict)
    project_bullets: dict[tuple[int, int], int] = field(default_factory=dict)
    education: dict[int, int] = field(default_factory=dict)


def clamp_rating(rating: int) -> int:
    return max(1, min(5, int(rating)))


def normalize_rating(rating: int) -> float:
    return RATING_TO_SCORE[clamp_rating(rating)]


def experience_bullet_composite(
    role_relevance: int,
    bullet_strength: int,
    *,
    weights: CompositeWeights | None = None,
) -> float:
    w = weights or CompositeWeights()
    role_rel = normalize_rating(role_relevance)
    bullet_str = normalize_rating(bullet_strength)
    return w.experience_role_weight * role_rel + w.experience_bullet_weight * bullet_str


def project_bullet_composite(
    project_relevance: int,
    bullet_strength: int,
    *,
    weights: CompositeWeights | None = None,
) -> float:
    w = weights or CompositeWeights()
    project_rel = normalize_rating(project_relevance)
    bullet_str = normalize_rating(bullet_strength)
    return w.project_relevance_weight * project_rel + w.project_bullet_weight * bullet_str


def education_composite(education_relevance: int) -> float:
    return normalize_rating(education_relevance)


def _experience_list(background_data: dict[str, Any]) -> list[dict[str, Any]]:
    experience = background_data.get("experience", [])
    return experience if isinstance(experience, list) else []


def _projects_list(background_data: dict[str, Any]) -> list[dict[str, Any]]:
    projects = background_data.get("projects", [])
    return projects if isinstance(projects, list) else []


def _education_list(background_data: dict[str, Any]) -> list[dict[str, Any]]:
    education = background_data.get("education", [])
    return education if isinstance(education, list) else []


def parse_resume_date(value: str) -> tuple[int, int]:
    """Parse YYYY-MM, YYYY, or present into (year, month) for sorting."""
    v = value.strip().lower()
    if not v:
        return (0, 0)
    if v == "present":
        return (9999, 12)
    if "-" in v:
        year_s, month_s = v.split("-", 1)
        try:
            return (int(year_s), int(month_s))
        except ValueError:
            return (0, 0)
    try:
        return (int(v), 0)
    except ValueError:
        return (0, 0)


def entry_has_dates(entry: dict[str, Any]) -> bool:
    return bool(str(entry.get("start", "")).strip() or str(entry.get("end", "")).strip())


def entry_recency_key(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    end = parse_resume_date(str(entry.get("end", "")))
    start = parse_resume_date(str(entry.get("start", "")))
    return (end[0], end[1], start[0], start[1])


def sort_experience_selections(
    selection: dict[str, Any],
    background_data: dict[str, Any],
) -> None:
    """Sort experience_selections most-recent-first by job end/start dates."""
    experience = _experience_list(background_data)
    entries = selection.get("experience_selections", [])
    if not entries:
        return

    def sort_key(entry: dict[str, Any]) -> tuple[int, ...]:
        role_index = entry["role_index"]
        if role_index < len(experience):
            recency = entry_recency_key(experience[role_index])
            return tuple(-part for part in recency)
        return (0, 0, 0, 0)

    entries.sort(key=sort_key)


def sort_project_selections(
    selection: dict[str, Any],
    background_data: dict[str, Any],
) -> None:
    """Sort project_selections by date when present; otherwise by background index."""
    projects = _projects_list(background_data)
    entries = selection.get("project_selections", [])
    if not entries:
        return

    def sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
        project_index = entry["project_index"]
        if project_index >= len(projects):
            return (1, project_index)
        project = projects[project_index]
        if entry_has_dates(project):
            recency = entry_recency_key(project)
            return (0, tuple(-part for part in recency), project_index)
        return (1, project_index)

    entries.sort(key=sort_key)


def heuristic_ratings(background_data: dict[str, Any]) -> ContentRatings:
    """Fallback 1-5 ratings when the LLM fails (recency-biased, no fine scores)."""
    ratings = ContentRatings()
    experience = _experience_list(background_data)
    for role_index, job in enumerate(experience):
        base = 4 if role_index == 0 else 3
        ratings.roles[role_index] = min(5, base + (1 if role_index == 0 else 0))
        for bullet_index, _bullet in enumerate(job.get("bullets", [])):
            ratings.experience_bullets[(role_index, bullet_index)] = 4 if bullet_index == 0 else 3

    for project_index, project in enumerate(_projects_list(background_data)):
        ratings.projects[project_index] = 3
        for bullet_index, _bullet in enumerate(project.get("bullets", [])):
            ratings.project_bullets[(project_index, bullet_index)] = 3

    for edu_index, _entry in enumerate(_education_list(background_data)):
        ratings.education[edu_index] = 3

    return ratings


def _rating_list(
    value: Any,
    *,
    label: str,
    expected: int,
    default: int = 3,
) -> list[int]:
    if value is None and expected == 0:
        return []
    if not isinstance(value, list):
        raise ValueError(f"'{label}' must be a list of ratings")
    ratings: list[int] = []
    for raw in value[:expected]:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"'{label}' ratings must be numbers between 1 and 5")
        ratings.append(clamp_rating(round(raw)))
    while len(ratings) < expected:
        ratings.append(clamp_rating(default))
    return ratings


def ratings_from_payload(data: Any, background_data: dict[str, Any]) -> ContentRatings:
    """Convert a model ratings payload (in-order arrays) into validated ContentRatings."""
    if not isinstance(data, dict):
        raise ValueError("Ratings payload must be a JSON object")

    experience = _experience_list(background_data)
    projects = _projects_list(background_data)
    education = _education_list(background_data)

    ratings = ContentRatings()

    role_ratings = _rating_list(data.get("roles"), label="roles", expected=len(experience))
    ratings.roles = dict(enumerate(role_ratings))

    raw_exp_bullets = data.get("experience_bullets")
    if raw_exp_bullets is None and experience:
        raise ValueError("'experience_bullets' is required")
    if experience:
        if not isinstance(raw_exp_bullets, list) or len(raw_exp_bullets) != len(experience):
            raise ValueError(
                f"'experience_bullets' must contain exactly {len(experience)} arrays "
                "(one per role, in order)"
            )
        for role_index, job in enumerate(experience):
            bullet_ratings = _rating_list(
                raw_exp_bullets[role_index],
                label=f"experience_bullets[{role_index}]",
                expected=len(job.get("bullets", [])),
            )
            for bullet_index, rating in enumerate(bullet_ratings):
                ratings.experience_bullets[(role_index, bullet_index)] = rating

    project_ratings = _rating_list(data.get("projects"), label="projects", expected=len(projects))
    ratings.projects = dict(enumerate(project_ratings))

    raw_project_bullets = data.get("project_bullets")
    if projects:
        if not isinstance(raw_project_bullets, list) or len(raw_project_bullets) != len(projects):
            raise ValueError(
                f"'project_bullets' must contain exactly {len(projects)} arrays "
                "(one per project, in order)"
            )
        for project_index, project in enumerate(projects):
            bullet_ratings = _rating_list(
                raw_project_bullets[project_index],
                label=f"project_bullets[{project_index}]",
                expected=len(project.get("bullets", [])),
            )
            for bullet_index, rating in enumerate(bullet_ratings):
                ratings.project_bullets[(project_index, bullet_index)] = rating

    education_ratings = _rating_list(
        data.get("education"), label="education", expected=len(education)
    )
    ratings.education = dict(enumerate(education_ratings))

    return ratings


def compute_content_composites(
    background_data: dict[str, Any],
    ratings: ContentRatings,
    *,
    weights: CompositeWeights | None = None,
) -> dict[str, float]:
    """Return composite scores keyed as exp:r:b, proj:p:b, edu:e."""
    composites: dict[str, float] = {}
    for role_index, job in enumerate(_experience_list(background_data)):
        role_rel = ratings.roles.get(role_index, 3)
        for bullet_index, _bullet in enumerate(job.get("bullets", [])):
            strength = ratings.experience_bullets.get((role_index, bullet_index), 3)
            composites[f"exp:{role_index}:{bullet_index}"] = experience_bullet_composite(
                role_rel, strength, weights=weights
            )

    for project_index, project in enumerate(_projects_list(background_data)):
        project_rel = ratings.projects.get(project_index, 2)
        for bullet_index, _bullet in enumerate(project.get("bullets", [])):
            strength = ratings.project_bullets.get((project_index, bullet_index), 3)
            composites[f"proj:{project_index}:{bullet_index}"] = project_bullet_composite(
                project_rel, strength, weights=weights
            )

    for edu_index, _entry in enumerate(_education_list(background_data)):
        rel = ratings.education.get(edu_index, 3)
        composites[f"edu:{edu_index}"] = education_composite(rel)

    return composites


def inventory_selection_policy(
    background_data: dict[str, Any],
    *,
    min_project_entries: int = 0,
    min_project_bullets: int = 0,
) -> SelectionPolicy:
    """Policy with max limits derived from background inventory (page fit drives final size)."""
    experience = _experience_list(background_data)
    projects = _projects_list(background_data)
    education = _education_list(background_data)
    max_bullets = max((len(job.get("bullets", [])) for job in experience), default=1)
    return SelectionPolicy(
        max_experience_entries=max(len(experience), 1),
        max_bullets_per_role=max(max_bullets, 1),
        max_project_entries=max(len(projects), 1),
        max_education_entries=max(len(education), 1),
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
    )


def ratings_to_dict(ratings: ContentRatings) -> dict[str, Any]:
    return {
        "roles": {str(k): v for k, v in ratings.roles.items()},
        "experience_bullets": {f"{r}:{b}": v for (r, b), v in ratings.experience_bullets.items()},
        "projects": {str(k): v for k, v in ratings.projects.items()},
        "project_bullets": {f"{p}:{b}": v for (p, b), v in ratings.project_bullets.items()},
        "education": {str(k): v for k, v in ratings.education.items()},
    }


def build_selection_from_scores(
    background_data: dict[str, Any],
    ratings: ContentRatings,
    *,
    policy: SelectionPolicy | None = None,
    weights: CompositeWeights | None = None,
) -> dict[str, Any]:
    """Build experience/project/education selection from ratings and explicit policy."""
    policy = policy or SelectionPolicy()
    composites = compute_content_composites(background_data, ratings, weights=weights)
    experience = _experience_list(background_data)
    projects = _projects_list(background_data)
    education = _education_list(background_data)

    role_candidates = [
        (role_index, ratings.roles.get(role_index, 3), normalize_rating(ratings.roles.get(role_index, 3)))
        for role_index in range(len(experience))
    ]
    role_candidates.sort(key=lambda item: item[2], reverse=True)

    selected_roles: list[int] = []
    for role_index, relevance_rating, _score in role_candidates:
        if len(selected_roles) >= policy.max_experience_entries:
            break
        if relevance_rating >= policy.min_role_relevance_rating:
            selected_roles.append(role_index)

    if not selected_roles and role_candidates:
        selected_roles = [role_candidates[0][0]]

    experience_selections: list[dict[str, Any]] = []
    all_exp_bullet_scores: list[tuple[int, int, float]] = []

    for role_index in selected_roles:
        job = experience[role_index]
        bullet_scores = [
            (
                bullet_index,
                composites.get(f"exp:{role_index}:{bullet_index}", 0.0),
            )
            for bullet_index in range(len(job.get("bullets", [])))
        ]
        bullet_scores.sort(key=lambda item: item[1], reverse=True)
        chosen = [idx for idx, _score in bullet_scores[: policy.max_bullets_per_role]]
        if chosen:
            experience_selections.append({"role_index": role_index, "bullet_indices": chosen})
            for bullet_index, score in bullet_scores[: policy.max_bullets_per_role]:
                all_exp_bullet_scores.append((role_index, bullet_index, score))

    weakest_exp_score = (
        min(score for _r, _b, score in all_exp_bullet_scores) if all_exp_bullet_scores else 0.0
    )

    scored_projects: list[tuple[int, list[tuple[int, float]], float]] = []
    for project_index, project in enumerate(projects):
        bullet_scores = [
            (
                bullet_index,
                composites.get(f"proj:{project_index}:{bullet_index}", 0.0),
            )
            for bullet_index in range(len(project.get("bullets", [])))
        ]
        if not bullet_scores:
            continue
        bullet_scores.sort(key=lambda item: item[1], reverse=True)
        scored_projects.append((project_index, bullet_scores, bullet_scores[0][1]))

    project_candidates: list[tuple[int, list[int], float]] = []
    for project_index, bullet_scores, best_score in scored_projects:
        if best_score < policy.min_project_composite:
            continue
        if all_exp_bullet_scores and best_score < weakest_exp_score + policy.project_beats_experience_gap:
            continue
        chosen = [idx for idx, _score in bullet_scores[: policy.max_bullets_per_role]]
        project_candidates.append((project_index, chosen, best_score))

    project_candidates.sort(key=lambda item: item[2], reverse=True)
    selected_project_entries = project_candidates[: policy.max_project_entries]

    # Floor: top up with the best remaining project(s), one bullet each, when gates
    # filtered out more than the configured minimum allows.
    project_floor = min(policy.min_project_entries, policy.max_project_entries)
    if len(selected_project_entries) < project_floor:
        chosen_indices = {entry[0] for entry in selected_project_entries}
        remaining = sorted(
            (sp for sp in scored_projects if sp[0] not in chosen_indices),
            key=lambda item: item[2],
            reverse=True,
        )
        for project_index, bullet_scores, best_score in remaining:
            if len(selected_project_entries) >= project_floor:
                break
            selected_project_entries.append((project_index, [bullet_scores[0][0]], best_score))

    project_selections: list[dict[str, Any]] = [
        {"project_index": project_index, "bullet_indices": bullet_indices}
        for project_index, bullet_indices, _score in selected_project_entries
    ]

    education_indices: list[int] = []
    edu_scored = [
        (
            edu_index,
            composites.get(f"edu:{edu_index}", 0.0),
            ratings.education.get(edu_index, 3),
        )
        for edu_index in range(len(education))
    ]
    edu_scored.sort(key=lambda item: item[1], reverse=True)
    for edu_index, score, relevance_rating in edu_scored:
        if len(education_indices) >= policy.max_education_entries:
            break
        if relevance_rating < policy.min_education_relevance_rating:
            continue
        if score >= policy.min_education_composite:
            education_indices.append(edu_index)

    if not education_indices and edu_scored:
        education_indices = [edu_scored[0][0]]

    result = {
        "experience_selections": experience_selections,
        "project_selections": project_selections,
        "education_indices": education_indices,
        "content_ratings": ratings_to_dict(ratings),
        "content_composites": composites,
    }
    sort_experience_selections(result, background_data)
    sort_project_selections(result, background_data)
    return result


def _anchor_role_index(
    selection: dict[str, Any],
    composites: dict[str, float],
) -> int | None:
    """Role with the highest composite among selected experience bullets."""
    best_role: int | None = None
    best_score = -1.0
    for entry in selection.get("experience_selections", []):
        role_index = entry["role_index"]
        for bullet_index in entry.get("bullet_indices", []):
            score = composites.get(f"exp:{role_index}:{bullet_index}", 0.0)
            if score > best_score:
                best_score = score
                best_role = role_index
    return best_role


def _trim_priority(kind: TrimKind, *, is_anchor_role: bool) -> int:
    if kind == "project_bullet":
        return 0
    if kind == "education":
        return 1
    if kind == "experience_bullet":
        return 3 if is_anchor_role else 2
    return 4


@dataclass(frozen=True)
class TrimmableItem:
    kind: TrimKind
    composite: float
    role_index: int | None = None
    bullet_index: int | None = None
    project_index: int | None = None
    education_index: int | None = None

    @property
    def key(self) -> str:
        if self.kind == "experience_bullet":
            return f"exp:{self.role_index}:{self.bullet_index}"
        if self.kind == "project_bullet":
            return f"proj:{self.project_index}:{self.bullet_index}"
        return f"edu:{self.education_index}"

    @property
    def trim_priority(self) -> int:
        is_anchor = False
        return _trim_priority(self.kind, is_anchor_role=is_anchor)


def _enumerate_trimmable_items(
    selection: dict[str, Any],
    composites: dict[str, float],
    policy: SelectionPolicy,
) -> tuple[list[TrimmableItem], list[TrimmableItem]]:
    """Return (trimmable, protected) items given current selection and floors."""
    anchor_role = _anchor_role_index(selection, composites)
    trimmable: list[TrimmableItem] = []
    protected: list[TrimmableItem] = []

    role_bullet_counts: dict[int, int] = {}
    for entry in selection.get("experience_selections", []):
        role_index = entry["role_index"]
        bullets = entry.get("bullet_indices", [])
        role_bullet_counts[role_index] = len(bullets)
        for bullet_index in bullets:
            item = TrimmableItem(
                kind="experience_bullet",
                composite=composites.get(f"exp:{role_index}:{bullet_index}", 0.0),
                role_index=role_index,
                bullet_index=bullet_index,
            )
            is_anchor = role_index == anchor_role
            at_floor = role_bullet_counts[role_index] <= policy.min_bullets_per_role
            only_role = len(selection.get("experience_selections", [])) <= policy.min_experience_roles
            if is_anchor and at_floor and only_role:
                protected.append(item)
            elif role_bullet_counts[role_index] <= 1 and only_role:
                protected.append(item)
            else:
                trimmable.append(item)

    project_entries = selection.get("project_selections", [])
    total_project_bullets = sum(len(p.get("bullet_indices", [])) for p in project_entries)
    at_project_floor = (
        policy.min_project_entries > 0
        and len(project_entries) <= policy.min_project_entries
        and total_project_bullets <= policy.min_project_bullets
    )
    for entry in project_entries:
        for bullet_index in entry.get("bullet_indices", []):
            item = TrimmableItem(
                kind="project_bullet",
                composite=composites.get(
                    f"proj:{entry['project_index']}:{bullet_index}", 0.0
                ),
                project_index=entry["project_index"],
                bullet_index=bullet_index,
            )
            if at_project_floor:
                protected.append(item)
            else:
                trimmable.append(item)

    edu_indices = selection.get("education_indices", [])
    if len(edu_indices) > 1:
        for edu_index in edu_indices:
            trimmable.append(
                TrimmableItem(
                    kind="education",
                    composite=composites.get(f"edu:{edu_index}", 0.0),
                    education_index=edu_index,
                )
            )
    elif len(edu_indices) == 1:
        edu_index = edu_indices[0]
        protected.append(
            TrimmableItem(
                kind="education",
                composite=composites.get(f"edu:{edu_index}", 0.0),
                education_index=edu_index,
            )
        )

    return trimmable, protected


def _pick_trim_candidate(
    trimmable: list[TrimmableItem],
    *,
    policy: SelectionPolicy,
    anchor_role: int | None,
) -> TrimmableItem | None:
    if not trimmable:
        return None

    min_composite = min(item.composite for item in trimmable)
    candidates = [item for item in trimmable if item.composite <= min_composite + 0.001]

    gap = policy.prefer_experience_within_gap
    near_min = [item for item in trimmable if item.composite <= min_composite + gap]
    if len(near_min) > 1:
        candidates = near_min

    def sort_key(item: TrimmableItem) -> tuple[float, int]:
        is_anchor = item.kind == "experience_bullet" and item.role_index == anchor_role
        priority = _trim_priority(item.kind, is_anchor_role=is_anchor)
        return (item.composite, priority)

    candidates.sort(key=sort_key)
    return candidates[0]


def _remove_trim_item(selection: dict[str, Any], item: TrimmableItem) -> dict[str, Any]:
    result = {
        "experience_selections": [
            {"role_index": e["role_index"], "bullet_indices": list(e["bullet_indices"])}
            for e in selection.get("experience_selections", [])
        ],
        "project_selections": [
            {"project_index": p["project_index"], "bullet_indices": list(p["bullet_indices"])}
            for p in selection.get("project_selections", [])
        ],
        "education_indices": list(selection.get("education_indices", [])),
        "content_ratings": deepcopy(selection.get("content_ratings", {})),
        "content_composites": deepcopy(selection.get("content_composites", {})),
        "trim_log": list(selection.get("trim_log", [])),
        "expand_log": list(selection.get("expand_log", [])),
    }

    if item.kind == "experience_bullet":
        for entry in result["experience_selections"]:
            if entry["role_index"] == item.role_index and item.bullet_index in entry["bullet_indices"]:
                entry["bullet_indices"].remove(item.bullet_index)
        result["experience_selections"] = [
            e for e in result["experience_selections"] if e["bullet_indices"]
        ]
    elif item.kind == "project_bullet":
        for entry in result["project_selections"]:
            if (
                entry["project_index"] == item.project_index
                and item.bullet_index in entry["bullet_indices"]
            ):
                entry["bullet_indices"].remove(item.bullet_index)
        result["project_selections"] = [
            p for p in result["project_selections"] if p["bullet_indices"]
        ]
    elif item.kind == "education" and item.education_index in result["education_indices"]:
        result["education_indices"].remove(item.education_index)

    return result


def trim_selection_by_lowest_score(
    selection: dict[str, Any],
    *,
    policy: SelectionPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Remove the lowest composite trimmable item (with policy tie-breaks).

    Returns (updated_selection, trim_event or None if nothing removed).
    """
    policy = policy or SelectionPolicy()
    composites = selection.get("content_composites", {})
    if not composites:
        return selection, None

    trimmable, _protected = _enumerate_trimmable_items(selection, composites, policy)
    anchor_role = _anchor_role_index(selection, composites)
    candidate = _pick_trim_candidate(trimmable, policy=policy, anchor_role=anchor_role)
    if candidate is None:
        return selection, None

    updated = _remove_trim_item(selection, candidate)
    is_anchor = candidate.kind == "experience_bullet" and candidate.role_index == anchor_role
    tie_break = (
        f"policy tie-break (priority {_trim_priority(candidate.kind, is_anchor_role=is_anchor)})"
        if any(
            abs(item.composite - candidate.composite) <= policy.prefer_experience_within_gap
            for item in trimmable
            if item.key != candidate.key
        )
        else "lowest composite"
    )
    trim_event = {
        "removed": candidate.key,
        "composite": candidate.composite,
        "reason": tie_break,
    }
    updated["trim_log"] = list(updated.get("trim_log", [])) + [trim_event]
    return updated, trim_event


def selection_trim_exhausted(
    selection: dict[str, Any],
    *,
    policy: SelectionPolicy | None = None,
) -> bool:
    """Return True when no score-driven trimmable content remains."""
    policy = policy or SelectionPolicy()
    composites = selection.get("content_composites", {})
    if not composites:
        return True
    trimmable, _ = _enumerate_trimmable_items(selection, composites, policy)
    return not trimmable


def _selected_experience_bullets(selection: dict[str, Any]) -> set[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    for entry in selection.get("experience_selections", []):
        role_index = entry["role_index"]
        for bullet_index in entry.get("bullet_indices", []):
            selected.add((role_index, bullet_index))
    return selected


def _selected_project_bullets(selection: dict[str, Any]) -> set[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    for entry in selection.get("project_selections", []):
        project_index = entry["project_index"]
        for bullet_index in entry.get("bullet_indices", []):
            selected.add((project_index, bullet_index))
    return selected


@dataclass(frozen=True)
class ExpandableItem:
    kind: ExpandKind
    composite: float
    role_index: int | None = None
    bullet_index: int | None = None
    project_index: int | None = None
    education_index: int | None = None

    @property
    def key(self) -> str:
        if self.kind == "experience_bullet":
            return f"exp:{self.role_index}:{self.bullet_index}"
        if self.kind == "project_bullet":
            return f"proj:{self.project_index}:{self.bullet_index}"
        return f"edu:{self.education_index}"


def _expand_priority(kind: ExpandKind) -> int:
    if kind == "experience_bullet":
        return 3
    if kind == "project_bullet":
        return 2
    return 1


def enumerate_expandable_items(
    selection: dict[str, Any],
    background_data: dict[str, Any],
    composites: dict[str, float],
) -> list[ExpandableItem]:
    """Return unselected scored items from background inventory, highest composite first."""
    selected_exp = _selected_experience_bullets(selection)
    selected_proj = _selected_project_bullets(selection)
    selected_edu = set(selection.get("education_indices", []))

    expandable: list[ExpandableItem] = []

    for role_index, job in enumerate(_experience_list(background_data)):
        for bullet_index, _bullet in enumerate(job.get("bullets", [])):
            if (role_index, bullet_index) in selected_exp:
                continue
            key = f"exp:{role_index}:{bullet_index}"
            if key not in composites:
                continue
            expandable.append(
                ExpandableItem(
                    kind="experience_bullet",
                    composite=composites[key],
                    role_index=role_index,
                    bullet_index=bullet_index,
                )
            )

    for project_index, project in enumerate(_projects_list(background_data)):
        for bullet_index, _bullet in enumerate(project.get("bullets", [])):
            if (project_index, bullet_index) in selected_proj:
                continue
            key = f"proj:{project_index}:{bullet_index}"
            if key not in composites:
                continue
            expandable.append(
                ExpandableItem(
                    kind="project_bullet",
                    composite=composites[key],
                    project_index=project_index,
                    bullet_index=bullet_index,
                )
            )

    for edu_index, _entry in enumerate(_education_list(background_data)):
        if edu_index in selected_edu:
            continue
        key = f"edu:{edu_index}"
        if key not in composites:
            continue
        expandable.append(
            ExpandableItem(
                kind="education",
                composite=composites[key],
                education_index=edu_index,
            )
        )

    expandable.sort(key=lambda item: (-item.composite, -_expand_priority(item.kind)))
    return expandable


def _pick_expand_candidate(expandable: list[ExpandableItem]) -> ExpandableItem | None:
    if not expandable:
        return None
    max_composite = max(item.composite for item in expandable)
    candidates = [item for item in expandable if item.composite >= max_composite - 0.001]
    candidates.sort(key=lambda item: (-item.composite, -_expand_priority(item.kind)))
    return candidates[0]


def _add_expand_item(
    selection: dict[str, Any],
    item: ExpandableItem,
    *,
    background_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "experience_selections": [
            {"role_index": e["role_index"], "bullet_indices": list(e["bullet_indices"])}
            for e in selection.get("experience_selections", [])
        ],
        "project_selections": [
            {"project_index": p["project_index"], "bullet_indices": list(p["bullet_indices"])}
            for p in selection.get("project_selections", [])
        ],
        "education_indices": list(selection.get("education_indices", [])),
        "content_ratings": deepcopy(selection.get("content_ratings", {})),
        "content_composites": deepcopy(selection.get("content_composites", {})),
        "trim_log": list(selection.get("trim_log", [])),
        "expand_log": list(selection.get("expand_log", [])),
    }

    if item.kind == "experience_bullet":
        assert item.role_index is not None and item.bullet_index is not None
        for entry in result["experience_selections"]:
            if entry["role_index"] == item.role_index:
                if item.bullet_index not in entry["bullet_indices"]:
                    entry["bullet_indices"].append(item.bullet_index)
                    entry["bullet_indices"].sort()
                break
        else:
            result["experience_selections"].append(
                {"role_index": item.role_index, "bullet_indices": [item.bullet_index]}
            )
    elif item.kind == "project_bullet":
        assert item.project_index is not None and item.bullet_index is not None
        for entry in result["project_selections"]:
            if entry["project_index"] == item.project_index:
                if item.bullet_index not in entry["bullet_indices"]:
                    entry["bullet_indices"].append(item.bullet_index)
                    entry["bullet_indices"].sort()
                break
        else:
            result["project_selections"].append(
                {"project_index": item.project_index, "bullet_indices": [item.bullet_index]}
            )
    elif item.kind == "education":
        assert item.education_index is not None
        if item.education_index not in result["education_indices"]:
            result["education_indices"].append(item.education_index)
            result["education_indices"].sort()

    if background_data is not None:
        sort_experience_selections(result, background_data)
        sort_project_selections(result, background_data)

    return result


def expand_selection_by_highest_score(
    selection: dict[str, Any],
    background_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Add the highest composite unselected item from background inventory.

    Returns (updated_selection, expand_event or None if nothing added).
    """
    composites = selection.get("content_composites", {})
    if not composites:
        return selection, None

    expandable = enumerate_expandable_items(selection, background_data, composites)
    candidate = _pick_expand_candidate(expandable)
    if candidate is None:
        return selection, None

    updated = _add_expand_item(selection, candidate, background_data=background_data)
    expand_event = {
        "added": candidate.key,
        "composite": candidate.composite,
        "reason": "highest composite",
    }
    updated["expand_log"] = list(updated.get("expand_log", [])) + [expand_event]
    return updated, expand_event


def selection_expand_exhausted(
    selection: dict[str, Any],
    background_data: dict[str, Any],
) -> bool:
    """Return True when no score-driven expandable content remains."""
    composites = selection.get("content_composites", {})
    if not composites:
        return True
    return not enumerate_expandable_items(selection, background_data, composites)


def high_quality_omitted_items(
    selection: dict[str, Any],
    background_data: dict[str, Any],
    *,
    min_composite: float = DEFAULT_OVERFLOW_WARNING_MIN_COMPOSITE,
) -> list[ExpandableItem]:
    """Unselected items at or above the high-quality composite threshold."""
    composites = selection.get("content_composites", {})
    if not composites:
        return []
    return [
        item
        for item in enumerate_expandable_items(selection, background_data, composites)
        if item.composite >= min_composite
    ]


def selection_with_all_items(
    selection: dict[str, Any],
    items: list[ExpandableItem],
    *,
    background_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of selection with all given expandable items added."""
    updated = selection
    for item in items:
        updated = _add_expand_item(updated, item, background_data=background_data)
    return updated


def top_experience_bullets_text(
    background_data: dict[str, Any],
    selection: dict[str, Any],
    *,
    limit: int = 12,
) -> str:
    """Format top-scored selected experience bullets for the skills prompt."""
    composites = selection.get("content_composites", {})
    experience = _experience_list(background_data)
    scored: list[tuple[float, str]] = []

    for entry in selection.get("experience_selections", []):
        role_index = entry["role_index"]
        if role_index >= len(experience):
            continue
        job = experience[role_index]
        title = job.get("title", "")
        company = job.get("company", "")
        for bullet_index in entry.get("bullet_indices", []):
            bullets = job.get("bullets", [])
            if bullet_index >= len(bullets):
                continue
            key = f"exp:{role_index}:{bullet_index}"
            score = composites.get(key, 0.0)
            scored.append((score, f"{title} at {company}: {bullets[bullet_index]}"))

    scored.sort(key=lambda item: item[0], reverse=True)
    lines = [text for _score, text in scored[:limit]]
    return "\n".join(lines)
