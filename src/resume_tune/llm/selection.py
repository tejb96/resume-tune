"""Job-aware selection of experience, project, and education content from background data."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

from resume_tune.llm.ai import AIResponseError, strip_json_fences
from resume_tune.content.scoring import (
    ContentRatings,
    SelectionPolicy,
    build_selection_from_scores,
    heuristic_ratings,
    inventory_selection_policy,
    ratings_from_payload,
    sort_experience_selections,
    sort_project_selections,
)

DEFAULT_MIN_PROJECT_ENTRIES = 1
DEFAULT_MIN_PROJECT_BULLETS = 1

RATINGS_SYSTEM_PROMPT_TEMPLATE = """You are rating a candidate's resume content for relevance to one specific job description (provided by the user).

Rate EVERY item below on this scale:
5 = direct match for the job's core stack and responsibilities
4 = strong match (same kind of work, closely adjacent technology)
3 = partially relevant or transferable
2 = weakly related
1 = unrelated to this job

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- Every rating must be an integer 1-5. Rate each item independently against the job description. Do NOT give every item the same rating.
- Rate each ROLE by how closely its overall work matches the job's stack, responsibilities, and domain.
- Rate each BULLET by how directly it proves a requirement stated in the job description (technology match counts most, then the kind of work, then impact).
- Rate each PROJECT by how relevant its stack and purpose are to the job.
- Rate each EDUCATION entry by how relevant the degree's field is to the job (rate degrees in unrelated fields 1-2).
- JSON schema — array lengths MUST match exactly, items in the same order as CANDIDATE CONTENT:
{schema_skeleton}
- Example output shape (your counts will differ): {{"roles": [5, 2], "experience_bullets": [[4, 5, 3], [2, 1]], "projects": [4], "project_bullets": [[3, 4]], "education": [5, 1]}}

CANDIDATE CONTENT:
{indexed_content}
"""


def _experience_list(background_data: dict[str, Any]) -> list[dict[str, Any]]:
    experience = background_data.get("experience", [])
    return experience if isinstance(experience, list) else []


def _projects_list(background_data: dict[str, Any]) -> list[dict[str, Any]]:
    projects = background_data.get("projects", [])
    return projects if isinstance(projects, list) else []


def _education_list(background_data: dict[str, Any]) -> list[dict[str, Any]]:
    education = background_data.get("education", [])
    return education if isinstance(education, list) else []


def format_indexed_content(background_data: dict[str, Any]) -> str:
    """Format experience, projects, and education with indices for the selection prompt."""
    lines: list[str] = []

    for role_index, job in enumerate(_experience_list(background_data)):
        lines.append(f"Role {role_index}: {job.get('title', '')} at {job.get('company', '')}")
        for bullet_index, bullet in enumerate(job.get("bullets", [])):
            lines.append(f"  [{role_index}][{bullet_index}] {bullet}")

    for project_index, project in enumerate(_projects_list(background_data)):
        lines.append(f"Project {project_index}: {project.get('name', '')}")
        for bullet_index, bullet in enumerate(project.get("bullets", [])):
            lines.append(f"  [P{project_index}][{bullet_index}] {bullet}")

    for edu_index, entry in enumerate(_education_list(background_data)):
        degree = entry.get("degree", "")
        institution = entry.get("institution", "")
        lines.append(f"Education {edu_index}: {degree} — {institution}")

    return "\n".join(lines)


def _all_education_indices(background_data: dict[str, Any]) -> list[int]:
    return list(range(len(_education_list(background_data))))


def empty_selection() -> dict[str, Any]:
    return {
        "experience_selections": [],
        "project_selections": [],
        "education_indices": [],
    }


def full_selection(background_data: dict[str, Any]) -> dict[str, Any]:
    """Select every experience/project bullet and all education (for page-fit trimming)."""
    experience_selections: list[dict[str, Any]] = []
    for role_index, job in enumerate(_experience_list(background_data)):
        bullet_indices = list(range(len(job.get("bullets", []))))
        if bullet_indices:
            experience_selections.append(
                {"role_index": role_index, "bullet_indices": bullet_indices}
            )

    project_selections: list[dict[str, Any]] = []
    for project_index, project in enumerate(_projects_list(background_data)):
        bullet_indices = list(range(len(project.get("bullets", []))))
        if bullet_indices:
            project_selections.append(
                {"project_index": project_index, "bullet_indices": bullet_indices}
            )

    return {
        "experience_selections": experience_selections,
        "project_selections": project_selections,
        "education_indices": _all_education_indices(background_data),
    }


def default_selection(
    background_data: dict[str, Any],
    *,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> dict[str, Any]:
    """Score-based seed selection using recency-biased heuristic ratings."""
    policy = inventory_selection_policy(
        background_data,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
    )
    return build_selection_from_scores(
        background_data,
        heuristic_ratings(background_data),
        policy=policy,
    )


def _project_bullet_count(selection: dict[str, Any]) -> int:
    return sum(len(p.get("bullet_indices", [])) for p in selection.get("project_selections", []))


def selection_at_project_floor(
    selection: dict[str, Any],
    *,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> bool:
    """Return True when project selections are at or below the configured floor."""
    project_entries = len(selection.get("project_selections", []))
    project_bullets = _project_bullet_count(selection)
    return project_entries <= min_project_entries and project_bullets <= min_project_bullets


def enforce_project_floor(
    background_data: dict[str, Any],
    selection: dict[str, Any],
    *,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> dict[str, Any]:
    """Ensure at least min projects with min total bullets when background has projects."""
    projects = _projects_list(background_data)
    if not projects:
        return selection

    result = {
        "experience_selections": list(selection.get("experience_selections", [])),
        "project_selections": [
            {
                "project_index": p["project_index"],
                "bullet_indices": list(p.get("bullet_indices", [])),
            }
            for p in selection.get("project_selections", [])
        ],
        "education_indices": list(selection.get("education_indices", [])),
    }

    for entry in result["project_selections"]:
        if not entry["bullet_indices"]:
            project_index = entry["project_index"]
            if 0 <= project_index < len(projects) and projects[project_index].get("bullets"):
                entry["bullet_indices"] = [0]

    result["project_selections"] = [
        e for e in result["project_selections"] if e["bullet_indices"]
    ]

    if not result["project_selections"] and min_project_entries > 0 and min_project_bullets > 0:
        for project_index, project in enumerate(projects):
            if project.get("bullets"):
                result["project_selections"] = [
                    {"project_index": project_index, "bullet_indices": [0]}
                ]
                break

    return result


def parse_ratings_response(raw: str, background_data: dict[str, Any]) -> ContentRatings:
    """Parse and validate a model JSON ratings response."""
    cleaned = strip_json_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        snippet = cleaned[:500] + ("..." if len(cleaned) > 500 else "")
        raise AIResponseError(
            f"Model returned invalid JSON for ratings: {exc}. Response snippet: {snippet!r}"
        ) from exc

    try:
        return ratings_from_payload(data, background_data)
    except ValueError as exc:
        raise AIResponseError(str(exc)) from exc


def apply_content_selection(
    background_data: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Return a background copy with experience/projects/education filtered by selection."""
    render_data = deepcopy(background_data)
    experience = _experience_list(background_data)
    projects = _projects_list(background_data)
    education = _education_list(background_data)

    ordered_selection = {
        "experience_selections": [
            {
                "role_index": e["role_index"],
                "bullet_indices": list(e["bullet_indices"]),
            }
            for e in selection.get("experience_selections", [])
        ],
        "project_selections": [
            {
                "project_index": p["project_index"],
                "bullet_indices": list(p["bullet_indices"]),
            }
            for p in selection.get("project_selections", [])
        ],
    }
    sort_experience_selections(ordered_selection, background_data)
    sort_project_selections(ordered_selection, background_data)

    selected_experience: list[dict[str, Any]] = []
    for entry in ordered_selection["experience_selections"]:
        role_index = entry["role_index"]
        job = deepcopy(experience[role_index])
        source_bullets = experience[role_index].get("bullets", [])
        job["bullets"] = [source_bullets[i] for i in entry["bullet_indices"]]
        selected_experience.append(job)

    selected_projects: list[dict[str, Any]] = []
    for entry in ordered_selection["project_selections"]:
        project_index = entry["project_index"]
        project = deepcopy(projects[project_index])
        source_bullets = projects[project_index].get("bullets", [])
        project["bullets"] = [source_bullets[i] for i in entry["bullet_indices"]]
        selected_projects.append(project)

    education_indices = selection.get("education_indices")
    if education_indices is not None:
        selected_education = [deepcopy(education[i]) for i in education_indices if i < len(education)]
    else:
        selected_education = list(education)

    render_data["experience"] = selected_experience
    render_data["projects"] = selected_projects
    render_data["education"] = selected_education
    return render_data


def static_content_stats(
    background_data: dict[str, Any],
    selection: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Count experience entries, bullets, and projects in render data."""
    render_data = (
        apply_content_selection(background_data, selection)
        if selection is not None
        else background_data
    )
    experience = render_data.get("experience", [])
    projects = render_data.get("projects", [])
    experience_bullets = sum(len(job.get("bullets", [])) for job in experience)
    project_bullets = sum(len(project.get("bullets", [])) for project in projects)
    return {
        "experience_entries": len(experience),
        "experience_bullets": experience_bullets,
        "project_entries": len(projects),
        "project_bullets": project_bullets,
        "education_entries": len(render_data.get("education", [])),
    }


def _copy_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "experience_selections": [
            {"role_index": e["role_index"], "bullet_indices": list(e["bullet_indices"])}
            for e in selection.get("experience_selections", [])
        ],
        "project_selections": [
            {"project_index": p["project_index"], "bullet_indices": list(p["bullet_indices"])}
            for p in selection.get("project_selections", [])
        ],
        "education_indices": list(selection.get("education_indices", [])),
    }


def trim_selection_one_step(
    selection: dict[str, Any],
    *,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> dict[str, Any]:
    """
    Remove the lowest-relevance content one step at a time.

    Order: experience bullets → project bullets (respecting floor) → education entries.
    """
    current = _copy_selection(selection)

    if current["experience_selections"]:
        last_role = current["experience_selections"][-1]
        if last_role["bullet_indices"]:
            last_role["bullet_indices"].pop()
            if not last_role["bullet_indices"]:
                current["experience_selections"].pop()
            return current

    at_floor = selection_at_project_floor(
        current,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
    )
    if current["project_selections"] and not at_floor:
        last_project = current["project_selections"][-1]
        if last_project["bullet_indices"]:
            last_project["bullet_indices"].pop()
            if not last_project["bullet_indices"]:
                current["project_selections"].pop()
            return current

    if len(current.get("education_indices", [])) > 1:
        current["education_indices"].pop()
        return current

    return current


def selection_trim_exhausted(
    selection: dict[str, Any],
    *,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> bool:
    """Return True when selection has no trimmable content left."""
    for entry in selection.get("experience_selections", []):
        if entry.get("bullet_indices"):
            return False

    if not selection_at_project_floor(
        selection,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
    ):
        for entry in selection.get("project_selections", []):
            if entry.get("bullet_indices"):
                return False

    if len(selection.get("education_indices", [])) > 1:
        return False

    return True


def _ratings_schema_skeleton(background_data: dict[str, Any]) -> str:
    """Render the expected ratings JSON shape with exact array lengths."""
    experience = _experience_list(background_data)
    projects = _projects_list(background_data)
    education = _education_list(background_data)

    exp_bullets = ", ".join(
        f"[{len(job.get('bullets', []))} integers]" for job in experience
    )
    proj_bullets = ", ".join(
        f"[{len(project.get('bullets', []))} integers]" for project in projects
    )
    return (
        "{"
        f'"roles": [{len(experience)} integers], '
        f'"experience_bullets": [{exp_bullets}], '
        f'"projects": [{len(projects)} integers], '
        f'"project_bullets": [{proj_bullets}], '
        f'"education": [{len(education)} integers]'
        "}"
    )


def build_ratings_system_prompt(background_data: dict[str, Any]) -> str:
    return RATINGS_SYSTEM_PROMPT_TEMPLATE.format(
        schema_skeleton=_ratings_schema_skeleton(background_data),
        indexed_content=format_indexed_content(background_data),
    )


def selection_policy_for_background(
    background_data: dict[str, Any],
    *,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> SelectionPolicy:
    return inventory_selection_policy(
        background_data,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
    )


def generate_content_selection(
    job_description: str,
    background_data: dict[str, Any],
    *,
    endpoint_url: str,
    model_name: str,
    api_key: str = "ollama",
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> dict[str, Any]:
    """
    Rate each background item against the job with the LLM, then build the
    selection deterministically from those scores.

    The model only grades items 1-5 (an easy, per-item task); role/bullet/project/
    education choices, caps, and floors are applied in code. Falls back to
    heuristic recency-based ratings when the model fails after retries.
    """
    if not job_description.strip():
        raise ValueError("Job description cannot be empty")

    policy = selection_policy_for_background(
        background_data,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
    )

    if not _experience_list(background_data):
        return build_selection_from_scores(
            background_data, heuristic_ratings(background_data), policy=policy
        )

    system_prompt = build_ratings_system_prompt(background_data)
    client = OpenAI(base_url=endpoint_url, api_key=api_key)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": job_description.strip()},
    ]

    last_error: AIResponseError | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
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
        if not choice or not choice.message or not choice.message.content:
            last_error = AIResponseError("Model returned an empty ratings response")
            continue

        try:
            ratings = parse_ratings_response(choice.message.content, background_data)
        except AIResponseError as exc:
            last_error = exc
            if attempt < 2:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was invalid: {exc} "
                            "Reply again with ONLY the JSON object matching the schema "
                            "(array lengths must match exactly), no markdown fences."
                        ),
                    }
                ]
            continue

        return build_selection_from_scores(background_data, ratings, policy=policy)

    print(f"[selection] falling back to heuristic ratings: {last_error}")
    return build_selection_from_scores(
        background_data, heuristic_ratings(background_data), policy=policy
    )
