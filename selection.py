"""Job-aware selection of experience, project, and education content from background data."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

from ai import AIResponseError, strip_json_fences

DEFAULT_MAX_BULLETS_PER_ROLE = 3
DEFAULT_MAX_EXPERIENCE_ENTRIES = 4
DEFAULT_MAX_PROJECT_ENTRIES = 2
DEFAULT_MIN_PROJECT_ENTRIES = 1
DEFAULT_MIN_PROJECT_BULLETS = 1

SELECTION_KEYS = frozenset(
    {"experience_selections", "project_selections", "education_indices"}
)

SELECTION_SYSTEM_PROMPT_TEMPLATE = """You are a resume editor selecting the most job-relevant content from a candidate's background.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation. No trailing text.
- JSON schema:
{{
  "experience_selections": [
    {{"role_index": <int>, "bullet_indices": [<int>, ...]}},
    ...
  ],
  "project_selections": [
    {{"project_index": <int>, "bullet_indices": [<int>, ...]}},
    ...
  ],
  "education_indices": [<int>, ...]
}}
- Select up to {max_experience_entries} roles and up to {max_bullets_per_role} bullets per role.
- Select up to {max_project_entries} projects and up to {max_bullets_per_role} bullets per project.
- When the candidate has projects, ALWAYS include at least 1 project with at least 1 bullet — pick the single most job-relevant project and its best bullet when space is tight.
- Order roles, projects, bullet_indices, and education_indices by relevance to the job description (most relevant first).
- Use ONLY indices that appear in CANDIDATE CONTENT below. Do NOT invent or rewrite bullet text.
- Include a role only when at least one of its bullets is relevant to the job.
- Include an education entry only when relevant to the job; omit unrelated degrees (e.g. omit Chemical Engineering for pure software/web roles unless the posting mentions chemical/process/industrial engineering or oil & gas domain work).
- Always include the most relevant software engineering degree when applicable.
- Do not include any keys other than "experience_selections", "project_selections", and "education_indices".

CANDIDATE CONTENT (indexed):
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
    max_experience_entries: int = DEFAULT_MAX_EXPERIENCE_ENTRIES,
    max_bullets_per_role: int = DEFAULT_MAX_BULLETS_PER_ROLE,
    max_project_entries: int = DEFAULT_MAX_PROJECT_ENTRIES,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> dict[str, Any]:
    """Select all content capped by configured limits (most recent roles first)."""
    experience_selections: list[dict[str, Any]] = []
    for role_index, job in enumerate(_experience_list(background_data)[:max_experience_entries]):
        bullet_indices = list(range(min(len(job.get("bullets", [])), max_bullets_per_role)))
        if bullet_indices:
            experience_selections.append(
                {"role_index": role_index, "bullet_indices": bullet_indices}
            )

    project_selections: list[dict[str, Any]] = []
    for project_index, project in enumerate(_projects_list(background_data)[:max_project_entries]):
        bullet_indices = list(range(min(len(project.get("bullets", [])), max_bullets_per_role)))
        if bullet_indices:
            project_selections.append(
                {"project_index": project_index, "bullet_indices": bullet_indices}
            )

    selection = {
        "experience_selections": experience_selections,
        "project_selections": project_selections,
        "education_indices": _all_education_indices(background_data),
    }
    return enforce_project_floor(
        background_data,
        selection,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
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


def _validate_index_list(
    indices: Any,
    *,
    label: str,
    max_count: int,
    upper_bound: int,
) -> list[int]:
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"{label} bullet_indices must be a non-empty list")
    if len(indices) > max_count:
        raise ValueError(f"{label} has too many bullet_indices (max {max_count})")
    validated: list[int] = []
    seen: set[int] = set()
    for raw in indices:
        if not isinstance(raw, int):
            raise ValueError(f"{label} bullet_indices must contain integers")
        if raw < 0 or raw >= upper_bound:
            raise ValueError(f"{label} bullet_index {raw} out of range (0-{upper_bound - 1})")
        if raw not in seen:
            seen.add(raw)
            validated.append(raw)
    return validated


def _validate_education_indices(
    indices: Any,
    *,
    education_count: int,
) -> list[int]:
    if not isinstance(indices, list):
        raise ValueError("education_indices must be a list")
    if education_count == 0:
        if indices:
            raise ValueError("education_indices must be empty when candidate has no education")
        return []

    if not indices:
        raise ValueError("education_indices must be a non-empty list when education exists")

    validated: list[int] = []
    seen: set[int] = set()
    for raw in indices:
        if not isinstance(raw, int):
            raise ValueError("education_indices must contain integers")
        if raw < 0 or raw >= education_count:
            raise ValueError(
                f"education_index {raw} out of range (0-{education_count - 1})"
            )
        if raw not in seen:
            seen.add(raw)
            validated.append(raw)
    return validated


def validate_selection(
    selection: dict[str, Any],
    background_data: dict[str, Any],
    *,
    max_experience_entries: int = DEFAULT_MAX_EXPERIENCE_ENTRIES,
    max_bullets_per_role: int = DEFAULT_MAX_BULLETS_PER_ROLE,
    max_project_entries: int = DEFAULT_MAX_PROJECT_ENTRIES,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> dict[str, Any]:
    """Validate selection indices against background; return normalized selection."""
    if not isinstance(selection, dict):
        raise ValueError("Selection must be a mapping")

    experience = _experience_list(background_data)
    projects = _projects_list(background_data)
    education = _education_list(background_data)

    raw_experience = selection.get("experience_selections", [])
    raw_projects = selection.get("project_selections", [])

    if not isinstance(raw_experience, list):
        raise ValueError("experience_selections must be a list")
    if not isinstance(raw_projects, list):
        raise ValueError("project_selections must be a list")

    raw_experience = raw_experience[:max_experience_entries]
    raw_projects = raw_projects[:max_project_entries]

    experience_selections: list[dict[str, Any]] = []
    seen_roles: set[int] = set()
    for i, entry in enumerate(raw_experience):
        if not isinstance(entry, dict):
            raise ValueError(f"experience_selections[{i}] must be a mapping")
        role_index = entry.get("role_index")
        if not isinstance(role_index, int):
            raise ValueError(f"experience_selections[{i}].role_index must be an integer")
        if role_index in seen_roles:
            raise ValueError(f"Duplicate role_index {role_index}")
        if role_index < 0 or role_index >= len(experience):
            raise ValueError(f"role_index {role_index} out of range (0-{len(experience) - 1})")
        seen_roles.add(role_index)
        bullets = experience[role_index].get("bullets", [])
        bullet_indices = _validate_index_list(
            entry.get("bullet_indices"),
            label=f"experience_selections[{i}]",
            max_count=max_bullets_per_role,
            upper_bound=len(bullets),
        )
        experience_selections.append({"role_index": role_index, "bullet_indices": bullet_indices})

    project_selections: list[dict[str, Any]] = []
    seen_projects: set[int] = set()
    for i, entry in enumerate(raw_projects):
        if not isinstance(entry, dict):
            raise ValueError(f"project_selections[{i}] must be a mapping")
        project_index = entry.get("project_index")
        if not isinstance(project_index, int):
            raise ValueError(f"project_selections[{i}].project_index must be an integer")
        if project_index in seen_projects:
            raise ValueError(f"Duplicate project_index {project_index}")
        if project_index < 0 or project_index >= len(projects):
            raise ValueError(
                f"project_index {project_index} out of range (0-{len(projects) - 1})"
            )
        seen_projects.add(project_index)
        bullets = projects[project_index].get("bullets", [])
        bullet_indices = _validate_index_list(
            entry.get("bullet_indices"),
            label=f"project_selections[{i}]",
            max_count=max_bullets_per_role,
            upper_bound=len(bullets),
        )
        project_selections.append(
            {"project_index": project_index, "bullet_indices": bullet_indices}
        )

    education_indices = _validate_education_indices(
        selection.get("education_indices", []),
        education_count=len(education),
    )

    normalized = {
        "experience_selections": experience_selections,
        "project_selections": project_selections,
        "education_indices": education_indices,
    }
    return enforce_project_floor(
        background_data,
        normalized,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
    )


def parse_selection_response(raw: str, background_data: dict[str, Any], **limits: Any) -> dict[str, Any]:
    """Parse and validate model JSON selection response."""
    cleaned = strip_json_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        snippet = cleaned[:500] + ("..." if len(cleaned) > 500 else "")
        raise AIResponseError(
            f"Model returned invalid JSON for selection: {exc}. Response snippet: {snippet!r}"
        ) from exc

    if not isinstance(data, dict):
        raise AIResponseError("Selection response must be a JSON object")
    if set(data.keys()) != SELECTION_KEYS:
        raise AIResponseError(
            "Selection response must contain exactly 'experience_selections', "
            f"'project_selections', and 'education_indices', got keys: {sorted(data.keys())}"
        )

    try:
        return validate_selection(data, background_data, **limits)
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

    selected_experience: list[dict[str, Any]] = []
    for entry in selection.get("experience_selections", []):
        role_index = entry["role_index"]
        job = deepcopy(experience[role_index])
        source_bullets = experience[role_index].get("bullets", [])
        job["bullets"] = [source_bullets[i] for i in entry["bullet_indices"]]
        selected_experience.append(job)

    selected_projects: list[dict[str, Any]] = []
    for entry in selection.get("project_selections", []):
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


def build_selection_system_prompt(
    background_data: dict[str, Any],
    *,
    max_experience_entries: int = DEFAULT_MAX_EXPERIENCE_ENTRIES,
    max_bullets_per_role: int = DEFAULT_MAX_BULLETS_PER_ROLE,
    max_project_entries: int = DEFAULT_MAX_PROJECT_ENTRIES,
) -> str:
    return SELECTION_SYSTEM_PROMPT_TEMPLATE.format(
        max_experience_entries=max_experience_entries,
        max_bullets_per_role=max_bullets_per_role,
        max_project_entries=max_project_entries,
        indexed_content=format_indexed_content(background_data),
    )


def generate_content_selection(
    job_description: str,
    background_data: dict[str, Any],
    *,
    endpoint_url: str,
    model_name: str,
    api_key: str = "ollama",
    max_experience_entries: int = DEFAULT_MAX_EXPERIENCE_ENTRIES,
    max_bullets_per_role: int = DEFAULT_MAX_BULLETS_PER_ROLE,
    max_project_entries: int = DEFAULT_MAX_PROJECT_ENTRIES,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
) -> dict[str, Any]:
    """
    Call LLM to select job-relevant experience/project bullets by index.

    Falls back to default_selection when the model fails after retries.
    """
    if not job_description.strip():
        raise ValueError("Job description cannot be empty")

    limits = {
        "max_experience_entries": max_experience_entries,
        "max_bullets_per_role": max_bullets_per_role,
        "max_project_entries": max_project_entries,
        "min_project_entries": min_project_entries,
        "min_project_bullets": min_project_bullets,
    }
    fallback = default_selection(background_data, **limits)

    if not _experience_list(background_data):
        return fallback

    system_prompt = build_selection_system_prompt(
        background_data,
        max_experience_entries=max_experience_entries,
        max_bullets_per_role=max_bullets_per_role,
        max_project_entries=max_project_entries,
    )
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
        if not choice or not choice.message or not choice.message.content:
            last_error = AIResponseError("Model returned an empty selection response")
            continue

        try:
            return parse_selection_response(choice.message.content, background_data, **limits)
        except AIResponseError as exc:
            last_error = exc
            if attempt < 2:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was invalid. Reply again with ONLY the JSON "
                            "object using valid indices from CANDIDATE CONTENT, no markdown fences."
                        ),
                    }
                ]

    print(f"[selection] falling back to default selection: {last_error}")
    return fallback
