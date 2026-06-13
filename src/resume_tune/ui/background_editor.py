"""Background editor helpers for the Streamlit UI."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import frontmatter
import streamlit as st
import yaml

from resume_tune.render.resume import (
    bootstrap_background_from_example,
    read_background_file,
    save_background,
)
from resume_tune.ui.common import example_background_path


DRAFT_KEY = "bg_draft"
RAW_KEY = "bg_raw_text"
LOADED_PATH_KEY = "bg_loaded_path"


def _empty_draft() -> dict[str, Any]:
    return {
        "header": {
            "name": "",
            "title": "",
            "email": "",
            "phone": "",
            "location": "",
            "links": [],
        },
        "experience": [
            {
                "company": "",
                "title": "",
                "location": "",
                "start": "",
                "end": "",
                "bullets": [""],
            }
        ],
        "education": [],
        "projects": [],
        "certifications": [],
        "skills_map": {},
    }


def _normalize_loaded_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    draft = _empty_draft()
    header = metadata.get("header") or {}
    draft["header"] = {
        "name": header.get("name", ""),
        "title": header.get("title", ""),
        "email": header.get("email", ""),
        "phone": header.get("phone", ""),
        "location": header.get("location", ""),
        "links": deepcopy(header.get("links") or []),
    }
    draft["experience"] = deepcopy(metadata.get("experience") or draft["experience"])
    draft["education"] = deepcopy(metadata.get("education") or [])
    draft["projects"] = deepcopy(metadata.get("projects") or [])
    draft["certifications"] = deepcopy(metadata.get("certifications") or [])
    draft["skills_map"] = deepcopy(metadata.get("skills_map") or {})
    return draft


def load_draft_from_disk(path) -> tuple[dict[str, Any], str]:
    metadata, body = read_background_file(path)
    return _normalize_loaded_metadata(metadata), body


def ensure_draft_loaded(path) -> None:
    path_str = str(path)
    if st.session_state.get(LOADED_PATH_KEY) == path_str and DRAFT_KEY in st.session_state:
        return
    metadata, body = load_draft_from_disk(path)
    st.session_state[DRAFT_KEY] = metadata
    st.session_state["bg_body"] = body
    st.session_state[RAW_KEY] = None
    st.session_state[LOADED_PATH_KEY] = path_str


def discard_draft(path) -> None:
    for key in (DRAFT_KEY, "bg_body", RAW_KEY, LOADED_PATH_KEY):
        st.session_state.pop(key, None)
    ensure_draft_loaded(path)


def _lines_to_bullets(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _bullets_to_lines(bullets: list[str]) -> str:
    return "\n".join(bullets)


def _skills_text(skills_map: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [(name, ", ".join(skills)) for name, skills in skills_map.items()]


def _skills_from_rows(rows: list[tuple[str, str]]) -> dict[str, list[str]]:
    skills_map: dict[str, list[str]] = {}
    for category, skills_text in rows:
        category = category.strip()
        if not category:
            continue
        skills = [part.strip() for part in skills_text.replace("\n", ",").split(",") if part.strip()]
        if skills:
            skills_map[category] = skills
    return skills_map


def _build_metadata_from_draft(draft: dict[str, Any]) -> dict[str, Any]:
    metadata = deepcopy(draft)
    for job in metadata.get("experience", []):
        if isinstance(job.get("bullets"), str):
            job["bullets"] = _lines_to_bullets(job["bullets"])
    for project in metadata.get("projects", []):
        if isinstance(project.get("bullets"), str):
            project["bullets"] = _lines_to_bullets(project["bullets"])
    if not metadata.get("education"):
        metadata.pop("education", None)
    if not metadata.get("projects"):
        metadata.pop("projects", None)
    if not metadata.get("certifications"):
        metadata.pop("certifications", None)
    if not metadata.get("skills_map"):
        metadata.pop("skills_map", None)
    header = metadata.get("header", {})
    links = header.get("links") or []
    header["links"] = [link for link in links if link.get("label") and link.get("url")]
    metadata["header"] = header
    return metadata


def save_structured(path, body: str) -> None:
    metadata = _build_metadata_from_draft(st.session_state[DRAFT_KEY])
    save_background(path, metadata, body)


def save_raw(path, raw_text: str) -> None:
    post = frontmatter.loads(raw_text)
    metadata = post.metadata if isinstance(post.metadata, dict) else {}
    body = post.content or ""
    save_background(path, metadata, body)


def render_structured_editor() -> None:
    draft = st.session_state[DRAFT_KEY]
    header = draft["header"]

    st.subheader("Header")
    col1, col2 = st.columns(2)
    header["name"] = col1.text_input("Name", value=header.get("name", ""), key="bg_header_name")
    header["title"] = col2.text_input("Title", value=header.get("title", ""), key="bg_header_title")
    col3, col4 = st.columns(2)
    header["email"] = col3.text_input("Email", value=header.get("email", ""), key="bg_header_email")
    header["phone"] = col4.text_input("Phone", value=header.get("phone", ""), key="bg_header_phone")
    header["location"] = st.text_input(
        "Location", value=header.get("location", ""), key="bg_header_location"
    )

    st.markdown("**Links**")
    links = header.setdefault("links", [])
    remove_link_idx: int | None = None
    for idx, link in enumerate(links):
        link_col1, link_col2, link_col3 = st.columns([2, 3, 1])
        link["label"] = link_col1.text_input(
            "Label", value=link.get("label", ""), key=f"bg_link_label_{idx}"
        )
        link["url"] = link_col2.text_input(
            "URL", value=link.get("url", ""), key=f"bg_link_url_{idx}"
        )
        if link_col3.button("Remove", key=f"bg_link_remove_{idx}"):
            remove_link_idx = idx
    if remove_link_idx is not None:
        links.pop(remove_link_idx)
        st.rerun()
    if st.button("Add link", key="bg_add_link"):
        links.append({"label": "", "url": ""})
        st.rerun()

    st.divider()
    st.subheader("Experience")
    experience = draft.setdefault("experience", [])
    remove_job_idx: int | None = None
    for idx, job in enumerate(experience):
        with st.expander(f"Role {idx + 1}: {job.get('title') or job.get('company') or 'New role'}", expanded=idx == 0):
            job["company"] = st.text_input("Company", value=job.get("company", ""), key=f"bg_job_company_{idx}")
            job["title"] = st.text_input("Title", value=job.get("title", ""), key=f"bg_job_title_{idx}")
            job["location"] = st.text_input(
                "Location", value=job.get("location", ""), key=f"bg_job_location_{idx}"
            )
            date_col1, date_col2 = st.columns(2)
            job["start"] = date_col1.text_input(
                "Start (YYYY-MM)", value=job.get("start", ""), key=f"bg_job_start_{idx}"
            )
            job["end"] = date_col2.text_input(
                "End (YYYY-MM or present)", value=job.get("end", ""), key=f"bg_job_end_{idx}"
            )
            bullets = job.get("bullets") or [""]
            job["bullets"] = _lines_to_bullets(
                st.text_area(
                    "Bullets (one per line)",
                    value=_bullets_to_lines(bullets if isinstance(bullets, list) else [str(bullets)]),
                    height=120,
                    key=f"bg_job_bullets_{idx}",
                )
            )
            if st.button("Remove role", key=f"bg_job_remove_{idx}"):
                remove_job_idx = idx
    if remove_job_idx is not None:
        experience.pop(remove_job_idx)
        st.rerun()
    if st.button("Add experience", key="bg_add_job"):
        experience.append(
            {
                "company": "",
                "title": "",
                "location": "",
                "start": "",
                "end": "",
                "bullets": [""],
            }
        )
        st.rerun()

    st.divider()
    st.subheader("Education")
    education = draft.setdefault("education", [])
    remove_edu_idx: int | None = None
    for idx, item in enumerate(education):
        with st.expander(f"Education {idx + 1}", expanded=False):
            item["institution"] = st.text_input(
                "Institution", value=item.get("institution", ""), key=f"bg_edu_inst_{idx}"
            )
            item["degree"] = st.text_input(
                "Degree", value=item.get("degree", ""), key=f"bg_edu_degree_{idx}"
            )
            item["graduation"] = st.text_input(
                "Graduation", value=item.get("graduation", ""), key=f"bg_edu_grad_{idx}"
            )
            item["location"] = st.text_input(
                "Location", value=item.get("location", ""), key=f"bg_edu_loc_{idx}"
            )
            if st.button("Remove education", key=f"bg_edu_remove_{idx}"):
                remove_edu_idx = idx
    if remove_edu_idx is not None:
        education.pop(remove_edu_idx)
        st.rerun()
    if st.button("Add education", key="bg_add_edu"):
        education.append({"institution": "", "degree": "", "graduation": "", "location": ""})
        st.rerun()

    st.divider()
    st.subheader("Projects")
    projects = draft.setdefault("projects", [])
    remove_project_idx: int | None = None
    for idx, project in enumerate(projects):
        with st.expander(f"Project {idx + 1}: {project.get('name') or 'New project'}", expanded=False):
            project["name"] = st.text_input(
                "Name", value=project.get("name", ""), key=f"bg_proj_name_{idx}"
            )
            project["url"] = st.text_input("URL", value=project.get("url", ""), key=f"bg_proj_url_{idx}")
            project["tech"] = st.text_input("Tech", value=project.get("tech", ""), key=f"bg_proj_tech_{idx}")
            bullets = project.get("bullets") or [""]
            project["bullets"] = _lines_to_bullets(
                st.text_area(
                    "Bullets (one per line)",
                    value=_bullets_to_lines(bullets if isinstance(bullets, list) else [str(bullets)]),
                    height=100,
                    key=f"bg_proj_bullets_{idx}",
                )
            )
            if st.button("Remove project", key=f"bg_proj_remove_{idx}"):
                remove_project_idx = idx
    if remove_project_idx is not None:
        projects.pop(remove_project_idx)
        st.rerun()
    if st.button("Add project", key="bg_add_project"):
        projects.append({"name": "", "url": "", "tech": "", "bullets": [""]})
        st.rerun()

    st.divider()
    st.subheader("Certifications")
    certifications = draft.setdefault("certifications", [])
    remove_cert_idx: int | None = None
    for idx, cert in enumerate(certifications):
        with st.expander(f"Certification {idx + 1}", expanded=False):
            cert["name"] = st.text_input("Name", value=cert.get("name", ""), key=f"bg_cert_name_{idx}")
            cert["issuer"] = st.text_input("Issuer", value=cert.get("issuer", ""), key=f"bg_cert_issuer_{idx}")
            cert["date"] = st.text_input("Date", value=cert.get("date", ""), key=f"bg_cert_date_{idx}")
            if st.button("Remove certification", key=f"bg_cert_remove_{idx}"):
                remove_cert_idx = idx
    if remove_cert_idx is not None:
        certifications.pop(remove_cert_idx)
        st.rerun()
    if st.button("Add certification", key="bg_add_cert"):
        certifications.append({"name": "", "issuer": "", "date": ""})
        st.rerun()

    st.divider()
    st.subheader("Skills map")
    skills_rows = _skills_text(draft.setdefault("skills_map", {}))
    remove_skill_idx: int | None = None
    updated_rows: list[tuple[str, str]] = []
    for idx, (category, skills_text) in enumerate(skills_rows):
        row_col1, row_col2, row_col3 = st.columns([2, 4, 1])
        category_val = row_col1.text_input(
            "Category", value=category, key=f"bg_skill_cat_{idx}"
        )
        skills_val = row_col2.text_input(
            "Skills (comma-separated)", value=skills_text, key=f"bg_skill_vals_{idx}"
        )
        updated_rows.append((category_val, skills_val))
        if row_col3.button("Remove", key=f"bg_skill_remove_{idx}"):
            remove_skill_idx = idx
    if remove_skill_idx is not None:
        updated_rows.pop(remove_skill_idx)
        draft["skills_map"] = _skills_from_rows(updated_rows)
        st.rerun()
    draft["skills_map"] = _skills_from_rows(updated_rows)
    if st.button("Add skill category", key="bg_add_skill_cat"):
        rows = _skills_text(draft["skills_map"])
        rows.append(("", ""))
        draft["skills_map"] = _skills_from_rows(rows)
        st.rerun()


def render_background_page(background_path) -> None:
    st.set_page_config(page_title="Resume data", layout="wide")
    st.title("Resume data")
    st.caption(
        "Edit the resume data in `background.md`. YAML frontmatter is rendered in exports; "
        "the career narrative is AI-only context."
    )

    example_path = example_background_path()
    if not background_path.exists():
        st.warning(f"`{background_path.name}` not found.")
        if example_path.exists() and st.button("Create from example"):
            bootstrap_background_from_example(background_path, example_path)
            discard_draft(background_path)
            st.success(f"Created `{background_path}` from example.")
            st.rerun()
        return

    ensure_draft_loaded(background_path)

    action_col1, action_col2 = st.columns(2)
    if action_col2.button("Discard changes", width="stretch"):
        discard_draft(background_path)
        st.info("Reloaded from disk.")
        st.rerun()

    tab_structured, tab_narrative, tab_raw = st.tabs(
        ["Resume sections", "Career narrative", "Raw file"]
    )

    with tab_structured:
        render_structured_editor()

    with tab_narrative:
        st.session_state["bg_body"] = st.text_area(
            "Career narrative (markdown, AI-only)",
            value=st.session_state.get("bg_body", ""),
            height=400,
            key="bg_body_editor",
        )

    with tab_raw:
        if st.session_state.get(RAW_KEY) is None:
            metadata = _build_metadata_from_draft(st.session_state[DRAFT_KEY])
            st.session_state[RAW_KEY] = frontmatter.dumps(
                frontmatter.Post(st.session_state.get("bg_body", ""), **metadata)
            )
        st.session_state[RAW_KEY] = st.text_area(
            "Full background.md",
            value=st.session_state[RAW_KEY],
            height=500,
            key="bg_raw_editor",
        )
        if st.button("Save raw file", type="primary", key="bg_save_raw"):
            try:
                save_raw(background_path, st.session_state.get(RAW_KEY, ""))
                discard_draft(background_path)
                st.success(f"Saved `{background_path}` from raw editor.")
            except (ValueError, yaml.YAMLError) as exc:
                st.error(str(exc))

    if action_col1.button("Save", type="primary", width="stretch"):
        try:
            save_structured(background_path, st.session_state.get("bg_body", ""))
            discard_draft(background_path)
            st.success(f"Saved `{background_path}` (backup: `{background_path.name}.bak`).")
        except (ValueError, yaml.YAMLError) as exc:
            st.error(str(exc))
