"""Streamlit dashboard for viewing and editing logged job applications."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from resume_tune.tracker.paths import JD_SNAPSHOT_FILENAME, RESUME_BASENAME, resolve_stored_path
from resume_tune.tracker.tracker import HEADERS, load_applications, save_applications

TERMINAL_STATUSES = frozenset(
    {
        "rejected",
        "offer accepted",
        "offer declined",
        "withdrawn",
        "hired",
        "declined",
        "accepted",
    }
)

EDITABLE_COLUMNS = frozenset(
    {
        "Company",
        "Role / Job Title",
        "Location",
        "Status",
        "Job URL",
        "Follow-up Date",
        "Notes",
    }
)

READONLY_COLUMNS = frozenset(header for header in HEADERS if header not in EDITABLE_COLUMNS)

DATE_COLUMNS = ("Date Applied", "Follow-up Date")


def _parse_iso_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    if getattr(value, "__class__", None).__name__ in {"NaTType", "NAType"}:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nat":
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _relative_path(path_value: Any, applications_dir: Path) -> str:
    text = str(path_value or "").strip()
    if not text:
        return ""
    path = Path(text)
    try:
        return str(path.resolve().relative_to(applications_dir.resolve()))
    except ValueError:
        return text


def _row_for_editor(row: dict[str, Any], applications_dir: Path) -> dict[str, Any]:
    editor_row = dict(row)
    for field in DATE_COLUMNS:
        editor_row[field] = _parse_iso_date(editor_row.get(field))
    editor_row["ATS Score at Apply"] = _parse_number(editor_row.get("ATS Score at Apply"))
    editor_row["Resume File"] = _relative_path(editor_row.get("Resume File"), applications_dir)
    editor_row["JD Snapshot"] = _relative_path(editor_row.get("JD Snapshot"), applications_dir)
    return editor_row


def _serialize_tracker_row(row: dict[str, Any], applications_dir: Path) -> dict[str, Any]:
    clean = {header: row.get(header, "") for header in HEADERS}
    for field in DATE_COLUMNS:
        parsed = _parse_iso_date(clean.get(field))
        clean[field] = parsed.isoformat() if parsed else ""
    ats = _parse_number(clean.get("ATS Score at Apply"))
    clean["ATS Score at Apply"] = ats if ats is not None else ""

    for field in ("Resume File", "JD Snapshot"):
        rel = str(clean.get(field) or "").strip()
        if rel and not Path(rel).is_absolute():
            clean[field] = str((applications_dir / rel).resolve())
        elif rel:
            clean[field] = str(Path(rel).resolve())

    return clean


def _matches_search(row: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    needle = query.lower()
    haystack = " ".join(
        str(row.get(field, "") or "") for field in ("Company", "Role / Job Title", "Notes")
    ).lower()
    return needle in haystack


def _matches_status(row: dict[str, Any], status_filter: str) -> bool:
    if status_filter == "All":
        return True
    return str(row.get("Status") or "").strip().lower() == status_filter.lower()


def _matches_date_applied(row: dict[str, Any], start: date | None, end: date | None) -> bool:
    applied = _parse_iso_date(row.get("Date Applied"))
    if start and (applied is None or applied < start):
        return False
    if end and (applied is None or applied > end):
        return False
    return True


def _matches_follow_up_overdue(row: dict[str, Any], enabled: bool) -> bool:
    if not enabled:
        return True
    follow_up = _parse_iso_date(row.get("Follow-up Date"))
    if follow_up is None:
        return False
    status = str(row.get("Status") or "").strip().lower()
    if status in TERMINAL_STATUSES:
        return False
    return follow_up <= date.today()


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    search: str,
    status_filter: str,
    applied_start: date | None,
    applied_end: date | None,
    follow_up_overdue: bool,
) -> list[tuple[int, dict[str, Any]]]:
    filtered: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if not _matches_search(row, search):
            continue
        if not _matches_status(row, status_filter):
            continue
        if not _matches_date_applied(row, applied_start, applied_end):
            continue
        if not _matches_follow_up_overdue(row, follow_up_overdue):
            continue
        filtered.append((index, row))
    return filtered


def _column_config() -> dict[str, Any]:
    config: dict[str, Any] = {}
    for header in READONLY_COLUMNS:
        if header == "ATS Score at Apply":
            config[header] = st.column_config.NumberColumn(header, disabled=True)
        elif header == "Date Applied":
            config[header] = st.column_config.DateColumn(
                header,
                format="YYYY-MM-DD",
                disabled=True,
            )
        else:
            config[header] = st.column_config.TextColumn(header, disabled=True)
    for header in EDITABLE_COLUMNS:
        if header == "Follow-up Date":
            config[header] = st.column_config.DateColumn(
                header,
                format="YYYY-MM-DD",
            )
        elif header == "Job URL":
            config[header] = st.column_config.LinkColumn("Job URL", display_text="Open")
        elif header == "Notes":
            config[header] = st.column_config.TextColumn(header, width="large")
        else:
            config[header] = st.column_config.TextColumn(header)
    return config


def _editor_rows(edited: Any) -> list[dict[str, Any]]:
    to_dict = getattr(edited, "to_dict", None)
    if callable(to_dict):
        return to_dict("records")
    return list(edited)


def _normalize_edited_rows(
    edited_rows: list[dict[str, Any]], applications_dir: Path
) -> list[dict[str, Any]]:
    return [_serialize_tracker_row(row, applications_dir) for row in edited_rows]


def _row_label(row: dict[str, Any], position: int) -> str:
    company = str(row.get("Company") or "Unknown company").strip()
    role = str(row.get("Role / Job Title") or "Unknown role").strip()
    applied = _parse_iso_date(row.get("Date Applied"))
    applied_text = applied.isoformat() if applied else ""
    suffix = f" ({applied_text})" if applied_text else ""
    return f"{position + 1}. {company} — {role}{suffix}"


def _render_application_preview(row: dict[str, Any], applications_dir: Path) -> None:
    job_url = str(row.get("Job URL") or "").strip()
    if job_url:
        st.link_button("Open job URL", job_url)

    resume_tab, jd_tab = st.tabs(["Resume", "Job description"])

    with resume_tab:
        resume_path = resolve_stored_path(row.get("Resume File"), applications_dir)
        if not str(resume_path):
            st.info("No resume file recorded.")
        elif resume_path.is_file():
            resume_bytes = resume_path.read_bytes()
            mime = (
                "application/pdf"
                if resume_path.suffix.lower() == ".pdf"
                else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            st.download_button(
                "Download resume",
                data=resume_bytes,
                file_name=resume_path.name,
                mime=mime,
                width="stretch",
            )
            pdf_preview_path = (
                resume_path
                if resume_path.suffix.lower() == ".pdf"
                else resume_path.parent / f"{RESUME_BASENAME}.pdf"
            )
            if pdf_preview_path.is_file():
                st.pdf(pdf_preview_path.read_bytes(), height=600)
            elif resume_path.suffix.lower() != ".pdf":
                st.caption("Inline preview requires PDF; download DOCX above.")
        else:
            st.warning("Resume file is missing on disk.")

    with jd_tab:
        jd_path = resolve_stored_path(row.get("JD Snapshot"), applications_dir)
        if not jd_path.is_file():
            resume_path = resolve_stored_path(row.get("Resume File"), applications_dir)
            if resume_path.parent.is_dir():
                fallback = resume_path.parent / JD_SNAPSHOT_FILENAME
                if fallback.is_file():
                    jd_path = fallback

        if jd_path.is_file():
            jd_text = jd_path.read_text(encoding="utf-8")
            st.download_button(
                "Download job description",
                data=jd_text,
                file_name=jd_path.name,
                mime="text/plain",
                width="stretch",
            )
            st.text_area("Job description", value=jd_text, height=320, disabled=True)
        else:
            st.info("No job description saved for this application.")


def render_applications_dashboard(applications_dir: Path, tracker_path: Path) -> None:
    st.title("Applications")
    st.caption(f"Folder: `{applications_dir}` · Tracker: `{tracker_path.name}`")

    try:
        rows = load_applications(tracker_path)
    except OSError as exc:
        st.error(f"Could not load applications: {exc}")
        return

    if not rows:
        st.info(
            "No applications logged yet. Save a resume on **Tailor resume** and use "
            "**Log this application** to add your first row."
        )
        return

    statuses = sorted(
        {str(row.get("Status") or "").strip() for row in rows if str(row.get("Status") or "").strip()},
        key=str.lower,
    )

    with st.sidebar:
        st.subheader("Filters")
        search = st.text_input("Search", placeholder="Company, role, or notes")
        status_filter = st.selectbox("Status", ["All", *statuses])
        use_applied_start = st.checkbox("Filter by date applied (from)")
        applied_start = (
            st.date_input("Applied on or after", value=date.today(), key="apps_applied_start")
            if use_applied_start
            else None
        )
        use_applied_end = st.checkbox("Filter by date applied (to)")
        applied_end = (
            st.date_input("Applied on or before", value=date.today(), key="apps_applied_end")
            if use_applied_end
            else None
        )
        follow_up_overdue = st.checkbox("Follow-up overdue only")

    filtered = _filter_rows(
        rows,
        search=search.strip(),
        status_filter=status_filter,
        applied_start=applied_start if use_applied_start else None,
        applied_end=applied_end if use_applied_end else None,
        follow_up_overdue=follow_up_overdue,
    )

    st.caption(f"Showing **{len(filtered)}** of **{len(rows)}** applications")

    if not filtered:
        st.warning("No applications match the current filters.")
        return

    display_rows = [_row_for_editor(row, applications_dir) for _, row in filtered]
    edited = st.data_editor(
        display_rows,
        column_config=_column_config(),
        width="stretch",
        num_rows="fixed",
        hide_index=True,
        key="applications_data_editor",
    )

    if st.button("Save changes", type="primary"):
        edited_rows = _normalize_edited_rows(_editor_rows(edited), applications_dir)
        updated = [dict(row) for row in rows]
        for (orig_index, _), new_row in zip(filtered, edited_rows, strict=True):
            updated[orig_index] = new_row
        try:
            save_applications(tracker_path, updated)
            st.success("Saved application updates.")
            st.rerun()
        except OSError as exc:
            st.error(f"Could not save applications: {exc}")

    edited_rows = _editor_rows(edited)
    st.divider()
    option_labels = [_row_label(row, position) for position, row in enumerate(edited_rows)]
    selected_label = st.selectbox("Application", options=option_labels)
    selected_position = option_labels.index(selected_label)
    _render_application_preview(edited_rows[selected_position], applications_dir)
