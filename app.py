"""Streamlit UI for local resume tailoring."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import yaml

from ai import (
    AIResponseError,
    DEFAULT_AI_OUTPUT_MAX_CHARS,
    EMPTY_AI_OUTPUT,
    build_evidence_text,
    filter_skill_categories,
    finalize_manual_skills,
    format_skill_categories,
    generate_tailored_content,
    parse_skill_categories,
    read_background_text,
    revise_tailored_content,
)
from skills_map import load_skills_map
from ats import analyze_ats_compatibility
from resume import (
    build_resume_artifacts,
    libreoffice_available,
    load_background,
    resume_filename,
    save_resume_to_disk,
    static_preview_sections,
)
from selection import (
    DEFAULT_MIN_PROJECT_BULLETS,
    DEFAULT_MIN_PROJECT_ENTRIES,
    default_selection,
    full_selection,
    generate_content_selection,
    static_content_stats,
)
from settings import ROOT, load_settings


@st.cache_data
def load_config() -> dict:
    return load_settings()


def model_options(config: dict) -> list[str]:
    models_section = config.get("models", {})
    ollama = models_section.get("ollama", [])
    lemonade = models_section.get("lemonade", [])
    combined = list(dict.fromkeys([*ollama, *lemonade, config.get("model_name", "")]))
    return [m for m in combined if m]


def rebuild_artifacts(
    background_data: dict,
    ai_output: dict,
    *,
    max_pages: int,
    max_chars: int,
    sections: list[str],
    content_selection: dict | None = None,
    max_certifications: int | None = None,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
    auto_fill_page_budget: bool = True,
    overflow_warning_min_composite: float = 75.0,
) -> dict:
    """Build DOCX, HTML preview, and optional PDF from current AI output."""
    artifacts = build_resume_artifacts(
        background_data,
        ai_output,
        max_pages=max_pages,
        max_chars=max_chars,
        sections=sections,
        content_selection=content_selection,
        max_certifications=max_certifications,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
        auto_fill_page_budget=auto_fill_page_budget,
        overflow_warning_min_composite=overflow_warning_min_composite,
    )
    candidate_name = background_data["header"]["name"]
    return {
        **artifacts,
        "filename_docx": resume_filename(candidate_name, "docx"),
        "filename_pdf": resume_filename(candidate_name, "pdf"),
    }


def apply_artifacts_to_result(
    result: dict,
    background_data: dict,
    ai_output: dict,
    skill_warnings: list[str] | None = None,
    skill_packer_info: dict | None = None,
    *,
    max_pages: int,
    max_chars: int,
    sections: list[str],
    content_selection: dict | None = None,
    max_certifications: int | None = None,
    min_project_entries: int = DEFAULT_MIN_PROJECT_ENTRIES,
    min_project_bullets: int = DEFAULT_MIN_PROJECT_BULLETS,
    auto_fill_page_budget: bool = True,
    overflow_warning_min_composite: float = 75.0,
) -> dict:
    """Update session result with new AI content and rebuilt preview artifacts."""
    artifacts = rebuild_artifacts(
        background_data,
        ai_output,
        max_pages=max_pages,
        max_chars=max_chars,
        sections=sections,
        content_selection=content_selection,
        max_certifications=max_certifications,
        min_project_entries=min_project_entries,
        min_project_bullets=min_project_bullets,
        auto_fill_page_budget=auto_fill_page_budget,
        overflow_warning_min_composite=overflow_warning_min_composite,
    )
    fitted = artifacts["ai_output"]
    result["summary"] = fitted["summary"]
    result["skill_categories"] = fitted["skill_categories"]
    result["content_selection"] = artifacts["content_selection"]
    result["diagnostics"] = artifacts["diagnostics"]
    result.pop("ats", None)
    if skill_warnings is not None:
        result["skill_warnings"] = skill_warnings
    if skill_packer_info is not None:
        result["skill_packer_info"] = skill_packer_info
    result["docx_bytes"] = artifacts["docx_bytes"]
    result["html"] = artifacts["html"]
    result["pdf_bytes"] = artifacts["pdf_bytes"]
    result["page_count"] = artifacts["page_count"]
    result["filename_docx"] = artifacts["filename_docx"]
    result["filename_pdf"] = artifacts["filename_pdf"]
    result["saved_paths"] = {"docx": None, "pdf": None}
    return result


def artifact_build_kwargs(result: dict | None = None) -> dict:
    """Common kwargs for rebuild/apply artifact helpers."""
    kwargs: dict = {
        "max_pages": max_pages,
        "max_chars": max_chars,
        "sections": resume_sections,
        "max_certifications": max_certifications,
        "min_project_entries": min_project_entries,
        "min_project_bullets": min_project_bullets,
        "auto_fill_page_budget": auto_fill_page_budget,
        "overflow_warning_min_composite": overflow_warning_min_composite,
    }
    if result is not None:
        kwargs["content_selection"] = result.get("content_selection")
    return kwargs


def skills_layout_kwargs() -> dict:
    return {
        "max_skill_categories": max_skill_categories,
        "max_skills_per_category": max_skills_per_category,
        "max_chars_per_skill_line": max_chars_per_skill_line,
    }


def preview_artifact_build_kwargs() -> dict:
    """Kwargs for background preview (static sections only, full content selection)."""
    return {
        "max_pages": max_pages,
        "max_chars": max_chars,
        "sections": static_preview_sections(resume_sections),
        "max_certifications": max_certifications,
        "min_project_entries": min_project_entries,
        "min_project_bullets": min_project_bullets,
        "auto_fill_page_budget": auto_fill_page_budget,
        "overflow_warning_min_composite": overflow_warning_min_composite,
    }


def build_background_preview(background_data: dict) -> dict:
    """Build resume preview from background.md without LLM calls."""
    preview_sections = static_preview_sections(resume_sections)
    content_selection = full_selection(background_data)
    result = apply_artifacts_to_result(
        {},
        background_data,
        dict(EMPTY_AI_OUTPUT),
        content_selection=content_selection,
        **preview_artifact_build_kwargs(),
    )
    result["preview_mode"] = True
    result["revision_history"] = []
    result["sections"] = preview_sections
    return result


def api_configured() -> bool:
    return bool(endpoint_url)


def render_page_indicator(page_count: int | None, max_pages: int) -> None:
    if page_count is None:
        st.caption(
            "Page count unavailable. Install LibreOffice for accurate PDF preview and page fitting."
        )
        return
    if page_count <= max_pages:
        st.success(f"Resume: {page_count} page{'s' if page_count != 1 else ''} (target: {max_pages})")
    else:
        st.warning(
            f"Resume: {page_count} pages — content may need shortening (target: {max_pages} pages)."
        )


def render_page_fit_diagnostic(
    diagnostics: dict | None,
    background_data: dict,
    *,
    max_chars: int,
    include_summary: bool,
    include_skills: bool,
    enable_job_aware_selection: bool,
) -> None:
    """Show page-fit stats and trim-stalled guidance in the sidebar."""
    if diagnostics:
        stats = diagnostics
    else:
        stats = {
            **static_content_stats(background_data),
            "page_count": None,
            "max_pages": None,
            "ai_char_count": 0,
            "ai_char_budget": max_chars,
            "trim_stalled": False,
        }

    with st.expander(
        "Page fit details",
        expanded=bool(stats.get("trim_stalled") or stats.get("overflow_warning")),
    ):
        page_count = stats.get("page_count")
        max_pages_val = stats.get("max_pages")
        if page_count is not None and max_pages_val is not None:
            st.caption(f"Pages: **{page_count}** / target **{max_pages_val}**")
        else:
            st.caption("Pages: unavailable (LibreOffice required for PDF measurement)")

        if include_summary or include_skills:
            ai_chars = stats.get("ai_char_count", 0)
            st.caption(f"AI content: **{ai_chars}** / **{max_chars}** characters")

        st.caption(
            f"Experience: **{stats.get('experience_entries', 0)}** roles, "
            f"**{stats.get('experience_bullets', 0)}** bullets"
        )
        st.caption(
            f"Projects: **{stats.get('project_entries', 0)}** entries, "
            f"**{stats.get('project_bullets', 0)}** bullets"
        )

        expand_log = stats.get("expand_log") or []
        if expand_log:
            st.caption(f"Auto-filled **{len(expand_log)}** item(s) to use available page space.")

        if stats.get("overflow_warning") and stats.get("overflow_message"):
            st.info(stats["overflow_message"])

        if stats.get("trim_stalled"):
            st.warning(
                "Still over the page budget after trimming experience and project bullets. "
                "Skills were not modified. Shorten bullets in background.md or omit sections "
                "in config.toml."
            )


def render_skill_packer_diagnostic(packer_info: dict | None) -> None:
    """Show skills guardrail results in the sidebar."""
    if not packer_info:
        return
    removed = packer_info.get("removed_skills") or []
    deduped = packer_info.get("deduped_skills") or []
    added = packer_info.get("added_skills") or []
    removed_irrelevant = packer_info.get("removed_irrelevant") or []
    dropped_categories = packer_info.get("dropped_categories") or []
    lines = packer_info.get("line_utilization") or []
    if (
        not removed
        and not deduped
        and not added
        and not removed_irrelevant
        and not dropped_categories
        and not lines
    ):
        return
    with st.expander(
        "Skills guardrails",
        expanded=bool(removed or deduped or added or removed_irrelevant or dropped_categories),
    ):
        if removed:
            st.caption("Removed (not in skills_map): **" + ", ".join(removed) + "**")
        if removed_irrelevant:
            st.caption("Removed (not job-relevant): **" + ", ".join(removed_irrelevant) + "**")
        if dropped_categories:
            st.caption("Dropped categories: **" + ", ".join(dropped_categories) + "**")
        if deduped:
            st.caption("Deduped: **" + ", ".join(deduped) + "**")
        if added:
            st.caption("Relevance top-up: **" + ", ".join(added) + "**")
        for line in lines:
            name = line.get("name", "") or "(unnamed)"
            chars = line.get("chars", 0)
            max_chars = line.get("max", 0)
            st.caption(f"{name}: **{chars}** / **{max_chars}** characters")


def run_ats_check(
    background_data: dict,
    result: dict,
    job_description: str,
    *,
    sections: list[str],
    max_certifications: int | None,
) -> dict | None:
    """Run on-demand ATS analysis against the current resume preview."""
    jd = job_description.strip()
    if not jd:
        return None

    ai_output = {
        "summary": result.get("summary", ""),
        "skill_categories": result.get("skill_categories", []),
    }
    report = analyze_ats_compatibility(
        job_description=jd,
        background_data=background_data,
        ai_output=ai_output,
        pdf_bytes=result.get("pdf_bytes"),
        sections=sections,
        content_selection=result.get("content_selection"),
        max_certifications=max_certifications,
    )
    return report.to_dict() if report is not None else None


def render_ats_section(
    result: dict,
    background_data: dict,
    job_description: str,
    *,
    sections: list[str],
    max_certifications: int | None,
    enable_ats_compat: bool,
) -> None:
    """On-demand ATS check button and results panel."""
    if not enable_ats_compat:
        return

    jd = (result.get("job_description") or job_description).strip()
    st.divider()
    st.caption("ATS compatibility (optional)")
    run_disabled = not jd
    if st.button(
        "Run ATS check",
        use_container_width=True,
        disabled=run_disabled,
        help="Paste a job description above, then run a deterministic keyword and parse check.",
    ):
        with st.spinner("Running ATS check..."):
            ats = run_ats_check(
                background_data,
                result,
                jd,
                sections=sections,
                max_certifications=max_certifications,
            )
            if ats is None:
                st.warning("Paste a job description to run an ATS check.")
            else:
                result["ats"] = ats
                result["job_description"] = jd
                st.session_state.result = result
                st.rerun()

    if run_disabled:
        st.caption("Paste a job description above to enable ATS check.")
    elif not result.get("ats"):
        st.caption("Optional — run when you're happy with the preview. Re-run after revisions.")

    ats = result.get("ats")
    if not ats:
        return

    pct = ats.get("keyword_match_pct", 0)
    expand = pct < 50 or (
        ats.get("pdf_fidelity") and ats["pdf_fidelity"].get("fidelity_score", 100) < 80
    )

    with st.expander(f"ATS compatibility — {pct:.0f}% keyword match", expanded=expand):
        jd_keywords = ats.get("jd_keywords") or []
        matched = ats.get("matched_keywords") or []
        missing = ats.get("missing_keywords") or []

        if not jd_keywords:
            st.caption("No tech keywords detected in the job description.")
        else:
            st.caption(f"Keywords: **{len(matched)}** / **{len(jd_keywords)}** matched")
            if missing:
                st.caption(f"Missing: {', '.join(missing)}")

        sections_found = ats.get("sections_found") or []
        sections_missing = ats.get("sections_missing") or []
        if sections_found:
            st.caption(f"Sections found: {', '.join(sections_found)}")
        if sections_missing:
            st.caption(f"Sections missing: {', '.join(sections_missing)}")

        contact = ats.get("contact") or {}
        contact_parts = []
        for label, key in (
            ("email", "email_found"),
            ("phone", "phone_found"),
            ("LinkedIn", "linkedin_found"),
            ("GitHub", "github_found"),
        ):
            mark = "✓" if contact.get(key) else "✗"
            contact_parts.append(f"{label} {mark}")
        st.caption("Contact: " + "  ".join(contact_parts))

        source = ats.get("resume_text_source", "flattened")
        fidelity = ats.get("pdf_fidelity")
        if fidelity:
            score = fidelity.get("fidelity_score", 0)
            st.caption(f"PDF fidelity: **{score:.0f}%** (keyword match scored on PDF text)")
            checks = []
            if fidelity.get("name_in_pdf") is not None:
                checks.append(f"Name {'✓' if fidelity['name_in_pdf'] else '✗'}")
            if fidelity.get("email_in_pdf") is not None:
                checks.append(f"Email {'✓' if fidelity['email_in_pdf'] else '✗'}")
            for section_id, found in (fidelity.get("sections_in_pdf") or {}).items():
                checks.append(f"{section_id} {'✓' if found else '✗'}")
            if checks:
                st.caption("PDF checks: " + "  ".join(checks))
        elif source == "flattened":
            st.caption(
                "PDF fidelity unavailable — install LibreOffice for PDF text extraction checks."
            )


def render_save_section(result: dict, output_dir: Path) -> None:
    st.subheader("Save")
    save_col_docx, save_col_pdf = st.columns(2)

    docx_bytes = result.get("docx_bytes")
    pdf_bytes = result.get("pdf_bytes")
    filename_docx = result.get("filename_docx") or "resume.docx"
    filename_pdf = result.get("filename_pdf") or "resume.pdf"

    with save_col_docx:
        if docx_bytes:
            if st.button("Save DOCX to disk", key="save_docx_disk", use_container_width=True):
                try:
                    saved_path = save_resume_to_disk(docx_bytes, output_dir, filename_docx)
                    result["saved_paths"]["docx"] = str(saved_path)
                    st.success(f"Saved to `{saved_path}`")
                except OSError as exc:
                    st.error(f"Could not save DOCX: {exc}")

            st.download_button(
                label="Download DOCX",
                data=docx_bytes,
                file_name=filename_docx,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True,
            )
            saved_docx = result.get("saved_paths", {}).get("docx")
            if saved_docx:
                st.caption(f"On disk: `{saved_docx}`")

    with save_col_pdf:
        if pdf_bytes:
            if st.button("Save PDF to disk", key="save_pdf_disk", use_container_width=True):
                try:
                    saved_path = save_resume_to_disk(pdf_bytes, output_dir, filename_pdf)
                    result["saved_paths"]["pdf"] = str(saved_path)
                    st.success(f"Saved to `{saved_path}`")
                except OSError as exc:
                    st.error(f"Could not save PDF: {exc}")

            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=filename_pdf,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
            saved_pdf = result.get("saved_paths", {}).get("pdf")
            if saved_pdf:
                st.caption(f"On disk: `{saved_pdf}`")
        else:
            st.button(
                "Download PDF",
                disabled=True,
                use_container_width=True,
                help="Install LibreOffice for PDF export",
            )
            st.caption(
                "PDF export requires LibreOffice (`sudo apt install libreoffice-writer`)."
            )


config = load_config()
background_path = (ROOT / config.get("background_file", "./background.md")).resolve()
output_dir = (ROOT / config.get("output_dir", "./output")).resolve()
endpoint_url = config.get("endpoint_url", "")
api_key = config.get("api_key", "ollama")
default_model = config.get("model_name", "")
max_chars = config.get("ai_output_max_chars", DEFAULT_AI_OUTPUT_MAX_CHARS)
max_pages = int(config.get("max_resume_pages", 2))
enable_job_aware_selection = bool(config.get("enable_job_aware_selection", False))
enable_ats_compat = bool(config.get("enable_ats_compat", True))
auto_fill_page_budget = bool(config.get("auto_fill_page_budget", True))
overflow_warning_min_composite = float(config.get("overflow_warning_min_composite", 75.0))
min_project_entries = int(config.get("min_project_entries", 1))
min_project_bullets = int(config.get("min_project_bullets", 1))
max_skill_categories = int(config.get("max_skill_categories", 4))
max_skills_per_category = int(config.get("max_skills_per_category", 5))
max_chars_per_skill_line = int(config.get("max_chars_per_skill_line", 88))
max_certifications = int(config.get("max_certifications", 1))
resume_sections = config["resume_sections"]
include_summary = "summary" in resume_sections
include_skills = "skills" in resume_sections
needs_ai = include_summary or include_skills
needs_job_description = needs_ai or enable_job_aware_selection
models = model_options(config)
if default_model and default_model not in models:
    models = [default_model, *models]
elif not models and default_model:
    models = [default_model]

st.set_page_config(page_title="Resume Tailor", layout="wide")
st.title("Resume Tailor")
st.caption(
    "Preview static sections from background.md to check formatting, or generate a "
    "job-tailored resume with AI summary and skills."
)

example_background_path = ROOT / "background.example.md"
try:
    background_data = load_background(background_path)
    background_ok = True
except (ValueError, FileNotFoundError, yaml.YAMLError) as exc:
    background_ok = False
    background_data = None
    if isinstance(exc, FileNotFoundError) and example_background_path.exists():
        st.error(
            "Background file not found. Copy `background.example.md` to `background.md` "
            "and add your resume data."
        )
    else:
        st.error(f"Background file error: {exc}")

with st.sidebar:
    st.header("Inputs")
    job_description = st.text_area(
        "Job description",
        height=280,
        placeholder=(
            "Paste the full job description here..."
            if needs_job_description
            else "Optional."
        ),
    )
    model_index = models.index(default_model) if default_model in models else 0
    selected_model = st.selectbox("Model", models, index=model_index)
    st.text_input("API endpoint", value=endpoint_url or "(set OPENAI_BASE_URL in .env)", disabled=True)
    preview = st.button("Preview from background", use_container_width=True)
    generate = st.button("Generate tailored content", type="primary", use_container_width=True)

    if (needs_ai or enable_job_aware_selection) and not endpoint_url:
        st.warning(
            "Set OPENAI_BASE_URL in `.env` (or `endpoint_url` in config.toml) to use "
            "**Generate tailored content**. Background preview works without an API."
        )

    st.divider()
    if libreoffice_available():
        st.caption("PDF preview and export available (LibreOffice detected).")
    else:
        st.caption(
            "PDF preview requires LibreOffice (`sudo apt install libreoffice-writer`). "
            "DOCX export always works."
        )

if not background_ok:
    st.stop()

if "result" not in st.session_state:
    st.session_state.result = None
if "review_nonce" not in st.session_state:
    st.session_state.review_nonce = 0

if preview:
    with st.spinner("Building background preview..."):
        st.session_state.review_nonce += 1
        st.session_state.result = build_background_preview(background_data)

if generate:
    if not api_configured() and (needs_ai or enable_job_aware_selection):
        st.error(
            "Set OPENAI_BASE_URL in a `.env` file (copy from `.env.example`) "
            "or `endpoint_url` in config.toml."
        )
    elif needs_job_description and not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        try:
            content_selection = None
            if enable_job_aware_selection:
                selection_limits = {
                    "min_project_entries": min_project_entries,
                    "min_project_bullets": min_project_bullets,
                }
                if job_description.strip():
                    with st.spinner("Selecting job-relevant experience and projects..."):
                        content_selection = generate_content_selection(
                            job_description,
                            background_data,
                            endpoint_url=endpoint_url,
                            model_name=selected_model,
                            api_key=api_key,
                            **selection_limits,
                        )
                else:
                    content_selection = default_selection(background_data, **selection_limits)

            if needs_ai:
                with st.spinner(
                    "Generating tailored summary and skills..."
                    if include_summary and include_skills
                    else (
                        "Generating tailored skills..."
                        if include_skills
                        else "Generating tailored summary..."
                    )
                ):
                    ai_output, skill_warnings, skill_packer_info = generate_tailored_content(
                        job_description,
                        background_path,
                        endpoint_url=endpoint_url,
                        model_name=selected_model,
                        api_key=api_key,
                        ai_output_max_chars=max_chars,
                        include_summary=include_summary,
                        include_skills=include_skills,
                        background_data=background_data,
                        content_selection=content_selection,
                        **skills_layout_kwargs(),
                    )
            else:
                ai_output, skill_warnings, skill_packer_info = dict(EMPTY_AI_OUTPUT), [], {
                    "removed_skills": [],
                    "deduped_skills": [],
                    "added_skills": [],
                    "line_utilization": [],
                }

            with st.spinner("Building resume preview..."):
                st.session_state.review_nonce += 1
                result = apply_artifacts_to_result(
                    {},
                    background_data,
                    ai_output,
                    skill_warnings=skill_warnings,
                    skill_packer_info=skill_packer_info,
                    content_selection=content_selection,
                    **artifact_build_kwargs(),
                )
                result["job_description"] = job_description.strip()
                result["revision_history"] = []
                result["preview_mode"] = False
                result.pop("sections", None)
                st.session_state.result = result
        except AIResponseError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))

result = st.session_state.result

if result is not None and "skill_categories" not in result and result.get("skills"):
    result["skill_categories"] = [{"name": "Skills", "skills": result["skills"]}]

if result is None:
    st.info(
        "Click **Preview from background** to check how experience, education, projects, "
        "and other static sections render (no API needed). "
        "Use **Generate tailored content** with a job description for AI skills and job-aware selection."
    )
else:
    if result.get("preview_mode"):
        st.caption(
            "Background preview — summary and skills omitted. "
            "Use **Generate tailored content** for tailored AI sections."
        )

    if result.get("skill_warnings"):
        st.warning(
            "Skills removed because they were not found in your background file: "
            + ", ".join(result["skill_warnings"])
        )

    diagnostics = result.get("diagnostics") or {}
    if diagnostics.get("trim_stalled"):
        page_count = diagnostics.get("page_count")
        max_p = diagnostics.get("max_pages", max_pages)
        st.error(
            f"Resume is still {page_count} page(s) (target: {max_p}) after trimming experience "
            "and project bullets. Skills were not modified. Shorten bullets in background.md "
            "or remove sections such as projects or certifications."
        )

    if diagnostics.get("overflow_warning") and diagnostics.get("overflow_message"):
        st.info(diagnostics["overflow_message"])

    left_col, right_col = st.columns([2, 3], gap="large")

    with left_col:
        if needs_ai and not result.get("preview_mode"):
            st.subheader("Revise")
            for message in result.get("revision_history", []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if include_summary and include_skills:
                revision_placeholder = "Ask for changes to the summary or skills..."
                revision_spinner = "Revising summary and skills..."
                revision_reply = "Updated the summary and skills. Check the preview on the right."
            elif include_skills:
                revision_placeholder = "Ask for changes to the skills..."
                revision_spinner = "Revising skills..."
                revision_reply = "Updated the skills. Check the preview on the right."
            else:
                revision_placeholder = "Ask for changes to the summary..."
                revision_spinner = "Revising summary..."
                revision_reply = "Updated the summary. Check the preview on the right."

            revision_prompt = st.chat_input(
                revision_placeholder,
                key="revision_chat_input",
            )
            if revision_prompt:
                if not api_configured():
                    st.error(
                        "Set OPENAI_BASE_URL in `.env` or `endpoint_url` in config.toml "
                        "to revise summary or skills."
                    )
                else:
                    with st.spinner(revision_spinner):
                        try:
                            current_output = {
                                "summary": result.get("summary", ""),
                                "skill_categories": result.get("skill_categories", []),
                            }
                            ai_output, skill_warnings, skill_packer_info = revise_tailored_content(
                                result.get("job_description", job_description),
                                current_output,
                                revision_prompt,
                                background_path,
                                endpoint_url=endpoint_url,
                                model_name=selected_model,
                                api_key=api_key,
                                ai_output_max_chars=max_chars,
                                revision_history=result.get("revision_history"),
                                include_summary=include_summary,
                                include_skills=include_skills,
                                background_data=background_data,
                                content_selection=result.get("content_selection"),
                                **skills_layout_kwargs(),
                            )
                            with st.spinner("Updating resume preview..."):
                                history = list(result.get("revision_history", []))
                                history.append({"role": "user", "content": revision_prompt})
                                history.append(
                                    {
                                        "role": "assistant",
                                        "content": revision_reply,
                                    }
                                )
                                apply_artifacts_to_result(
                                    result,
                                    background_data,
                                    ai_output,
                                    skill_warnings=skill_warnings,
                                    skill_packer_info=skill_packer_info,
                                    **artifact_build_kwargs(result),
                                )
                                result["revision_history"] = history
                                st.session_state.review_nonce += 1
                                st.rerun()
                        except AIResponseError as exc:
                            st.error(str(exc))
                        except ValueError as exc:
                            st.error(str(exc))

            with st.expander("Advanced edit"):
                nonce = st.session_state.review_nonce
                skill_categories = result.get("skill_categories", [])
                manual_summary = ""
                manual_skills_text = ""
                if include_summary:
                    manual_summary = st.text_area(
                        "Professional Summary",
                        value=result.get("summary", ""),
                        height=120,
                        key=f"edit_summary_{nonce}",
                    )
                if include_skills:
                    manual_skills_text = st.text_area(
                        "Skills (Category: skill1, skill2 — one category per line)",
                        value=format_skill_categories(skill_categories),
                        height=160,
                        key=f"edit_skills_{nonce}",
                    )
                if st.button("Apply changes", key="apply_manual_edit"):
                    manual_categories = (
                        parse_skill_categories(manual_skills_text) if include_skills else []
                    )
                    apply_error: str | None = None
                    filtered = manual_categories
                    skill_packer_info: dict | None = None

                    if include_summary and not manual_summary.strip():
                        apply_error = "Professional summary cannot be empty."
                    elif include_skills and not manual_categories:
                        apply_error = "Add at least one skill category (e.g. Languages: Python, Go)."
                    elif include_skills:
                        skills_map = load_skills_map(background_path)
                        if not skills_map:
                            apply_error = (
                                "background.md must define skills_map or ## Core strengths bullets."
                            )
                        else:
                            filtered, dropped = filter_skill_categories(
                                manual_categories,
                                skills_map,
                            )
                            if dropped:
                                apply_error = (
                                    "Skills not in skills_map: " + ", ".join(dropped)
                                )
                            elif not filtered:
                                apply_error = "No valid skills remain after skills_map check."
                            else:
                                evidence_text = build_evidence_text(
                                    read_background_text(background_path),
                                    background_data,
                                    result.get("content_selection"),
                                )
                                filtered, skill_packer_info = finalize_manual_skills(
                                    filtered,
                                    skills_map,
                                    result.get("job_description", job_description),
                                    evidence_text=evidence_text,
                                    **skills_layout_kwargs(),
                                )
                                if not filtered:
                                    apply_error = (
                                        "Skills could not fit the configured one-line "
                                        "layout limits."
                                    )

                    if apply_error:
                        st.error(apply_error)
                    else:
                        with st.spinner("Updating resume preview..."):
                            apply_artifacts_to_result(
                                result,
                                background_data,
                                {
                                    "summary": manual_summary if include_summary else "",
                                    "skill_categories": filtered if include_skills else [],
                                },
                                skill_packer_info=skill_packer_info if include_skills else None,
                                **artifact_build_kwargs(result),
                            )
                            st.session_state.review_nonce += 1
                            st.rerun()
        elif result.get("preview_mode"):
            st.caption(
                "Revise experience and project bullets in background.md, then click "
                "**Preview from background** again."
            )
        else:
            st.caption("No AI sections configured. Edit background.md or config.toml to enable revision.")

        render_save_section(result, output_dir)
        preview_sections = result.get("sections") or resume_sections
        preview_include_summary = "summary" in preview_sections
        preview_include_skills = "skills" in preview_sections
        render_page_fit_diagnostic(
            result.get("diagnostics"),
            background_data,
            max_chars=max_chars,
            include_summary=preview_include_summary,
            include_skills=preview_include_skills,
            enable_job_aware_selection=enable_job_aware_selection and not result.get("preview_mode"),
        )
        if preview_include_skills:
            render_skill_packer_diagnostic(result.get("skill_packer_info"))
        render_ats_section(
            result,
            background_data,
            job_description,
            sections=preview_sections,
            max_certifications=max_certifications,
            enable_ats_compat=enable_ats_compat,
        )

    with right_col:
        st.subheader("Resume preview")
        render_page_indicator(result.get("page_count"), max_pages)

        pdf_bytes = result.get("pdf_bytes")
        if pdf_bytes:
            st.pdf(pdf_bytes, height=900)
        elif result.get("html"):
            st.caption("Showing HTML fallback — install LibreOffice for PDF-accurate preview.")
            components.html(result["html"], height=900, scrolling=True)
        else:
            st.info("Preview will appear after generation.")
