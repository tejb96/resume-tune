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
    filter_skill_categories,
    format_skill_categories,
    generate_tailored_content,
    parse_skill_categories,
    read_background_text,
    revise_tailored_content,
)
from resume import (
    build_resume_artifacts,
    libreoffice_available,
    load_background,
    resume_filename,
    save_resume_to_disk,
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
) -> dict:
    """Build DOCX, HTML preview, and optional PDF from current AI output."""
    artifacts = build_resume_artifacts(
        background_data,
        ai_output,
        max_pages=max_pages,
        max_chars=max_chars,
        sections=sections,
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
    *,
    max_pages: int,
    max_chars: int,
    sections: list[str],
) -> dict:
    """Update session result with new AI content and rebuilt preview artifacts."""
    artifacts = rebuild_artifacts(
        background_data,
        ai_output,
        max_pages=max_pages,
        max_chars=max_chars,
        sections=sections,
    )
    fitted = artifacts["ai_output"]
    result["summary"] = fitted["summary"]
    result["skill_categories"] = fitted["skill_categories"]
    if skill_warnings is not None:
        result["skill_warnings"] = skill_warnings
    result["docx_bytes"] = artifacts["docx_bytes"]
    result["html"] = artifacts["html"]
    result["pdf_bytes"] = artifacts["pdf_bytes"]
    result["page_count"] = artifacts["page_count"]
    result["filename_docx"] = artifacts["filename_docx"]
    result["filename_pdf"] = artifacts["filename_pdf"]
    result["saved_paths"] = {"docx": None, "pdf": None}
    return result


def render_page_indicator(page_count: int | None, max_pages: int) -> None:
    if page_count is None:
        st.caption(
            "Page count unavailable. Install LibreOffice for accurate PDF preview and page fitting."
        )
        return
    if page_count <= max_pages:
        st.success(f"Resume: {page_count} page{'s' if page_count != 1 else ''}")
    else:
        st.warning(
            f"Resume: {page_count} pages — content may need shortening (target: {max_pages} pages)."
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
resume_sections = config["resume_sections"]
include_summary = "summary" in resume_sections
include_skills = "skills" in resume_sections
needs_ai = include_summary or include_skills
models = model_options(config)
if default_model and default_model not in models:
    models = [default_model, *models]
elif not models and default_model:
    models = [default_model]

st.set_page_config(page_title="Resume Tailor", layout="wide")
st.title("Resume Tailor")
if needs_ai:
    st.caption(
        "Generate a tailored resume from a job description, preview it, revise summary or "
        "skills via chat, then save as DOCX or PDF."
    )
else:
    st.caption(
        "Build a resume from background.md using your configured sections, then save as DOCX or PDF."
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
            if needs_ai
            else "Optional when no AI sections are configured in config.toml."
        ),
    )
    model_index = models.index(default_model) if default_model in models else 0
    selected_model = st.selectbox("Model", models, index=model_index)
    st.text_input("API endpoint", value=endpoint_url or "(set OPENAI_BASE_URL in .env)", disabled=True)
    generate = st.button("Generate tailored content", type="primary", use_container_width=True)

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

if needs_ai and not endpoint_url:
    st.error(
        "Set OPENAI_BASE_URL in a `.env` file (copy from `.env.example`) "
        "or `endpoint_url` in config.toml."
    )
    st.stop()

if "result" not in st.session_state:
    st.session_state.result = None
if "review_nonce" not in st.session_state:
    st.session_state.review_nonce = 0

if generate:
    if needs_ai and not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        try:
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
                    ai_output, skill_warnings = generate_tailored_content(
                        job_description,
                        background_path,
                        endpoint_url=endpoint_url,
                        model_name=selected_model,
                        api_key=api_key,
                        ai_output_max_chars=max_chars,
                        include_summary=include_summary,
                        include_skills=include_skills,
                    )
            else:
                ai_output, skill_warnings = dict(EMPTY_AI_OUTPUT), []
            with st.spinner("Building resume preview..."):
                st.session_state.review_nonce += 1
                result = apply_artifacts_to_result(
                    {},
                    background_data,
                    ai_output,
                    skill_warnings=skill_warnings,
                    max_pages=max_pages,
                    max_chars=max_chars,
                    sections=resume_sections,
                )
                result["job_description"] = job_description.strip()
                result["revision_history"] = []
                st.session_state.result = result
        except AIResponseError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))

result = st.session_state.result

if result is not None and "skill_categories" not in result and result.get("skills"):
    result["skill_categories"] = [{"name": "Skills", "skills": result["skills"]}]

if result is None:
    if needs_ai:
        st.info(
            "Paste a job description in the sidebar and click **Generate tailored content**. "
            "You'll see a resume preview immediately. Use the chat to revise summary or skills, "
            "then save as DOCX or PDF."
        )
    else:
        st.info(
            "Click **Generate tailored content** to build a resume from background.md. "
            "Then save as DOCX or PDF."
        )
else:
    if result.get("skill_warnings"):
        st.warning(
            "Skills removed because they were not found in your background file: "
            + ", ".join(result["skill_warnings"])
        )

    left_col, right_col = st.columns([2, 3], gap="large")

    with left_col:
        if needs_ai:
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
                with st.spinner(revision_spinner):
                    try:
                        current_output = {
                            "summary": result.get("summary", ""),
                            "skill_categories": result.get("skill_categories", []),
                        }
                        ai_output, skill_warnings = revise_tailored_content(
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
                                max_pages=max_pages,
                                max_chars=max_chars,
                                sections=resume_sections,
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

                    if include_summary and not manual_summary.strip():
                        apply_error = "Professional summary cannot be empty."
                    elif include_skills and not manual_categories:
                        apply_error = "Add at least one skill category (e.g. Languages: Python, Go)."
                    elif include_skills:
                        background_text = read_background_text(background_path)
                        filtered, dropped = filter_skill_categories(
                            manual_categories,
                            background_text,
                        )
                        if dropped:
                            apply_error = (
                                "Skills not found in your background file: "
                                + ", ".join(dropped)
                            )
                        elif not filtered:
                            apply_error = "No valid skills remain after background check."

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
                                max_pages=max_pages,
                                max_chars=max_chars,
                                sections=resume_sections,
                            )
                            st.session_state.review_nonce += 1
                            st.rerun()
        else:
            st.caption("No AI sections configured. Edit background.md or config.toml to enable revision.")

        render_save_section(result, output_dir)

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
