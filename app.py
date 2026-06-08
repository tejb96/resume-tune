"""Streamlit UI for local resume tailoring."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

from ai import (
    AIResponseError,
    DEFAULT_AI_OUTPUT_MAX_CHARS,
    ai_output_char_count,
    generate_tailored_content,
)
from resume import build_resume, load_background, resume_filename, save_resume_to_disk
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


def parse_skills(skills_text: str) -> list[str]:
    return [line.strip() for line in skills_text.splitlines() if line.strip()]


config = load_config()
background_path = (ROOT / config.get("background_file", "./background.md")).resolve()
output_dir = (ROOT / config.get("output_dir", "./output")).resolve()
endpoint_url = config.get("endpoint_url", "")
api_key = config.get("api_key", "ollama")
default_model = config.get("model_name", "")
max_chars = config.get("ai_output_max_chars", DEFAULT_AI_OUTPUT_MAX_CHARS)
models = model_options(config)
if default_model and default_model not in models:
    models = [default_model, *models]
elif not models and default_model:
    models = [default_model]

st.set_page_config(page_title="Resume Tailor", layout="wide")
st.title("Resume Tailor")
st.caption(
    "Generate tailored summary and skills from a job description, review and edit, "
    "then build and download a formatted DOCX."
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
        placeholder="Paste the full job description here...",
    )
    model_index = models.index(default_model) if default_model in models else 0
    selected_model = st.selectbox("Model", models, index=model_index)
    st.text_input("API endpoint", value=endpoint_url or "(set OPENAI_BASE_URL in .env)", disabled=True)
    generate = st.button("Generate tailored content", type="primary", use_container_width=True)

if not background_ok:
    st.stop()

if not endpoint_url:
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
    if not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        with st.spinner("Generating tailored summary and skills..."):
            try:
                ai_output, skill_warnings = generate_tailored_content(
                    job_description,
                    background_path,
                    endpoint_url=endpoint_url,
                    model_name=selected_model,
                    api_key=api_key,
                    ai_output_max_chars=max_chars,
                )
                st.session_state.review_nonce += 1
                st.session_state.result = {
                    "summary": ai_output["summary"],
                    "skills": ai_output["skills"],
                    "skill_warnings": skill_warnings,
                    "docx_bytes": None,
                    "saved_path": None,
                    "filename": None,
                }
            except AIResponseError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))

result = st.session_state.result

if result is None:
    st.info(
        "Paste a job description in the sidebar and click **Generate tailored content**. "
        "Review and edit the summary and skills, check the character counts, then click "
        "**Build DOCX** to create your resume."
    )
else:
    if result.get("skill_warnings"):
        st.warning(
            "Some skills were not found verbatim in your background file: "
            + ", ".join(result["skill_warnings"])
        )

    nonce = st.session_state.review_nonce
    st.subheader("Review tailored content")

    summary = st.text_area(
        "Professional Summary",
        value=result["summary"],
        height=140,
        key=f"edit_summary_{nonce}",
    )
    skills_text = st.text_area(
        "Skills (one per line)",
        value="\n".join(result["skills"]),
        height=200,
        key=f"edit_skills_{nonce}",
    )
    skills = parse_skills(skills_text)
    summary_stripped = summary.strip()
    summary_chars = len(summary_stripped)
    summary_words = len(summary_stripped.split()) if summary_stripped else 0
    skills_chars = sum(len(skill) for skill in skills)
    total_chars = ai_output_char_count(summary_stripped, skills)

    st.subheader("Character budget")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Summary characters", summary_chars)
    m2.metric("Summary words", summary_words)
    m3.metric("Skills", f"{len(skills)} ({skills_chars} chars)")
    m4.metric("Total characters", f"{total_chars} / {max_chars}")

    if total_chars > max_chars:
        st.warning(
            f"Total characters ({total_chars}) exceed the page budget ({max_chars}). "
            "Extra text may spill onto an additional page. Trim the summary or skills "
            "if you want a tighter layout."
        )
    else:
        st.caption("Within page budget.")

    build_docx = st.button("Build DOCX", type="primary")
    if build_docx:
        if not summary_stripped:
            st.error("Professional summary cannot be empty.")
        elif not skills:
            st.error("Add at least one skill (one per line).")
        else:
            try:
                ai_output = {"summary": summary_stripped, "skills": skills}
                docx_bytes = build_resume(background_data, ai_output)
                filename = resume_filename(background_data["header"]["name"])
                saved_path = None
                try:
                    saved_path = save_resume_to_disk(
                        docx_bytes,
                        output_dir,
                        filename,
                    )
                except OSError as exc:
                    st.warning(f"Could not save to disk: {exc}")

                st.session_state.result["docx_bytes"] = docx_bytes
                st.session_state.result["filename"] = filename
                st.session_state.result["saved_path"] = (
                    str(saved_path) if saved_path else None
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    if result.get("docx_bytes"):
        download_name = result.get("filename") or "resume.docx"
        st.success(f"DOCX ready: {download_name}")
        st.download_button(
            label="Download DOCX",
            data=result["docx_bytes"],
            file_name=download_name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )
        if result.get("saved_path"):
            st.caption(f"Also saved to: `{result['saved_path']}`")
        st.caption(
            "If you edited the summary or skills above, click **Build DOCX** again "
            "before downloading."
        )
