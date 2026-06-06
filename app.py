"""Streamlit UI for local resume tailoring."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

from ai import AIResponseError, generate_tailored_content
from resume import build_resume, format_skill_groups, load_background, save_resume_to_disk
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


config = load_config()
background_path = (ROOT / config.get("background_file", "./background.md")).resolve()
output_dir = (ROOT / config.get("output_dir", "./output")).resolve()
endpoint_url = config.get("endpoint_url", "")
api_key = config.get("api_key", "ollama")
default_model = config.get("model_name", "")
models = model_options(config)
if default_model and default_model not in models:
    models = [default_model, *models]
elif not models and default_model:
    models = [default_model]

st.set_page_config(page_title="Resume Tailor", layout="wide")
st.title("Resume Tailor")
st.caption("Tailor summary and skills from a job description, then download a formatted DOCX.")

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
    generate = st.button("Generate resume", type="primary", use_container_width=True)

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
                )
                docx_bytes = build_resume(background_data, ai_output)
                saved_path = None
                try:
                    saved_path = save_resume_to_disk(
                        docx_bytes,
                        output_dir,
                        slug="resume",
                    )
                except OSError as exc:
                    st.warning(f"Could not save to disk: {exc}")

                st.session_state.result = {
                    "summary": ai_output["summary"],
                    "skills": ai_output["skills"],
                    "docx_bytes": docx_bytes,
                    "saved_path": str(saved_path) if saved_path else None,
                    "skill_warnings": skill_warnings,
                }
            except AIResponseError as exc:
                st.error(str(exc))
            except ValueError as exc:
                st.error(str(exc))

result = st.session_state.result

if result is None:
    st.info(
        "Paste a job description in the sidebar and click **Generate resume**. "
        "You'll review the tailored summary and skills before downloading the DOCX."
    )
else:
    if result.get("skill_warnings"):
        st.warning(
            "Some skills were not found verbatim in your background file: "
            + ", ".join(result["skill_warnings"])
        )

    st.subheader("Professional Summary")
    st.write(result["summary"])

    st.subheader("Skills")
    st.markdown("\n".join(f"• {line}" for line in format_skill_groups(result["skills"])))

    st.download_button(
        label="Download DOCX",
        data=result["docx_bytes"],
        file_name="resume.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        type="primary",
    )

    if result.get("saved_path"):
        st.caption(f"Also saved to: `{result['saved_path']}`")
