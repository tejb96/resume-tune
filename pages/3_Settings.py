"""Streamlit page for editing app settings."""

from __future__ import annotations

import streamlit as st

from resume_tune.render.resume import DEFAULT_RESUME_SECTIONS, VALID_RESUME_SECTIONS
from resume_tune.settings import save_env_settings, save_local_config
from resume_tune.ui.common import clear_config_cache, load_config


def _ordered_sections(current: list[str], selected: list[str]) -> list[str]:
    kept = [section for section in current if section in selected]
    added = [section for section in selected if section not in kept]
    return kept + added if kept or added else list(DEFAULT_RESUME_SECTIONS)


st.set_page_config(page_title="Settings", layout="wide")
st.title("Settings")
st.caption(
    "LLM credentials are saved to `.env`. Other overrides are saved to `config.local.toml` "
    "and merged over committed `config.toml` defaults."
)

config = load_config()
current_sections = config.get("resume_sections") or list(DEFAULT_RESUME_SECTIONS)

with st.form("settings_form"):
    with st.expander("LLM (.env)", expanded=True):
        endpoint_url = st.text_input("API endpoint", value=config.get("endpoint_url", ""))
        api_key = st.text_input("API key", value=config.get("api_key", ""), type="password")
        model_name = st.text_input("Model name", value=config.get("model_name", ""))
        ai_output_max_chars = st.number_input(
            "AI output max chars",
            min_value=1,
            value=int(config.get("ai_output_max_chars", 600)),
        )

    with st.expander("Resume sections"):
        available = sorted(VALID_RESUME_SECTIONS)
        selected_sections = st.multiselect(
            "Included sections (order preserved from current config, then new picks)",
            options=available,
            default=[section for section in current_sections if section in available],
        )
        ordered_sections = _ordered_sections(current_sections, selected_sections)
        st.caption("Render order: " + ", ".join(ordered_sections))
        enable_job_aware_selection = st.toggle(
            "Enable job-aware bullet selection",
            value=bool(config.get("enable_job_aware_selection", False)),
        )
        enable_ats_compat = st.toggle(
            "Enable ATS compatibility checks",
            value=bool(config.get("enable_ats_compat", True)),
        )

    with st.expander("Page fitting"):
        max_resume_pages = st.number_input(
            "Max resume pages",
            min_value=1,
            value=int(config.get("max_resume_pages", 2)),
        )
        auto_fill_page_budget = st.toggle(
            "Auto-fill page budget",
            value=bool(config.get("auto_fill_page_budget", True)),
        )
        overflow_warning_min_composite = st.number_input(
            "Overflow warning min composite score",
            min_value=0.0,
            max_value=100.0,
            value=float(config.get("overflow_warning_min_composite", 75.0)),
        )
        min_project_entries = st.number_input(
            "Min project entries",
            min_value=0,
            value=int(config.get("min_project_entries", 1)),
        )
        min_project_bullets = st.number_input(
            "Min project bullets",
            min_value=0,
            value=int(config.get("min_project_bullets", 1)),
        )
        max_certifications = st.number_input(
            "Max certifications exported",
            min_value=0,
            value=int(config.get("max_certifications", 1)),
        )

    with st.expander("Skills layout"):
        max_skill_categories = st.number_input(
            "Max skill categories",
            min_value=1,
            value=int(config.get("max_skill_categories", 4)),
        )
        max_skills_per_category = st.number_input(
            "Max skills per category",
            min_value=1,
            value=int(config.get("max_skills_per_category", 5)),
        )
        max_chars_per_skill_line = st.number_input(
            "Max chars per skill line",
            min_value=1,
            value=int(config.get("max_chars_per_skill_line", 88)),
        )
        max_completion_tokens_raw = st.text_input(
            "Max completion tokens (optional)",
            value=(
                ""
                if config.get("max_completion_tokens") is None
                else str(config.get("max_completion_tokens"))
            ),
            help="Leave blank to auto-compute from layout settings.",
        )

    with st.expander("Paths (config.local.toml)"):
        output_dir = st.text_input(
            "Applications directory",
            value=str(config.get("output_dir", "./Applications")),
        )
        background_file = st.text_input(
            "Background file", value=str(config.get("background_file", "./background.md"))
        )
        tracker_file = st.text_input(
            "Tracker file",
            value=str(config.get("tracker_file", "./Applications/applications.xlsx")),
        )

    save = st.form_submit_button("Save settings", type="primary")

if save:
    try:
        save_env_settings(
            endpoint_url=endpoint_url,
            api_key=api_key,
            model_name=model_name,
            ai_output_max_chars=int(ai_output_max_chars),
        )
        local_updates = {
            "output_dir": output_dir.strip(),
            "background_file": background_file.strip(),
            "tracker_file": tracker_file.strip(),
            "max_resume_pages": int(max_resume_pages),
            "auto_fill_page_budget": bool(auto_fill_page_budget),
            "overflow_warning_min_composite": float(overflow_warning_min_composite),
            "enable_job_aware_selection": bool(enable_job_aware_selection),
            "enable_ats_compat": bool(enable_ats_compat),
            "min_project_entries": int(min_project_entries),
            "min_project_bullets": int(min_project_bullets),
            "max_skill_categories": int(max_skill_categories),
            "max_skills_per_category": int(max_skills_per_category),
            "max_chars_per_skill_line": int(max_chars_per_skill_line),
            "max_certifications": int(max_certifications),
            "resume_sections": ordered_sections,
        }
        tokens_text = max_completion_tokens_raw.strip()
        local_updates["max_completion_tokens"] = int(tokens_text) if tokens_text else None
        save_local_config(local_updates)
        clear_config_cache()
        st.success("Settings saved. Changes apply on the next page rerun.")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
