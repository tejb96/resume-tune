"""Shared Streamlit helpers for Resume Tune pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from resume_tune.settings import ROOT, load_settings


@st.cache_data
def load_config() -> dict:
    return load_settings()


def clear_config_cache() -> None:
    load_config.clear()


def model_options(config: dict) -> list[str]:
    models_section = config.get("models", {})
    ollama = models_section.get("ollama", [])
    lemonade = models_section.get("lemonade", [])
    combined = list(dict.fromkeys([*ollama, *lemonade, config.get("model_name", "")]))
    return [m for m in combined if m]


def resolve_paths(config: dict) -> tuple[Path, Path, Path]:
    background_path = (ROOT / config.get("background_file", "./background.md")).resolve()
    output_dir = (ROOT / config.get("output_dir", "./output")).resolve()
    tracker_path = (ROOT / config.get("tracker_file", "./output/applications.xlsx")).resolve()
    return background_path, output_dir, tracker_path


def example_background_path() -> Path:
    return ROOT / "background.example.md"
