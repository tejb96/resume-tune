"""Streamlit page for editing background.md (resume data)."""

from resume_tune.ui.background_editor import render_background_page
from resume_tune.ui.common import load_config, resolve_paths

config = load_config()
background_path, _, _ = resolve_paths(config)
render_background_page(background_path)
