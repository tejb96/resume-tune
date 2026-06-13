"""Streamlit page for viewing and editing logged job applications."""

from resume_tune.ui.applications_dashboard import render_applications_dashboard
from resume_tune.ui.common import load_config, resolve_paths

config = load_config()
_, applications_dir, tracker_path = resolve_paths(config)
render_applications_dashboard(applications_dir, tracker_path)
