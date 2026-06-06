"""Load settings from .env / environment, with config.toml fallbacks."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.toml"


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)


def load_settings() -> dict:
    _load_dotenv()

    with CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)

    endpoint_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or config.get("endpoint_url", "")
    ).strip()
    api_key = (os.getenv("OPENAI_API_KEY") or config.get("api_key") or "ollama").strip()
    model_name = (os.getenv("OPENAI_MODEL") or config.get("model_name") or "").strip()

    return {
        **config,
        "endpoint_url": endpoint_url,
        "api_key": api_key,
        "model_name": model_name,
    }
