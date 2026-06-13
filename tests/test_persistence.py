"""Tests for settings and background persistence helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import resume_tune.settings as settings
from resume_tune.render.resume import load_background, read_background_file, save_background
from resume_tune.settings import load_settings, read_env_file, save_env_settings, save_local_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        (ROOT / "config.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(settings, "CONFIG_PATH", config_path)
    monkeypatch.setattr(settings, "CONFIG_LOCAL_PATH", tmp_path / "config.local.toml")
    monkeypatch.setattr(settings, "ENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("AI_OUTPUT_MAX_CHARS", raising=False)
    return tmp_path


def test_save_background_round_trip(tmp_path: Path) -> None:
    source = ROOT / "background.example.md"
    target = tmp_path / "background.md"
    metadata, body = read_background_file(source)
    save_background(target, metadata, body)
    loaded = load_background(target)
    assert loaded["header"]["name"] == metadata["header"]["name"]
    assert "experience" in loaded


def test_save_background_rejects_invalid_metadata(tmp_path: Path) -> None:
    target = tmp_path / "background.md"
    target.write_text("placeholder", encoding="utf-8")
    with pytest.raises(ValueError):
        save_background(target, {"header": {"name": "Only Name"}}, "body")
    assert target.read_text(encoding="utf-8") == "placeholder"


def test_save_env_settings_preserves_unrelated_lines(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# custom comment\nCUSTOM_FLAG=1\nOPENAI_MODEL=old-model\n",
        encoding="utf-8",
    )
    save_env_settings(
        endpoint_url="http://localhost:9999/api/v1",
        api_key="secret",
        model_name="new-model",
        ai_output_max_chars=700,
        path=env_path,
    )
    text = env_path.read_text(encoding="utf-8")
    assert "# custom comment" in text
    assert "CUSTOM_FLAG=1" in text
    assert "OPENAI_MODEL=new-model" in text
    assert "OPENAI_BASE_URL=http://localhost:9999/api/v1" in text
    assert read_env_file(env_path)["OPENAI_API_KEY"] == "secret"


def test_load_settings_merges_config_local(isolated_settings: Path) -> None:
    save_local_config(
        {
            "max_resume_pages": 3,
            "resume_sections": ["experience", "skills"],
        },
        path=isolated_settings / "config.local.toml",
    )
    config = load_settings()
    assert config["max_resume_pages"] == 3
    assert config["resume_sections"] == ["experience", "skills"]
