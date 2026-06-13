"""Tests for job-description metadata extraction."""

from __future__ import annotations

import pytest

from resume_tune.tracker.jd_parse import (
    JobDescriptionMetadata,
    extract_jd_metadata_heuristic,
    extract_jd_metadata_llm,
    resolve_jd_metadata,
)

LINKEDIN_JD = """
Senior Software Engineer
Acme Corp · San Francisco, CA (Hybrid)

About the role
We are looking for a full-stack engineer.
"""

LABELED_JD = """
Company: Acme Corp
Job Title: Software Engineer
Location: Remote

Requirements:
- Python experience
"""

TITLE_STACK_JD = """
Backend Developer
Beta Labs
Austin, TX

Responsibilities:
- Build APIs
"""

URL_JD = """
Apply here: https://boards.greenhouse.io/acme/jobs/12345

Senior Engineer role at Acme.
"""

BODY_ONLY_JD = """
We are seeking a talented engineer to join our growing team.
You will work with Python, React, and AWS to deliver features.
"""

BOILERPLATE_JD = """
About the role

We are a fast-growing startup building developer tools.
"""


def test_linkedin_style_extracts_role_company_location() -> None:
    meta = extract_jd_metadata_heuristic(LINKEDIN_JD)
    assert meta.role == "Senior Software Engineer"
    assert meta.company == "Acme Corp"
    assert "San Francisco" in meta.location


def test_labeled_fields_extract_all_three() -> None:
    meta = extract_jd_metadata_heuristic(LABELED_JD)
    assert meta.company == "Acme Corp"
    assert meta.role == "Software Engineer"
    assert meta.location == "Remote"


def test_title_stack_extracts_role_company_location() -> None:
    meta = extract_jd_metadata_heuristic(TITLE_STACK_JD)
    assert meta.role == "Backend Developer"
    assert meta.company == "Beta Labs"
    assert meta.location == "Austin, TX"


def test_url_extraction() -> None:
    meta = extract_jd_metadata_heuristic(URL_JD)
    assert meta.job_url == "https://boards.greenhouse.io/acme/jobs/12345"


def test_body_only_jd_returns_empty_primary_fields() -> None:
    meta = extract_jd_metadata_heuristic(BODY_ONLY_JD)
    assert not meta.has_primary_fields()


def test_boilerplate_first_does_not_false_positive() -> None:
    meta = extract_jd_metadata_heuristic(BOILERPLATE_JD)
    assert meta.company == ""
    assert meta.role == ""


def test_empty_jd_returns_empty_metadata() -> None:
    meta = extract_jd_metadata_heuristic("")
    assert meta == JobDescriptionMetadata()


def test_resolve_skips_llm_when_heuristics_find_primary_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_llm(*_args, **_kwargs) -> JobDescriptionMetadata:
        nonlocal called
        called = True
        return JobDescriptionMetadata(company="LLM Co", role="LLM Role")

    monkeypatch.setattr(
        "resume_tune.tracker.jd_parse.extract_jd_metadata_llm",
        fake_llm,
    )
    meta = resolve_jd_metadata(LINKEDIN_JD, endpoint_url="http://local", model_name="test")
    assert meta.role == "Senior Software Engineer"
    assert meta.company == "Acme Corp"
    assert called is False


def test_resolve_calls_llm_when_heuristics_find_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_llm(
        job_description: str,
        *,
        endpoint_url: str,
        model_name: str,
        api_key: str = "ollama",
    ) -> JobDescriptionMetadata:
        assert job_description == BODY_ONLY_JD.strip()
        assert endpoint_url == "http://local"
        assert model_name == "test-model"
        return JobDescriptionMetadata(
            company="Inferred Co",
            role="Inferred Role",
            location="Remote",
            job_url="https://example.com/jobs/1",
        )

    monkeypatch.setattr(
        "resume_tune.tracker.jd_parse.extract_jd_metadata_llm",
        fake_llm,
    )
    meta = resolve_jd_metadata(
        BODY_ONLY_JD,
        endpoint_url="http://local",
        model_name="test-model",
    )
    assert meta.company == "Inferred Co"
    assert meta.role == "Inferred Role"
    assert meta.location == "Remote"
    assert meta.job_url == "https://example.com/jobs/1"


def test_resolve_llm_fills_only_empty_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_llm(
        _job_description: str,
        *,
        endpoint_url: str,
        model_name: str,
        api_key: str = "ollama",
    ) -> JobDescriptionMetadata:
        return JobDescriptionMetadata(
            company="LLM Co",
            role="LLM Role",
            location="New York, NY",
            job_url="https://example.com/jobs/2",
        )

    monkeypatch.setattr(
        "resume_tune.tracker.jd_parse.extract_jd_metadata_llm",
        fake_llm,
    )
    meta = resolve_jd_metadata(
        LABELED_JD,
        endpoint_url="http://local",
        model_name="test-model",
    )
    assert meta.company == "Acme Corp"
    assert meta.role == "Software Engineer"
    assert meta.location == "Remote"
    assert meta.job_url == ""


def test_resolve_skips_llm_without_api_config(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_llm(*_args, **_kwargs) -> JobDescriptionMetadata:
        nonlocal called
        called = True
        return JobDescriptionMetadata(company="LLM Co", role="LLM Role")

    monkeypatch.setattr(
        "resume_tune.tracker.jd_parse.extract_jd_metadata_llm",
        fake_llm,
    )
    meta = resolve_jd_metadata(BODY_ONLY_JD, endpoint_url="", model_name="")
    assert not meta.has_primary_fields()
    assert called is False


def test_extract_jd_metadata_llm_returns_empty_without_endpoint() -> None:
    meta = extract_jd_metadata_llm("Some JD", endpoint_url="", model_name="test")
    assert meta == JobDescriptionMetadata()
