"""Tests for resume export helpers."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from resume import (
    BULLET_TEXT_INDENT,
    build_resume,
    build_resume_artifacts,
    docx_to_html,
    docx_to_pdf,
    fit_resume_to_page_budget,
    libreoffice_available,
    load_background,
    pdf_page_count,
)

ROOT = Path(__file__).resolve().parent.parent
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def _bullet_paragraphs(docx_bytes: bytes) -> list[ET.Element]:
    with ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    paragraphs: list[ET.Element] = []
    for paragraph in root.iter(f"{{{W_NS}}}p"):
        text_parts = [node.text or "" for node in paragraph.iter(f"{{{W_NS}}}t")]
        if "▪" in "".join(text_parts):
            paragraphs.append(paragraph)
    return paragraphs


def _paragraph_indent_twips(paragraph: ET.Element) -> tuple[str | None, str | None]:
    ind = paragraph.find("w:pPr/w:ind", NS)
    if ind is None:
        return None, None
    return ind.get(f"{{{W_NS}}}left"), ind.get(f"{{{W_NS}}}hanging")


def _paragraph_has_tab_after_bullet(paragraph: ET.Element) -> bool:
    children = list(paragraph)
    saw_bullet = False
    for child in children:
        if child.tag != f"{{{W_NS}}}r":
            continue
        text_nodes = child.findall("w:t", NS)
        text = "".join(node.text or "" for node in text_nodes)
        if "▪" in text:
            saw_bullet = True
            continue
        if saw_bullet and child.find("w:tab", NS) is not None:
            return True
    return False


@pytest.fixture
def sample_background() -> dict:
    return load_background(ROOT / "background.example.md")


@pytest.fixture
def sample_ai_output() -> dict:
    return {
        "summary": (
            "Software engineer with 5+ years building scalable backend systems and APIs "
            "handling 2M+ daily requests. Deep experience in Python, Go, AWS, and Kubernetes."
        ),
        "skill_categories": [
            {"name": "Languages", "skills": ["Python", "Go", "SQL"]},
            {"name": "Infrastructure", "skills": ["AWS", "Docker", "Kubernetes"]},
        ],
    }


def test_docx_to_html_contains_candidate_name(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    docx_bytes = build_resume(sample_background, sample_ai_output)
    html = docx_to_html(docx_bytes)

    assert "<html>" in html
    assert sample_background["header"]["name"] in html


def test_build_resume_artifacts_returns_docx_and_html(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    artifacts = build_resume_artifacts(sample_background, sample_ai_output)

    assert artifacts["docx_bytes"]
    assert artifacts["ai_output"]["skill_categories"]
    assert "content_selection" in artifacts
    assert "diagnostics" in artifacts
    assert sample_background["header"]["name"] in artifacts["html"]
    assert "pdf_bytes" in artifacts
    assert "page_count" in artifacts


def test_docx_to_pdf_returns_none_without_libreoffice(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    docx_bytes = build_resume(sample_background, sample_ai_output)

    with patch("resume.libreoffice_available", return_value=False):
        assert docx_to_pdf(docx_bytes) is None


@pytest.mark.skipif(not libreoffice_available(), reason="LibreOffice not installed")
def test_docx_to_pdf_returns_bytes_when_libreoffice_available(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    docx_bytes = build_resume(sample_background, sample_ai_output)
    pdf_bytes = docx_to_pdf(docx_bytes)

    assert pdf_bytes
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_page_count(pdf_bytes) >= 1


@pytest.mark.skipif(not libreoffice_available(), reason="LibreOffice not installed")
def test_fit_resume_to_page_budget_returns_page_count(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    fitted, _selection, _docx, pdf_bytes, page_count, diagnostics = fit_resume_to_page_budget(
        sample_background,
        sample_ai_output,
        max_pages=2,
    )

    assert fitted["summary"]
    assert fitted["skill_categories"]
    assert pdf_bytes
    assert page_count is not None
    assert page_count >= 1
    assert "experience_entries" in diagnostics


def test_bullet_paragraphs_use_hanging_indent_and_tab(
    sample_background: dict,
    sample_ai_output: dict,
) -> None:
    background = {
        **sample_background,
        "experience": [
            {
                **sample_background["experience"][0],
                "bullets": [
                    (
                        "Designed and shipped microservices handling 2M+ daily requests with "
                        "99.9% uptime and enough trailing detail to wrap onto a second line"
                    ),
                ],
            },
        ],
    }
    docx_bytes = build_resume(background, sample_ai_output)
    bullet_paragraphs = _bullet_paragraphs(docx_bytes)

    assert bullet_paragraphs, "Expected at least one bullet paragraph in DOCX"
    expected_twips = str(BULLET_TEXT_INDENT.twips)

    for paragraph in bullet_paragraphs:
        left, hanging = _paragraph_indent_twips(paragraph)
        assert left == expected_twips
        assert hanging == expected_twips
        assert _paragraph_has_tab_after_bullet(paragraph)


def test_legacy_flat_skills_migrated_in_build_resume(
    sample_background: dict,
) -> None:
    ai_output = {
        "summary": "Engineer with Python and Go experience on AWS.",
        "skills": ["Python", "Go", "AWS"],
    }
    docx_bytes = build_resume(sample_background, ai_output)
    html = docx_to_html(docx_bytes)

    assert "Python" in html
    assert "Go" in html
