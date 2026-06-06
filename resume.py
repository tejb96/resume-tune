"""Deterministic python-docx resume formatter."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import frontmatter
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

FONT_NAME = "Calibri"
FONT_BODY = Pt(10.5)
FONT_NAME_SIZE = Pt(16)
FONT_TITLE_SIZE = Pt(11)
FONT_SECTION = Pt(11)
MARGIN = Inches(0.6)
SECTION_SPACING_BEFORE = Pt(8)
SECTION_SPACING_AFTER = Pt(4)
BULLET_INDENT = Inches(0.25)
BULLET_HANGING = Inches(0.18)


def load_background(path: Path) -> dict[str, Any]:
    """Load and validate YAML frontmatter from background.md."""
    if not path.exists():
        raise FileNotFoundError(f"Background file not found: {path}")
    post = frontmatter.load(path)
    return validate_background(post.metadata)


def validate_background(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate frontmatter schema; return metadata unchanged if valid."""
    if not isinstance(metadata, dict):
        raise ValueError("Background frontmatter must be a YAML mapping")

    header = metadata.get("header")
    if not isinstance(header, dict):
        raise ValueError("background.md: 'header' must be a mapping")
    for key in ("name", "email"):
        if not header.get(key):
            raise ValueError(f"background.md: header.{key} is required")

    experience = metadata.get("experience")
    if not isinstance(experience, list) or not experience:
        raise ValueError("background.md: 'experience' must be a non-empty list")

    for i, job in enumerate(experience):
        if not isinstance(job, dict):
            raise ValueError(f"background.md: experience[{i}] must be a mapping")
        for key in ("company", "title", "start", "end", "bullets"):
            if key not in job:
                raise ValueError(f"background.md: experience[{i}].{key} is required")
        if not isinstance(job["bullets"], list) or not job["bullets"]:
            raise ValueError(f"background.md: experience[{i}].bullets must be non-empty")

    for section, required_keys in (
        ("education", ("institution", "degree", "graduation")),
        ("certifications", ("name",)),
        ("projects", ("name", "bullets")),
    ):
        items = metadata.get(section, [])
        if items is None:
            metadata[section] = []
            continue
        if not isinstance(items, list):
            raise ValueError(f"background.md: '{section}' must be a list")
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"background.md: {section}[{i}] must be a mapping")
            for key in required_keys:
                if key not in item:
                    raise ValueError(f"background.md: {section}[{i}].{key} is required")

    links = header.get("links", [])
    if links is not None and not isinstance(links, list):
        raise ValueError("background.md: header.links must be a list")

    return metadata


def build_resume(data: dict[str, Any], ai_output: dict[str, Any]) -> bytes:
    """Merge structured background + AI fields, return DOCX bytes."""
    summary = ai_output.get("summary", "").strip()
    skills = ai_output.get("skills", [])
    if not summary:
        raise ValueError("AI output missing non-empty 'summary'")
    if not isinstance(skills, list) or not skills:
        raise ValueError("AI output missing non-empty 'skills' list")
    if not all(isinstance(s, str) and s.strip() for s in skills):
        raise ValueError("AI output 'skills' must be a list of non-empty strings")

    doc = Document()
    _set_document_margins(doc)
    _set_default_font(doc)

    _add_header(doc, data["header"])
    _add_section_heading(doc, "Professional Summary")
    _add_body_paragraph(doc, summary)
    _add_section_heading(doc, "Skills")
    _add_skills(doc, skills)

    experience = data.get("experience", [])
    if experience:
        _add_section_heading(doc, "Experience")
        for job in experience:
            _add_experience_entry(doc, job)

    education = data.get("education", [])
    if education:
        _add_section_heading(doc, "Education")
        for entry in education:
            _add_education_entry(doc, entry)

    certifications = data.get("certifications", [])
    if certifications:
        _add_section_heading(doc, "Certifications")
        for cert in certifications:
            _add_certification_entry(doc, cert)

    projects = data.get("projects", [])
    if projects:
        _add_section_heading(doc, "Projects")
        for project in projects:
            _add_project_entry(doc, project)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _set_document_margins(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = MARGIN
        section.bottom_margin = MARGIN
        section.left_margin = MARGIN
        section.right_margin = MARGIN


def _set_default_font(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = FONT_NAME
    style.font.size = FONT_BODY
    rfonts = style.element.rPr.rFonts if style.element.rPr is not None else None
    if rfonts is None:
        rpr = style.element.get_or_add_rPr()
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), FONT_NAME)
    rfonts.set(qn("w:hAnsi"), FONT_NAME)


def _format_run(run, *, bold: bool = False, size: Pt | None = None, color: RGBColor | None = None) -> None:
    run.font.name = FONT_NAME
    run.font.size = size or FONT_BODY
    run.font.bold = bold
    if color:
        run.font.color.rgb = color


def _add_paragraph_border_bottom(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "404040")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def _add_section_heading(doc: Document, title: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = SECTION_SPACING_BEFORE
    p.paragraph_format.space_after = SECTION_SPACING_AFTER
    run = p.add_run(title.upper())
    _format_run(run, bold=True, size=FONT_SECTION)
    _add_paragraph_border_bottom(p)


def _add_body_paragraph(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    _format_run(run)


def _add_header(doc: Document, header: dict[str, Any]) -> None:
    name_p = doc.add_paragraph()
    name_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name_p.paragraph_format.space_after = Pt(2)
    name_run = name_p.add_run(header["name"])
    _format_run(name_run, bold=True, size=FONT_NAME_SIZE)

    title = header.get("title")
    if title:
        title_p = doc.add_paragraph()
        title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_p.paragraph_format.space_after = Pt(4)
        title_run = title_p.add_run(title)
        _format_run(title_run, size=FONT_TITLE_SIZE)

    contact_parts = [
        header.get("email"),
        header.get("phone"),
        header.get("location"),
    ]
    contact_line = " | ".join(p for p in contact_parts if p)
    if contact_line:
        contact_p = doc.add_paragraph()
        contact_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        contact_p.paragraph_format.space_after = Pt(2)
        contact_run = contact_p.add_run(contact_line)
        _format_run(contact_run, size=Pt(10))

    links = header.get("links") or []
    if links:
        link_labels = []
        for link in links:
            label = link.get("label", "")
            url = link.get("url", "")
            if label and url:
                link_labels.append(f"{label}: {url}")
        if link_labels:
            links_p = doc.add_paragraph()
            links_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            links_p.paragraph_format.space_after = Pt(8)
            links_run = links_p.add_run(" | ".join(link_labels))
            _format_run(links_run, size=Pt(10), color=RGBColor(0x2E, 0x5A, 0x88))


def _chunk_skills(skills: list[str], size: int = 4) -> list[list[str]]:
    return [skills[i : i + size] for i in range(0, len(skills), size)]


def format_skill_groups(skills: list[str], size: int = 4) -> list[str]:
    """Return comma-joined skill groups for preview or DOCX bullets."""
    return [", ".join(group) for group in _chunk_skills(skills, size)]


def _add_skills(doc: Document, skills: list[str]) -> None:
    for line in format_skill_groups(skills):
        _add_bullet(doc, line)


def _normalize_date(value: str) -> str:
    if value.lower() == "present":
        return "Present"
    return value


def _format_date_range(start: str, end: str) -> str:
    return f"{_normalize_date(start)} – {_normalize_date(end)}"


def _add_experience_entry(doc: Document, job: dict[str, Any]) -> None:
    title_line = doc.add_paragraph()
    title_line.paragraph_format.space_before = Pt(4)
    title_line.paragraph_format.space_after = Pt(1)
    company_run = title_line.add_run(job["company"])
    _format_run(company_run, bold=True)
    sep = title_line.add_run(" — ")
    _format_run(sep)
    role_run = title_line.add_run(job["title"])
    _format_run(role_run, bold=True)

    meta_parts = []
    if job.get("location"):
        meta_parts.append(job["location"])
    meta_parts.append(_format_date_range(job["start"], job["end"]))
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(2)
    meta_run = meta_p.add_run(" | ".join(meta_parts))
    _format_run(meta_run, size=Pt(10), color=RGBColor(0x55, 0x55, 0x55))

    for bullet in job["bullets"]:
        _add_bullet(doc, bullet)


def _add_education_entry(doc: Document, entry: dict[str, Any]) -> None:
    line = doc.add_paragraph()
    line.paragraph_format.space_before = Pt(2)
    line.paragraph_format.space_after = Pt(1)
    inst_run = line.add_run(entry["institution"])
    _format_run(inst_run, bold=True)
    if entry.get("location"):
        loc_run = line.add_run(f" — {entry['location']}")
        _format_run(loc_run)

    detail_p = doc.add_paragraph()
    detail_p.paragraph_format.space_after = Pt(4)
    detail = entry["degree"]
    if entry.get("graduation"):
        detail += f" | {entry['graduation']}"
    detail_run = detail_p.add_run(detail)
    _format_run(detail_run)


def _add_certification_entry(doc: Document, cert: dict[str, Any]) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    name_run = p.add_run(cert["name"])
    _format_run(name_run, bold=True)
    extras = []
    if cert.get("issuer"):
        extras.append(cert["issuer"])
    if cert.get("date"):
        extras.append(str(cert["date"]))
    if extras:
        extra_run = p.add_run(f" — {', '.join(extras)}")
        _format_run(extra_run)


def _add_project_entry(doc: Document, project: dict[str, Any]) -> None:
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(4)
    title_p.paragraph_format.space_after = Pt(1)
    name_run = title_p.add_run(project["name"])
    _format_run(name_run, bold=True)
    if project.get("url"):
        url_run = title_p.add_run(f" — {project['url']}")
        _format_run(url_run, size=Pt(10), color=RGBColor(0x2E, 0x5A, 0x88))

    for bullet in project.get("bullets", []):
        _add_bullet(doc, bullet)


def _add_bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = BULLET_INDENT
    p.paragraph_format.first_line_indent = -BULLET_HANGING
    p.paragraph_format.space_after = Pt(2)
    bullet_run = p.add_run("• ")
    _format_run(bullet_run)
    text_run = p.add_run(text)
    _format_run(text_run)


def save_resume_to_disk(docx_bytes: bytes, output_dir: Path, slug: str = "resume") -> Path:
    """Write DOCX bytes to output_dir with a timestamped filename."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    safe_slug = re.sub(r"[^\w\-]+", "_", slug).strip("_") or "resume"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_{safe_slug}.docx"
    path.write_bytes(docx_bytes)
    return path
