"""Deterministic python-docx resume formatter."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import mammoth
from pypdf import PdfReader

import frontmatter
from ai import enforce_output_budget, normalize_ai_output
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

FONT_NAME = "Calibri"
FONT_BODY = Pt(10)
FONT_NAME_SIZE = Pt(17)
FONT_TITLE_SIZE = Pt(10.5)
FONT_SECTION = Pt(10.5)
MARGIN_TB = Inches(0.55)
MARGIN_LR = Inches(0.65)
SECTION_SPACING_BEFORE = Pt(7)
SECTION_SPACING_AFTER = Pt(3)
BULLET_INDENT = Inches(0.22)
BULLET_HANGING = Inches(0.15)

ACCENT_COLOR = RGBColor(0x1A, 0x56, 0x8C)
RULE_COLOR_HEX = "1A568C"
META_COLOR = RGBColor(0x55, 0x55, 0x55)
LINK_COLOR = RGBColor(0x1A, 0x56, 0x8C)

DEFAULT_RESUME_SECTIONS = [
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
]
VALID_RESUME_SECTIONS = frozenset(DEFAULT_RESUME_SECTIONS)


def resolve_resume_sections(sections: list[str] | None) -> list[str]:
    """Validate and normalize section ids; fall back to default when unset or empty."""
    if not sections:
        return list(DEFAULT_RESUME_SECTIONS)

    unknown = [section for section in sections if section not in VALID_RESUME_SECTIONS]
    if unknown:
        raise ValueError(
            f"Unknown resume section(s): {', '.join(unknown)}. "
            f"Valid ids: {', '.join(sorted(VALID_RESUME_SECTIONS))}"
        )

    seen: set[str] = set()
    resolved: list[str] = []
    for section in sections:
        if section not in seen:
            seen.add(section)
            resolved.append(section)
    return resolved if resolved else list(DEFAULT_RESUME_SECTIONS)


def ai_section_flags(sections: list[str] | None) -> tuple[bool, bool]:
    """Return (include_summary, include_skills) from resolved resume section config."""
    resolved = resolve_resume_sections(sections)
    return "summary" in resolved, "skills" in resolved


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


def _render_summary_section(
    doc: Document,
    data: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    _add_section_heading(doc, "Professional Summary")
    _add_body_paragraph(doc, normalized["summary"])


def _render_skills_section(
    doc: Document,
    data: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    _add_section_heading(doc, "Skills")
    _add_skill_categories(doc, normalized["skill_categories"])


def _render_experience_section(
    doc: Document,
    data: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    experience = data.get("experience", [])
    if experience:
        _add_section_heading(doc, "Experience")
        for job in experience:
            _add_experience_entry(doc, job)


def _render_education_section(
    doc: Document,
    data: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    education = data.get("education", [])
    if education:
        _add_section_heading(doc, "Education")
        for entry in education:
            _add_education_entry(doc, entry)


def _render_projects_section(
    doc: Document,
    data: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    projects = data.get("projects", [])
    if projects:
        _add_section_heading(doc, "Projects")
        for project in projects:
            _add_project_entry(doc, project)


def _render_certifications_section(
    doc: Document,
    data: dict[str, Any],
    normalized: dict[str, Any],
) -> None:
    certifications = data.get("certifications", [])
    if certifications:
        _add_section_heading(doc, "Certifications")
        for cert in certifications:
            _add_certification_entry(doc, cert)


_SECTION_RENDERERS: dict[str, Any] = {
    "summary": _render_summary_section,
    "skills": _render_skills_section,
    "experience": _render_experience_section,
    "education": _render_education_section,
    "projects": _render_projects_section,
    "certifications": _render_certifications_section,
}


def build_resume(
    data: dict[str, Any],
    ai_output: dict[str, Any],
    *,
    sections: list[str] | None = None,
) -> bytes:
    """Merge structured background + AI fields, return DOCX bytes."""
    resolved_sections = resolve_resume_sections(sections)
    include_summary, include_skills = ai_section_flags(resolved_sections)
    normalized = normalize_ai_output(
        ai_output,
        include_summary=include_summary,
        include_skills=include_skills,
    )

    doc = Document()
    _set_document_margins(doc)
    _set_default_font(doc)

    _add_header(doc, data["header"])
    for section_id in resolved_sections:
        _SECTION_RENDERERS[section_id](doc, data, normalized)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _set_document_margins(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = MARGIN_TB
        section.bottom_margin = MARGIN_TB
        section.left_margin = MARGIN_LR
        section.right_margin = MARGIN_LR


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


def _add_paragraph_border_bottom(paragraph, color_hex: str = RULE_COLOR_HEX) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color_hex)
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def _add_section_heading(doc: Document, title: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = SECTION_SPACING_BEFORE
    p.paragraph_format.space_after = SECTION_SPACING_AFTER
    run = p.add_run(title.upper())
    _format_run(run, bold=True, size=FONT_SECTION, color=ACCENT_COLOR)
    _add_paragraph_border_bottom(p)


def _add_hyperlink(paragraph, url: str, display_text: str) -> None:
    """Insert a genuine clickable hyperlink run into paragraph."""
    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    new_run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")

    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    r_pr.append(u)

    color_el = OxmlElement("w:color")
    color_el.set(qn("w:val"), RULE_COLOR_HEX)
    r_pr.append(color_el)

    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "20")
    r_pr.append(sz)

    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), FONT_NAME)
    r_fonts.set(qn("w:hAnsi"), FONT_NAME)
    r_pr.append(r_fonts)

    new_run.append(r_pr)
    t = OxmlElement("w:t")
    t.text = display_text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


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
        _format_run(contact_run, size=Pt(10), color=META_COLOR)

    links = header.get("links") or []
    if links:
        links_p = doc.add_paragraph()
        links_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        links_p.paragraph_format.space_after = Pt(8)

        for i, link in enumerate(links):
            label = link.get("label", "")
            url = link.get("url", "")
            if not (label and url):
                continue
            if i > 0:
                sep_run = links_p.add_run("  |  ")
                _format_run(sep_run, size=Pt(10), color=META_COLOR)
            _add_hyperlink(links_p, url, label)


def _add_skill_categories(doc: Document, skill_categories: list[dict[str, Any]]) -> None:
    """Render one bullet per skill category with comma-joined skills."""
    if not skill_categories:
        return

    for cat in skill_categories:
        skills_text = ", ".join(cat["skills"])
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Inches(0.05)
        bullet_run = p.add_run("▪  ")
        _format_run(bullet_run, color=ACCENT_COLOR)
        name_run = p.add_run(f"{cat['name']}: ")
        _format_run(name_run, bold=True)
        text_run = p.add_run(skills_text)
        _format_run(text_run)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def _normalize_date(value: str) -> str:
    if value.lower() == "present":
        return "Present"
    return value


def _format_date_range(start: str, end: str) -> str:
    return f"{_normalize_date(start)} – {_normalize_date(end)}"


def _add_experience_entry(doc: Document, job: dict[str, Any]) -> None:
    title_line = doc.add_paragraph()
    title_line.paragraph_format.space_before = Pt(5)
    title_line.paragraph_format.space_after = Pt(0)

    role_run = title_line.add_run(job["title"])
    _format_run(role_run, bold=True)

    sep = title_line.add_run("  ·  ")
    _format_run(sep, color=META_COLOR)

    company_run = title_line.add_run(job["company"])
    _format_run(company_run, color=META_COLOR)

    if job.get("location"):
        loc_sep = title_line.add_run("  ·  ")
        _format_run(loc_sep, color=META_COLOR)
        loc_run = title_line.add_run(job["location"])
        _format_run(loc_run, color=META_COLOR, size=Pt(9.5))

    tab_run = title_line.add_run("\t" + _format_date_range(job["start"], job["end"]))
    _format_run(tab_run, color=META_COLOR, size=Pt(9.5))

    p_pr = title_line._p.get_or_add_pPr()
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "right")
    tab.set(qn("w:pos"), "9072")
    tabs.append(tab)
    p_pr.append(tabs)

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
    title_p.paragraph_format.space_after = Pt(0)

    name_run = title_p.add_run(project["name"])
    _format_run(name_run, bold=True)

    tech = project.get("tech") or project.get("stack")
    if tech:
        tech_run = title_p.add_run(f"  —  {tech}")
        _format_run(tech_run, size=Pt(9.5), color=META_COLOR)

    if project.get("url"):
        url_p = doc.add_paragraph()
        url_p.paragraph_format.space_after = Pt(1)
        _add_hyperlink(url_p, project["url"], project["url"])

    for bullet in project.get("bullets", []):
        _add_bullet(doc, bullet)


def _add_bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = BULLET_INDENT
    p.paragraph_format.first_line_indent = -BULLET_HANGING
    p.paragraph_format.space_after = Pt(1.5)
    bullet_run = p.add_run("▪  ")
    _format_run(bullet_run, color=ACCENT_COLOR)
    text_run = p.add_run(text)
    _format_run(text_run)


PREVIEW_HTML_STYLE = """
<style>
  html, body {
    background: #ffffff;
    font-family: Calibri, "Segoe UI", sans-serif;
    font-size: 10pt;
    line-height: 1.35;
    color: #222;
    max-width: 8.5in;
    margin: 0 auto;
    padding: 0.55in 0.65in;
  }
  p { margin: 0.25em 0; }
  a { color: #1A568C; }
  strong { color: #1A568C; }
</style>
"""


def resume_filename(candidate_name: str, ext: str = "docx") -> str:
    """Build a stable resume filename from the candidate name in background.md."""
    safe = re.sub(r"[^\w\-]+", "_", candidate_name.strip()).strip("_")
    normalized_ext = ext.lstrip(".") or "docx"
    base = f"{safe}_Resume" if safe else "resume"
    return f"{base}.{normalized_ext}"


def save_resume_to_disk(file_bytes: bytes, output_dir: Path, filename: str) -> Path:
    """Write resume bytes to output_dir, overwriting any existing file with the same name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", ".."):
        safe_name = "resume.docx"
    path = output_dir / safe_name
    path.write_bytes(file_bytes)
    return path


@lru_cache(maxsize=1)
def libreoffice_available() -> bool:
    """Return True when LibreOffice headless conversion is available."""
    return shutil.which("soffice") is not None


def docx_to_html(docx_bytes: bytes) -> str:
    """Convert DOCX bytes to styled HTML for in-app preview."""
    result = mammoth.convert_to_html(io.BytesIO(docx_bytes))
    body = result.value
    return f"<!DOCTYPE html><html><head>{PREVIEW_HTML_STYLE}</head><body>{body}</body></html>"


def docx_to_pdf(docx_bytes: bytes) -> bytes | None:
    """Convert DOCX bytes to PDF via LibreOffice headless, or None if unavailable."""
    if not libreoffice_available():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        docx_path = tmp_path / "resume.docx"
        docx_path.write_bytes(docx_bytes)
        try:
            subprocess.run(
                [
                    "soffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(tmp_path),
                    str(docx_path),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None

        pdf_path = tmp_path / "resume.pdf"
        if not pdf_path.exists():
            return None
        return pdf_path.read_bytes()


def pdf_page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in a PDF."""
    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)


def trim_ai_output_one_step(
    ai_output: dict[str, Any],
    *,
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Remove one unit of content to reduce rendered page count."""
    include_summary, include_skills = ai_section_flags(sections)
    normalized = normalize_ai_output(
        ai_output,
        include_summary=include_summary,
        include_skills=include_skills,
    )
    summary = normalized["summary"]
    categories = [
        {"name": cat["name"], "skills": list(cat["skills"])}
        for cat in normalized["skill_categories"]
    ]

    if include_skills and categories and categories[-1]["skills"]:
        categories[-1]["skills"].pop()
        if not categories[-1]["skills"]:
            categories.pop()
        return {"summary": summary, "skill_categories": categories}

    if include_summary:
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        if len(sentences) > 1:
            sentences.pop()
            return {"summary": " ".join(sentences).strip(), "skill_categories": categories}

    return normalized


def fit_resume_to_page_budget(
    background_data: dict[str, Any],
    ai_output: dict[str, Any],
    *,
    max_pages: int = 2,
    max_chars: int | None = None,
    sections: list[str] | None = None,
) -> tuple[dict[str, Any], bytes, bytes | None, int | None]:
    """
    Trim AI output until the rendered PDF fits max_pages.

    Returns (fitted_ai_output, docx_bytes, pdf_bytes, page_count).
    page_count is None when LibreOffice is unavailable.
    """
    include_summary, include_skills = ai_section_flags(sections)
    current = normalize_ai_output(
        ai_output,
        include_summary=include_summary,
        include_skills=include_skills,
    )
    if max_chars is not None and (include_summary or include_skills):
        current = enforce_output_budget(
            current["summary"],
            current["skill_categories"],
            max_chars,
            include_summary=include_summary,
            include_skills=include_skills,
        )

    docx_bytes = build_resume(background_data, current, sections=sections)
    pdf_bytes = docx_to_pdf(docx_bytes)
    if pdf_bytes is None:
        return current, docx_bytes, None, None

    page_count = pdf_page_count(pdf_bytes)
    for _ in range(30):
        if page_count <= max_pages:
            break
        trimmed = trim_ai_output_one_step(current, sections=sections)
        if trimmed == current:
            break
        current = trimmed
        docx_bytes = build_resume(background_data, current, sections=sections)
        pdf_bytes = docx_to_pdf(docx_bytes)
        if pdf_bytes is None:
            break
        page_count = pdf_page_count(pdf_bytes)

    return current, docx_bytes, pdf_bytes, page_count


def build_resume_artifacts(
    background_data: dict[str, Any],
    ai_output: dict[str, Any],
    *,
    max_pages: int = 2,
    max_chars: int | None = None,
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Build DOCX and derived preview/export artifacts from background + AI output."""
    fitted_output, docx_bytes, pdf_bytes, page_count = fit_resume_to_page_budget(
        background_data,
        ai_output,
        max_pages=max_pages,
        max_chars=max_chars,
        sections=sections,
    )
    return {
        "ai_output": fitted_output,
        "docx_bytes": docx_bytes,
        "html": docx_to_html(docx_bytes),
        "pdf_bytes": pdf_bytes,
        "page_count": page_count,
    }
