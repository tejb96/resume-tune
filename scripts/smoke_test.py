"""Build a sample DOCX without calling the LLM."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import EMPTY_AI_OUTPUT
from resume import build_resume, build_resume_artifacts, load_background, static_preview_sections
from selection import full_selection
from settings import load_settings

DEFAULT_AI_OUTPUT = {
    "summary": (
        "Software engineer with 5+ years building scalable backend systems and APIs "
        "handling 2M+ daily requests. Deep experience in Python, Go, AWS, and Kubernetes "
        "matched to cloud-native platform roles. Ships reliable services with strong CI/CD "
        "discipline and mentors engineers through code review and pairing."
    ),
    "skill_categories": [
        {
            "name": "Languages",
            "skills": ["Python", "Go", "SQL"],
        },
        {
            "name": "Infrastructure",
            "skills": ["AWS", "Docker", "Kubernetes", "CI/CD"],
        },
        {
            "name": "Data & APIs",
            "skills": ["PostgreSQL", "Redis", "REST APIs"],
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sample DOCX without calling the LLM.")
    parser.add_argument(
        "--background",
        default="background.example.md",
        help="Background file path relative to repo root (default: background.example.md)",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Mirror in-app background preview: static YAML sections only, no summary/skills",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_settings()
    config_sections = config["resume_sections"]
    background_path = ROOT / args.background
    data = load_background(background_path)

    if args.static_only:
        sections = static_preview_sections(config_sections)
        ai_output = dict(EMPTY_AI_OUTPUT)
        content_selection = full_selection(data)
    else:
        sections = config_sections
        ai_output = DEFAULT_AI_OUTPUT
        content_selection = None

    docx_bytes = build_resume(
        data,
        ai_output,
        sections=sections,
        content_selection=content_selection,
    )
    artifacts = build_resume_artifacts(
        data,
        ai_output,
        sections=sections,
        content_selection=content_selection,
    )
    out = ROOT / "output" / "design_preview.docx"
    html_out = ROOT / "output" / "design_preview.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(docx_bytes)
    html_out.write_text(artifacts["html"], encoding="utf-8")
    mode = "static background preview" if args.static_only else "design preview"
    print(f"Wrote {out} ({len(docx_bytes)} bytes) [{mode}]")
    print(f"Wrote {html_out} ({len(artifacts['html'])} chars)")
    if artifacts["pdf_bytes"]:
        pdf_out = ROOT / "output" / "design_preview.pdf"
        pdf_out.write_bytes(artifacts["pdf_bytes"])
        print(f"Wrote {pdf_out} ({len(artifacts['pdf_bytes'])} bytes)")
        print(f"Page count: {artifacts['page_count']}")
    else:
        print("PDF preview skipped (LibreOffice not available)")
    print("Verify in Word/LibreOffice or open design_preview.html in a browser:")
    print("  - Section headings navy with bold underline")
    print("  - LinkedIn/GitHub are Ctrl+click hyperlinks")
    if not args.static_only:
        print("  - Skills: one bullet per category with comma-joined skills")
    print("  - Experience: Role · Company · Location with dates right-aligned on same line")
    print(f"  - Section order: {', '.join(sections)}")
    print("  - Project shows inline tech + clickable URL")


if __name__ == "__main__":
    main()
