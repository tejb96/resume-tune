"""Build a sample DOCX without calling the LLM."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resume import build_resume, build_resume_artifacts, docx_to_html, load_background


def main() -> None:
    data = load_background(ROOT / "background.example.md")
    ai_output = {
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
    docx_bytes = build_resume(data, ai_output)
    artifacts = build_resume_artifacts(data, ai_output)
    out = ROOT / "output" / "design_preview.docx"
    html_out = ROOT / "output" / "design_preview.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(docx_bytes)
    html_out.write_text(artifacts["html"], encoding="utf-8")
    print(f"Wrote {out} ({len(docx_bytes)} bytes)")
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
    print("  - Skills: one bullet per category with comma-joined skills")
    print("  - Experience: Role · Company · Location with dates right-aligned on same line")
    print("  - Projects section appears before Certifications")
    print("  - Project shows inline tech + clickable URL")


if __name__ == "__main__":
    main()
