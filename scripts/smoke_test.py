"""Build a sample DOCX without calling the LLM."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resume import build_resume, load_background


def main() -> None:
    data = load_background(ROOT / "background.example.md")
    ai_output = {
        "summary": (
            "Software engineer with 5+ years building scalable backend systems and APIs "
            "handling 2M+ daily requests. Deep experience in Python, Go, AWS, and Kubernetes "
            "matched to cloud-native platform roles. Ships reliable services with strong CI/CD "
            "discipline and mentors engineers through code review and pairing."
        ),
        "skills": [
            "Python",
            "Go",
            "PostgreSQL",
            "Redis",
            "AWS",
            "Docker",
            "Kubernetes",
            "REST APIs",
            "CI/CD",
        ],
    }
    docx_bytes = build_resume(data, ai_output)
    out = ROOT / "output" / "design_preview.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(docx_bytes)
    print(f"Wrote {out} ({len(docx_bytes)} bytes)")
    print("Verify in Word/LibreOffice:")
    print("  - Section headings navy with bold underline")
    print("  - LinkedIn/GitHub are Ctrl+click hyperlinks")
    print("  - Skills in two columns with ▪ bullets")
    print("  - Experience: Role · Company with dates right-aligned on same line")
    print("  - Project shows inline tech + clickable URL")


if __name__ == "__main__":
    main()
