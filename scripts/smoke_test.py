"""Build a sample DOCX without calling the LLM."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resume import build_resume, load_background


def main() -> None:
    data = load_background(ROOT / "background.md")
    ai_output = {
        "summary": (
            "Software engineer with 5+ years building scalable backend systems and APIs. "
            "Experienced in Python, Go, and cloud infrastructure with a track record of "
            "leading migrations and mentoring engineers."
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
    out = ROOT / "output" / "smoke_test_resume.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(docx_bytes)
    print(f"Wrote {out} ({len(docx_bytes)} bytes)")


if __name__ == "__main__":
    main()
