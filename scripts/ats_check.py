#!/usr/bin/env python3
"""Print deterministic ATS compatibility report as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from resume_tune.llm.ai import EMPTY_AI_OUTPUT
from resume_tune.ats.ats import analyze_ats_compatibility
from resume_tune.render.resume import build_resume, docx_to_pdf, load_background
from resume_tune.settings import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="ATS compatibility check (deterministic)")
    parser.add_argument(
        "--background",
        type=Path,
        default=ROOT / "background.example.md",
        help="Path to background.md",
    )
    parser.add_argument(
        "--jd-file",
        type=Path,
        required=True,
        help="Path to job description text file",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Optional existing PDF to analyze (skips DOCX→PDF conversion)",
    )
    parser.add_argument(
        "--no-pdf-convert",
        action="store_true",
        help="Skip LibreOffice PDF conversion; use flattened text only",
    )
    args = parser.parse_args()

    background_data = load_background(args.background)
    job_description = args.jd_file.read_text(encoding="utf-8")

    pdf_bytes: bytes | None = None
    if args.pdf:
        pdf_bytes = args.pdf.read_bytes()
    elif not args.no_pdf_convert:
        docx_bytes = build_resume(background_data, EMPTY_AI_OUTPUT)
        pdf_bytes = docx_to_pdf(docx_bytes)

    report = analyze_ats_compatibility(
        job_description=job_description,
        background_data=background_data,
        ai_output=EMPTY_AI_OUTPUT,
        pdf_bytes=pdf_bytes,
        sections=None,
        content_selection=None,
        max_certifications=None,
    )

    if report is None:
        print(json.dumps({"error": "empty job description"}, indent=2))
        return 1

    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
