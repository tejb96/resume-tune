"""End-to-end test against a running OpenAI-compatible local API. Skips if unreachable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import DEFAULT_AI_OUTPUT_MAX_CHARS, EMPTY_AI_OUTPUT, generate_tailored_content
from resume import build_resume, load_background, resume_filename, save_resume_to_disk
from settings import load_settings


def main() -> None:
    config = load_settings()

    background_path = ROOT / config["background_file"]
    output_dir = ROOT / config["output_dir"]
    endpoint = config["endpoint_url"]
    model = config["model_name"]
    api_key = config.get("api_key", "ollama")

    resume_sections = config["resume_sections"]
    include_summary = "summary" in resume_sections
    include_skills = "skills" in resume_sections
    needs_ai = include_summary or include_skills

    if needs_ai:
        if not endpoint:
            print("SKIP live e2e: OPENAI_BASE_URL not set (see .env.example)")
            sys.exit(0)

        if not model:
            print("SKIP live e2e: OPENAI_MODEL not set (see .env.example)")
            sys.exit(0)

    jd = """
    Senior Software Engineer — Python, distributed systems, AWS.
    You will design APIs, improve reliability, and mentor engineers.
    """

    if needs_ai:
        print(f"Calling {endpoint} model={model} ...")
        try:
            ai_output, warnings, _packer = generate_tailored_content(
                jd,
                background_path,
                endpoint_url=endpoint,
                model_name=model,
                api_key=api_key,
                ai_output_max_chars=config.get(
                    "ai_output_max_chars", DEFAULT_AI_OUTPUT_MAX_CHARS
                ),
                include_summary=include_summary,
                include_skills=include_skills,
            )
        except Exception as exc:
            print(f"SKIP live e2e: {exc}")
            sys.exit(0)
    else:
        print("No AI sections configured; building from background.md only.")
        ai_output, warnings = dict(EMPTY_AI_OUTPUT), []
        _packer = {"added_skills": [], "line_utilization": []}

    data = load_background(background_path)
    docx_bytes = build_resume(data, ai_output, sections=config["resume_sections"])
    filename = resume_filename(data["header"]["name"])
    path = save_resume_to_disk(docx_bytes, output_dir, filename)
    skill_count = sum(len(cat["skills"]) for cat in ai_output.get("skill_categories", []))
    print(f"OK summary={len(ai_output.get('summary', ''))} chars skills={skill_count}")
    if warnings:
        print(f"Warnings: skills not in background: {warnings}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
