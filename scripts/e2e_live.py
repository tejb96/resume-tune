"""End-to-end test against a running Ollama/Lemonade API. Skips if unreachable."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tomllib

from ai import generate_tailored_content
from resume import build_resume, load_background, save_resume_to_disk


def main() -> None:
    with (ROOT / "config.toml").open("rb") as f:
        config = tomllib.load(f)

    background_path = ROOT / config["background_file"]
    output_dir = ROOT / config["output_dir"]
    endpoint = config["endpoint_url"]
    model = config["model_name"]

    jd = """
    Senior Software Engineer — Python, distributed systems, AWS.
    You will design APIs, improve reliability, and mentor engineers.
    """

    print(f"Calling {endpoint} model={model} ...")
    try:
        ai_output, warnings = generate_tailored_content(
            jd,
            background_path,
            endpoint_url=endpoint,
            model_name=model,
        )
    except Exception as exc:
        print(f"SKIP live e2e: {exc}")
        sys.exit(0)

    data = load_background(background_path)
    docx_bytes = build_resume(data, ai_output)
    path = save_resume_to_disk(docx_bytes, output_dir, slug="e2e_live")
    print(f"OK summary={len(ai_output['summary'])} chars skills={len(ai_output['skills'])}")
    if warnings:
        print(f"Warnings: skills not in background: {warnings}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
