"""Application folder layout under the Applications directory."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

JD_SNAPSHOT_FILENAME = "job_description.txt"
RESUME_BASENAME = "resume"


def _sanitize_folder_part(text: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", text.strip()).strip("_")
    return safe or "unknown"


def application_folder_basename(company: str, role: str, date_applied: str) -> str:
    """Build a stable folder name: ``{date}_{company}_{role}``."""
    return (
        f"{date_applied}_{_sanitize_folder_part(company)}_{_sanitize_folder_part(role)}"
    )


def resolve_application_dir(
    applications_root: Path,
    company: str,
    role: str,
    date_applied: str,
) -> Path:
    """Return a unique application folder path under ``applications_root``."""
    applications_root.mkdir(parents=True, exist_ok=True)
    base = application_folder_basename(company, role, date_applied)
    candidate = applications_root / base
    if not candidate.exists():
        return candidate

    suffix = 2
    while (applications_root / f"{base}_{suffix}").exists():
        suffix += 1
    return applications_root / f"{base}_{suffix}"


def organize_application_files(
    applications_root: Path,
    *,
    resume_path: Path,
    company: str,
    role: str,
    date_applied: str,
    job_description: str,
) -> tuple[Path, Path | None]:
    """Move the saved resume and JD snapshot into a company/role folder."""
    folder = resolve_application_dir(applications_root, company, role, date_applied)
    folder.mkdir(parents=True, exist_ok=True)

    source = resume_path.expanduser()
    suffix = source.suffix.lower() or ".docx"
    dest_resume = (folder / f"{RESUME_BASENAME}{suffix}").resolve()

    if source.is_file():
        resolved_source = source.resolve()
        if resolved_source != dest_resume:
            shutil.move(str(resolved_source), str(dest_resume))
    else:
        dest_resume = source.resolve()

    jd_path: Path | None = None
    jd = job_description.strip()
    if jd:
        jd_path = (folder / JD_SNAPSHOT_FILENAME).resolve()
        jd_path.write_text(jd, encoding="utf-8")

    return dest_resume, jd_path
