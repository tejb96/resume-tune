"""Extract application metadata (company, role, location, URL) from pasted job descriptions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI

from resume_tune.llm.ai import strip_json_fences

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

_COMPANY_LABEL_RE = re.compile(
    r"^(?:company|employer|organization|hiring\s+company)\s*[:–-]\s*(?P<value>.+)$",
    re.IGNORECASE,
)
_ROLE_LABEL_RE = re.compile(
    r"^(?:role|job\s+title|position|title)\s*[:–-]\s*(?P<value>.+)$",
    re.IGNORECASE,
)
_LOCATION_LABEL_RE = re.compile(
    r"^(?:location|office|work\s+location)\s*[:–-]\s*(?P<value>.+)$",
    re.IGNORECASE,
)

_LINKEDIN_LINE_RE = re.compile(
    r"^(?P<company>.+?)\s*(?:[·•|]|\s+-\s+)\s*(?P<location>.+)$"
)

_INLINE_LOCATION_RE = re.compile(
    r"\s*[\(（](remote|hybrid|on[- ]site)[\)）]\s*$",
    re.IGNORECASE,
)
_TITLE_LOCATION_SUFFIX_RE = re.compile(
    r"\s*(?:[—–-]\s*|\s+)\s*(?P<location>"
    r"(?:remote|hybrid|on[- ]site"
    r"|[\w\s.'-]+,\s*[A-Z]{2}(?:\s*\([^)]+\))?))\s*$",
    re.IGNORECASE,
)

_LOCATION_LINE_RE = re.compile(
    r"^(?:remote|hybrid|on[- ]site"
    r"|[\w\s.'-]+,\s*[A-Z]{2}(?:\s*\([^)]+\))?)$",
    re.IGNORECASE,
)

_BOILERPLATE_HEADERS = frozenset(
    {
        "job description",
        "about the role",
        "about the job",
        "about us",
        "about the company",
        "responsibilities",
        "requirements",
        "qualifications",
        "what you'll do",
        "what you will do",
        "what we're looking for",
        "what we are looking for",
        "apply now",
        "apply",
        "overview",
        "summary",
        "description",
        "benefits",
        "who you are",
        "the role",
        "the opportunity",
    }
)

_SENTENCE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|will|can|seeking|looking|join|build|develop|lead)\b",
    re.IGNORECASE,
)

_EMPLOYMENT_TYPE_SUFFIX_RE = re.compile(
    r"\s*[\(（]?(?:full[- ]?time|part[- ]?time|contract|internship|temporary)"
    r"(?:\s*/\s*(?:remote|hybrid|on[- ]?site))?[\)）]?\s*$",
    re.IGNORECASE,
)

_LLM_SYSTEM_PROMPT = """You extract job application metadata from a pasted job description.

RULES (strict):
- Output ONLY valid JSON. No markdown code fences. No preamble. No explanation.
- JSON schema: {"company": "<string>", "role": "<string>", "location": "<string>", "job_url": "<string>"}
- Use only information explicitly present in the job description. Do not invent or guess.
- Use "" for any field you cannot find with confidence.
- "role" is the job title/position name.
- "company" is the hiring organization name.
- "location" is city/region or Remote/Hybrid/On-site when stated.
- "job_url" is a job posting URL if present in the text.
"""

_LLM_MAX_COMPLETION_TOKENS = 120


@dataclass(frozen=True)
class JobDescriptionMetadata:
    company: str = ""
    role: str = ""
    location: str = ""
    job_url: str = ""

    def has_primary_fields(self) -> bool:
        return bool(self.company or self.role)

    def any_field(self) -> bool:
        return bool(self.company or self.role or self.location or self.job_url)


def _non_empty_lines(text: str, *, limit: int = 20) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t-–—|·•")


def _is_boilerplate(line: str) -> bool:
    normalized = re.sub(r"[^\w\s]", "", line.lower()).strip()
    if not normalized:
        return True
    if normalized in _BOILERPLATE_HEADERS:
        return True
    return any(normalized.startswith(header) for header in _BOILERPLATE_HEADERS)


def _looks_like_title(line: str) -> bool:
    if not line or len(line) > 100:
        return False
    if _is_boilerplate(line):
        return False
    if _URL_RE.search(line):
        return False
    if _COMPANY_LABEL_RE.match(line) or _ROLE_LABEL_RE.match(line):
        return False
    if _SENTENCE_VERB_RE.search(line):
        return False
    if line.endswith(".") and len(line.split()) > 5:
        return False
    return True


def _looks_like_company(line: str) -> bool:
    if not line or len(line) > 100:
        return False
    if _is_boilerplate(line):
        return False
    if _URL_RE.search(line):
        return False
    if _LOCATION_LINE_RE.match(line):
        return False
    if _SENTENCE_VERB_RE.search(line):
        return False
    return True


def _strip_employment_suffix(location: str) -> str:
    cleaned = _EMPLOYMENT_TYPE_SUFFIX_RE.sub("", location).strip()
    cleaned = re.sub(r"\s*[\(（](?:hybrid|remote|on[- ]site)[\)）]\s*$", "", cleaned, flags=re.IGNORECASE)
    return _clean_value(cleaned)


def _extract_url(text: str) -> str:
    for line in _non_empty_lines(text, limit=30):
        match = _URL_RE.search(line)
        if match:
            return match.group(0).rstrip(".,);]")
    return ""


def _extract_labeled_fields(lines: list[str]) -> JobDescriptionMetadata:
    company = ""
    role = ""
    location = ""
    for line in lines:
        if not company:
            match = _COMPANY_LABEL_RE.match(line)
            if match:
                company = _clean_value(match.group("value"))
        if not role:
            match = _ROLE_LABEL_RE.match(line)
            if match:
                role = _clean_value(match.group("value"))
        if not location:
            match = _LOCATION_LABEL_RE.match(line)
            if match:
                location = _clean_value(match.group("value"))
    return JobDescriptionMetadata(company=company, role=role, location=location)


def _extract_linkedin_style(lines: list[str]) -> JobDescriptionMetadata:
    if len(lines) < 2:
        return JobDescriptionMetadata()
    title_line = lines[0]
    second_line = lines[1]
    if not _looks_like_title(title_line):
        return JobDescriptionMetadata()
    match = _LINKEDIN_LINE_RE.match(second_line)
    if not match:
        return JobDescriptionMetadata()
    company = _clean_value(match.group("company"))
    location = _strip_employment_suffix(match.group("location"))
    if not company:
        return JobDescriptionMetadata()
    return JobDescriptionMetadata(
        company=company,
        role=_clean_value(title_line),
        location=location,
    )


def _extract_title_stack(lines: list[str]) -> JobDescriptionMetadata:
    if not lines or not _looks_like_title(lines[0]):
        return JobDescriptionMetadata()

    role = _clean_value(lines[0])
    location = ""

    inline_loc = _INLINE_LOCATION_RE.search(role)
    if inline_loc:
        location = inline_loc.group(1)
        role = _clean_value(_INLINE_LOCATION_RE.sub("", role))
    else:
        suffix_match = _TITLE_LOCATION_SUFFIX_RE.search(role)
        if suffix_match:
            location = _clean_value(suffix_match.group("location"))
            role = _clean_value(role[: suffix_match.start()])

    company = ""
    if len(lines) > 1 and _looks_like_company(lines[1]):
        company = _clean_value(lines[1])

    if not location and len(lines) > 2 and _LOCATION_LINE_RE.match(lines[2]):
        location = _clean_value(lines[2])
    elif not location and len(lines) > 1 and _LOCATION_LINE_RE.match(lines[1]) and not company:
        location = _clean_value(lines[1])

    if not role and not company:
        return JobDescriptionMetadata()
    return JobDescriptionMetadata(company=company, role=role, location=location)


def _merge_metadata(
    primary: JobDescriptionMetadata, secondary: JobDescriptionMetadata
) -> JobDescriptionMetadata:
    return JobDescriptionMetadata(
        company=primary.company or secondary.company,
        role=primary.role or secondary.role,
        location=primary.location or secondary.location,
        job_url=primary.job_url or secondary.job_url,
    )


def extract_jd_metadata_heuristic(job_description: str) -> JobDescriptionMetadata:
    """Parse company, role, location, and URL from common pasted JD formats."""
    text = (job_description or "").strip()
    if not text:
        return JobDescriptionMetadata()

    job_url = _extract_url(text)
    lines = _non_empty_lines(text)

    labeled = _extract_labeled_fields(lines)
    linkedin = _extract_linkedin_style(lines)
    title_stack = _extract_title_stack(lines)

    merged = JobDescriptionMetadata(job_url=job_url)
    for partial in (labeled, linkedin, title_stack):
        merged = _merge_metadata(merged, partial)

    return merged


def _parse_llm_metadata_payload(data: dict) -> JobDescriptionMetadata:
    def field(key: str) -> str:
        value = data.get(key, "")
        return _clean_value(str(value)) if value else ""

    return JobDescriptionMetadata(
        company=field("company"),
        role=field("role"),
        location=field("location"),
        job_url=field("job_url"),
    )


def extract_jd_metadata_llm(
    job_description: str,
    *,
    endpoint_url: str,
    model_name: str,
    api_key: str = "ollama",
) -> JobDescriptionMetadata:
    """Call the configured LLM to extract metadata when heuristics find nothing."""
    text = (job_description or "").strip()
    if not text or not endpoint_url or not model_name:
        return JobDescriptionMetadata()

    client = OpenAI(base_url=endpoint_url, api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_completion_tokens=_LLM_MAX_COMPLETION_TOKENS,
            temperature=0,
        )
    except (APIConnectionError, APIStatusError, ValueError):
        return JobDescriptionMetadata()

    content = (response.choices[0].message.content or "").strip()
    if not content:
        return JobDescriptionMetadata()

    cleaned = strip_json_fences(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return JobDescriptionMetadata()

    if not isinstance(data, dict):
        return JobDescriptionMetadata()

    return _parse_llm_metadata_payload(data)


def resolve_jd_metadata(
    job_description: str,
    *,
    endpoint_url: str = "",
    model_name: str = "",
    api_key: str = "ollama",
) -> JobDescriptionMetadata:
    """Heuristic extraction with optional LLM fallback when company and role are missing."""
    jd = (job_description or "").strip()
    if not jd:
        return JobDescriptionMetadata()

    heuristic = extract_jd_metadata_heuristic(jd)
    if heuristic.has_primary_fields():
        return heuristic

    if not endpoint_url or not model_name:
        return heuristic

    llm = extract_jd_metadata_llm(
        jd,
        endpoint_url=endpoint_url,
        model_name=model_name,
        api_key=api_key,
    )
    return _merge_metadata(heuristic, llm)
