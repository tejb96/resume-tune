# Background file (`background.md`)

Your resume data lives in `background.md` — a YAML frontmatter file plus a markdown body. Copy from [`background.example.md`](../background.example.md):

```bash
cp background.example.md background.md
```

`background.md` is gitignored; never commit your real resume.

## Structure

Two parts separated by `---`:

1. **YAML frontmatter** — rendered in DOCX/PDF
2. **Markdown body** — career context for the LLM only (not exported)

## YAML schema

### `header` (required)

```yaml
header:
  name: "Your Name"
  title: "Software Engineer"
  email: "you@example.com"
  phone: "+1 (555) 123-4567"
  location: "City, ST"
  links:
    - label: "LinkedIn"
      url: "https://linkedin.com/in/yourprofile"
    - label: "GitHub"
      url: "https://github.com/yourusername"
```

Links render as clickable hyperlinks in DOCX/PDF.

### `experience`

```yaml
experience:
  - company: "Acme Corp"
    title: "Senior Software Engineer"
    location: "Remote"
    start: "2022-01"
    end: "present"        # or "2024-06"
    bullets:
      - "Achievement with metrics..."
      - "Another bullet..."
```

List most recent role first. The app may select subsets of bullets per job when job-aware selection is enabled.

### `education`

```yaml
education:
  - institution: "State University"
    degree: "B.S. Computer Science"
    location: "City, ST"
    graduation: "2019"
```

### `projects`

```yaml
projects:
  - name: "Resume Tune"
    url: "https://github.com/yourusername/resume-tune"
    tech: "Python · Streamlit · python-docx"
    bullets:
      - "One-line project description..."
```

### `certifications`

```yaml
certifications:
  - name: "AWS Solutions Architect – Associate"
    issuer: "Amazon Web Services"
    date: "2023"
```

When `max_certifications = 1` in config, only the **first** entry exports. Put your primary cert first (e.g. AWS SAA).

### `skills_map` (required when `"skills"` is in `resume_sections`)

```yaml
skills_map:
  languages:
    - Python
    - Go
  infrastructure:
    - AWS
    - Kubernetes
```

The LLM may **only** output skills from this map. Anything else is dropped by guardrails.

Rules:

- Use exact spellings you want on the resume
- Bucket names are organizational only — exported skill lines use empty category labels
- Include all skills you might want the LLM to pick for any job

### Fallback when `skills_map` is omitted

The app parses `## Core strengths` bullets from the markdown body:

```markdown
## Core strengths

- Python, Go, SQL
- AWS, Docker, Kubernetes
```

Prefer explicit `skills_map` when using the skills section — it is clearer and easier to validate.

## Markdown body (AI-only)

Everything after the closing `---` is narrative context for the LLM:

```markdown
# Career context (AI-only — not rendered in DOCX)

Software engineer with 5+ years building backend systems...
Describe domains, leadership, and strengths not obvious from bullets alone.
```

Use this for:

- Career themes the LLM should emphasize
- Context that does not fit a bullet format
- Evidence for skill selection when the JD is vague

## Preview without LLM

Check layout and bullet wording before generating:

```bash
uv run python scripts/smoke_test.py --background background.md --static-only
```

Or click **Preview from background** in the Streamlit sidebar.

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Skills section enabled but no `skills_map` or Core strengths | Add `skills_map` to YAML |
| LLM drops expected skills | Add exact spelling to `skills_map`; check JD relevance |
| Wrong cert exported | Reorder `certifications`; check `max_certifications` |
| Comma-joined skills in one YAML string | One skill per list item |
| Empty section in export | YAML list is empty — add entries or remove section from `resume_sections` |

See [CONFIGURATION.md](CONFIGURATION.md) for section order and [ARCHITECTURE.md](ARCHITECTURE.md) for how selection uses your bullets.
