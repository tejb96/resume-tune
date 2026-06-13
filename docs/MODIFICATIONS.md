# Modification guide

Task-oriented cookbook: what to edit, what to change, and how to verify.

## Change resume section order or omit sections

**File:** `[config.toml](../config.toml)`

**Change:** Edit `resume_sections` list — valid ids: `summary`, `skills`, `experience`, `education`, `projects`, `certifications`.

**Verify:**

```bash
uv run python scripts/smoke_test.py --background background.md --static-only
```

Restart Streamlit after editing config.

---

## Adjust one-page fit aggressiveness

**Files:**

- `[config.toml](../config.toml)` — `max_resume_pages`, `auto_fill_page_budget`, `overflow_warning_min_composite`, `min_project_entries`, `min_project_bullets`
- `[src/resume_tune/content/scoring.py](../src/resume_tune/content/scoring.py)` — trim/expand algorithms, `SelectionPolicy` defaults

**Change:**

- More content on one page → increase `max_resume_pages` or disable `auto_fill_page_budget`
- Keep more projects → raise `min_project_entries` / `min_project_bullets`
- Trim less aggressively → adjust composite thresholds in `scoring.py` (`trim_selection_by_lowest_score`, `expand_selection_by_highest_score`)

**Verify:**

```bash
uv run pytest tests/test_selection.py tests/test_quality_fit.py
```

Requires LibreOffice for PDF trial-render tests; some tests mock PDF output.

---

## Modify LLM prompts or output format

**File:** `[src/resume_tune/llm/ai.py](../src/resume_tune/llm/ai.py)`

**Change:**

- Prompt templates: `SKILLS_LAYOUT_RULES`, `SKILLS_ONLY_OUTPUT_RULES`, `generate_tailored_content`, `revise_tailored_content`
- JSON parsing/repair: `strip_json_fences`, `_parse_ai_json`
- Skills guardrails: `apply_skills_guardrails`, `enforce_skills_layout`

**Verify:**

```bash
uv run pytest tests/test_ai_sections.py tests/test_ai_revision.py
uv run python scripts/e2e_live.py   # when API is running
```

---

## Change job-aware bullet selection prompts

**File:** `[src/resume_tune/llm/selection.py](../src/resume_tune/llm/selection.py)`

**Change:** Rating prompt in `generate_content_selection`, default selection in `default_selection`.

**Verify:**

```bash
uv run pytest tests/test_selection.py tests/test_scoring.py
```

---

## Change DOCX styling (fonts, margins, colors)

**File:** `[src/resume_tune/render/resume.py](../src/resume_tune/render/resume.py)`

**Change:**

- Constants at top: `FONT_NAME`, `FONT_BODY`, `FONT_NAME_SIZE`, `BULLET_TEXT_INDENT`
- Section builders: `_add_section_heading`, `_add_experience_entry`, etc.

**Verify:**

```bash
uv run python scripts/smoke_test.py
uv run pytest tests/test_export.py tests/test_resume_sections.py
```

Open `output/design_preview.docx` or `.html` to inspect visually.

---

## Add ATS keywords or section checks

**File:** `[src/resume_tune/ats/ats.py](../src/resume_tune/ats/ats.py)`

**Change:**

- `TECH_TERMS` — curated technology keyword list
- `detect_sections`, `match_keywords`, `parse_contact_info`
- `analyze_ats_compatibility` scoring weights

**Verify:**

```bash
uv run pytest tests/test_ats.py
uv run python scripts/ats_check.py --jd-file path/to/jd.txt --background background.example.md
```

---

## Change skills packing rules

**Files:**

- `[src/resume_tune/skills/skills_selection.py](../src/resume_tune/skills/skills_selection.py)` — relevance scoring, line packing
- `[src/resume_tune/skills/skills_map.py](../src/resume_tune/skills/skills_map.py)` — map loading, validation
- `[config.toml](../config.toml)` — `max_skill_categories`, `max_skills_per_category`, `max_chars_per_skill_line`

**Verify:**

```bash
uv run pytest tests/test_skills_selection.py tests/test_skills_map.py
```

---

## Add tracker columns

**File:** `[src/resume_tune/tracker/tracker.py](../src/resume_tune/tracker/tracker.py)`

**Change:** Update `HEADERS` list and `log_application()` to write new fields. Update the form in `[app.py](../app.py)` to collect the data.

**Verify:**

```bash
uv run pytest tests/test_tracker.py
```

---

## Change character budget for summary + skills

**Files:**

- `[config.toml](../config.toml)` — `ai_output_max_chars` (runtime default: `600`)
- `[.env](../.env.example)` — `AI_OUTPUT_MAX_CHARS` override
- `[src/resume_tune/llm/ai.py](../src/resume_tune/llm/ai.py)` — `DEFAULT_AI_OUTPUT_MAX_CHARS` (code fallback only: `967`)

**Verify:** Generate a resume and check page fit diagnostics in the sidebar.

See [CONFIGURATION.md](CONFIGURATION.md) for precedence: `.env` → `config.toml` → code default.