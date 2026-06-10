"""Deterministic ATS compatibility: keyword extraction, matching, and parse checks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from resume import (
    ai_section_flags,
    extract_pdf_text,
    flatten_resume_text,
    resolve_resume_sections,
)

# Curated tech terms — longest entries first for greedy matching.
TECH_TERMS: tuple[str, ...] = (
    "Amazon Web Services",
    "AWS Lambda",
    "AWS S3",
    "AWS EC2",
    "AWS IAM",
    "AWS SAA",
    "CloudFront",
    "Route 53",
    "GitHub Actions",
    "GitHub Projects",
    "Google Cloud Platform",
    "Google Cloud",
    "Azure DevOps",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Prompt Engineering",
    "RESTful APIs",
    "REST API",
    "CI/CD",
    "TensorFlow.js",
    "Material UI",
    "Tailwind CSS",
    "Next.js",
    "Node.js",
    "Express.js",
    "React Native",
    "React.js",
    "Vue.js",
    "Angular.js",
    "FastAPI",
    "Django",
    "Flask",
    "Spring Boot",
    "Ruby on Rails",
    "PostgreSQL",
    "MongoDB",
    "Firestore",
    "MySQL",
    "Redis",
    "GraphQL",
    "Kubernetes",
    "Terraform",
    "Prometheus",
    "Grafana",
    "CloudTrail",
    "Databricks",
    "Delta Lake",
    "Unity Catalog",
    "Vector Search",
    "Hugging Face",
    "OpenAI API",
    "TensorFlow",
    "Streamlit",
    "TypeScript",
    "JavaScript",
    "WordPress",
    "HubSpot",
    "LibreOffice",
    "python-docx",
    "MERN stack",
    "MERN",
    "Firebase",
    "Firebase Auth",
    "Firebase Functions",
    "GitHub",
    "GitLab",
    "Docker",
    "GHCR",
    "Ollama",
    "YOLOv8",
    "Agile",
    "Scrum",
    "Jira",
    "Linux",
    "Ubuntu",
    "Windows",
    "macOS",
    "PHP",
    "Python",
    "Java",
    "Ruby",
    "Rust",
    "Swift",
    "Kotlin",
    "Scala",
    "Golang",
    "React",
    "Vue",
    "Angular",
    "Svelte",
    "Next",
    "Node",
    "Express",
    "FastAPI",
    "Rails",
    "Laravel",
    "Spring",
    "Flask",
    "Django",
    "AWS",
    "GCP",
    "Azure",
    "S3",
    "EC2",
    "IAM",
    "Lambda",
    "Docker",
    "Kubernetes",
    "k8s",
    "Terraform",
    "Ansible",
    "Jenkins",
    "CircleCI",
    "Travis CI",
    "Git",
    "SVN",
    "Mercurial",
    "SQL",
    "NoSQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "Elasticsearch",
    "DynamoDB",
    "Cassandra",
    "SQLite",
    "Oracle",
    "SQL Server",
    "Snowflake",
    "BigQuery",
    "Redshift",
    "Spark",
    "Hadoop",
    "Kafka",
    "RabbitMQ",
    "gRPC",
    "GraphQL",
    "REST",
    "SOAP",
    "Microservices",
    "Event-driven",
    "OAuth",
    "JWT",
    "RBAC",
    "HTML",
    "CSS",
    "SASS",
    "LESS",
    "Tailwind",
    "Bootstrap",
    "Material",
    "Shadcn",
    "Redux",
    "Zustand",
    "Webpack",
    "Vite",
    "Babel",
    "ESLint",
    "Prettier",
    "Jest",
    "Pytest",
    "Mocha",
    "Cypress",
    "Selenium",
    "Playwright",
    "Puppeteer",
    "Nginx",
    "Apache",
    "HAProxy",
    "Load Balancing",
    "CDN",
    "DNS",
    "TCP/IP",
    "HTTP",
    "HTTPS",
    "WebSocket",
    "WebSockets",
    "NLP",
    "LLM",
    "GPT",
    "OpenAI",
    "HuggingFace",
    "PyTorch",
    "Keras",
    "Scikit-learn",
    "Pandas",
    "NumPy",
    "Matplotlib",
    "Seaborn",
    "Jupyter",
    "R",
    "MATLAB",
    "C++",
    "C#",
    ".NET",
    "ASP.NET",
    "Go",
    "C",
    "Objective-C",
    "Shell",
    "Bash",
    "PowerShell",
    "Perl",
    "Lua",
    "Dart",
    "Flutter",
    "React Native",
    "Xamarin",
    "Electron",
    "Tauri",
    "Figma",
    "Sketch",
    "Adobe XD",
    "Photoshop",
    "Illustrator",
    "Tableau",
    "Power BI",
    "Looker",
    "Metabase",
    "Superset",
    "Grafana",
    "Prometheus",
    "Datadog",
    "New Relic",
    "Sentry",
    "Splunk",
    "ELK",
    "Logstash",
    "Kibana",
    "OpenTelemetry",
    "Jaeger",
    "Zipkin",
    "Helm",
    "ArgoCD",
    "Istio",
    "Envoy",
    "Consul",
    "Vault",
    "Pulumi",
    "CloudFormation",
    "Serverless",
    "FaaS",
    "PaaS",
    "IaaS",
    "SaaS",
    "Microservices",
    "Monolith",
    "SOA",
    "DDD",
    "TDD",
    "BDD",
    "SOLID",
    "Design Patterns",
    "Algorithms",
    "Data Structures",
    "System Design",
    "Distributed Systems",
    "High Availability",
    "Scalability",
    "Performance",
    "Optimization",
    "Caching",
    "Indexing",
    "Sharding",
    "Replication",
    "Backup",
    "Disaster Recovery",
    "Security",
    "Encryption",
    "SSL",
    "TLS",
    "Penetration Testing",
    "OWASP",
    "SOC2",
    "HIPAA",
    "GDPR",
    "PCI",
    "Compliance",
    "Audit",
    "Monitoring",
    "Logging",
    "Alerting",
    "Incident Response",
    "On-call",
    "SRE",
    "DevOps",
    "DevSecOps",
    "Platform Engineering",
    "Infrastructure",
    "Networking",
    "Virtualization",
    "VMware",
    "Hyper-V",
    "KVM",
    "OpenStack",
    "Proxmox",
    "Rancher",
    "OpenShift",
    "EKS",
    "GKE",
    "AKS",
    "ECS",
    "Fargate",
    "CloudWatch",
    "CloudTrail",
    "VPC",
    "Subnet",
    "NAT",
    "VPN",
    "Firewall",
    "WAF",
    "API Gateway",
    "Load Balancer",
    "Auto Scaling",
    "Spot Instances",
    "Reserved Instances",
    "Cost Optimization",
    "FinOps",
    "Gmail API",
    "Stripe",
    "PayPal",
    "Twilio",
    "SendGrid",
    "Mailchimp",
    "Salesforce",
    "SAP",
    "ERP",
    "CRM",
    "CMS",
    "E-commerce",
    "Shopify",
    "WooCommerce",
    "Magento",
    "BigCommerce",
    "Squarespace",
    "Wix",
    "Drupal",
    "Joomla",
    "Contentful",
    "Sanity",
    "Strapi",
    "Prisma",
    "TypeORM",
    "Sequelize",
    "SQLAlchemy",
    "Hibernate",
    "Entity Framework",
    "ActiveRecord",
    "Drizzle",
    "Knex",
    "Mongoose",
    "Pydantic",
    "Marshmallow",
    "Zod",
    "Joi",
    "Yup",
    "Formik",
    "React Hook Form",
    "Storybook",
    "Chromatic",
    "Cypress",
    "Vitest",
    "Testing Library",
    "JUnit",
    "TestNG",
    "Mockito",
    "RSpec",
    "Cucumber",
    "Gherkin",
    "Postman",
    "Insomnia",
    "Swagger",
    "OpenAPI",
    "AsyncAPI",
    "Protocol Buffers",
    "Avro",
    "Parquet",
    "ORC",
    "CSV",
    "JSON",
    "XML",
    "YAML",
    "TOML",
    "INI",
    "Markdown",
    "LaTeX",
    "AsciiDoc",
    "reStructuredText",
    "Sphinx",
    "MkDocs",
    "Docusaurus",
    "Gatsby",
    "Hugo",
    "Jekyll",
    "Eleventy",
    "Astro",
    "Remix",
    "SvelteKit",
    "Nuxt",
    "Nuxt.js",
    "SolidJS",
    "Qwik",
    "Lit",
    "Stencil",
    "Web Components",
    "PWA",
    "Service Worker",
    "IndexedDB",
    "LocalStorage",
    "SessionStorage",
    "Cookies",
    "CORS",
    "CSRF",
    "XSS",
    "SQL Injection",
    "Authentication",
    "Authorization",
    "SSO",
    "SAML",
    "OIDC",
    "LDAP",
    "Active Directory",
    "Kerberos",
    "MFA",
    "2FA",
    "Biometrics",
    "Blockchain",
    "Ethereum",
    "Solidity",
    "Web3",
    "Smart Contracts",
    "NFT",
    "DeFi",
    "Cryptocurrency",
    "Bitcoin",
    "Hyperledger",
    "Cordova",
    "Capacitor",
    "Ionic",
    "Expo",
    "React Navigation",
    "Redux Toolkit",
    "RTK Query",
    "TanStack Query",
    "SWR",
    "Apollo",
    "Relay",
    "urql",
    "tRPC",
    "Socket.io",
    "Pusher",
    "Ably",
    "Firebase Realtime",
    "Supabase",
    "shadcn/ui",
    "Vercel",
    "Coolify",
    "Cursor",
    "Codex",
    "Claude Code",
    "GitHub Copilot",
    "Appwrite",
    "PocketBase",
    "PlanetScale",
    "Neon",
    "CockroachDB",
    "TimescaleDB",
    "InfluxDB",
    "Timeseries",
    "Grafana Loki",
    "Thanos",
    "VictoriaMetrics",
    "Mimir",
    "Cortex",
    "Tempo",
    "Loki",
    "Fluentd",
    "Fluent Bit",
    "Vector",
    "Filebeat",
    "Metricbeat",
    "Heartbeat",
    "Auditbeat",
    "Packetbeat",
    "Winlogbeat",
    "Functionbeat",
    "Journalbeat",
    "Osquerybeat",
)

# Deduplicate and sort by length descending for longest-match-first.
_seen: set[str] = set()
_SORTED_TECH_TERMS: tuple[str, ...] = tuple(
    term
    for term in sorted(set(TECH_TERMS), key=lambda t: (-len(t), t.lower()))
    if not (term.lower() in _seen or _seen.add(term.lower()))  # type: ignore[func-returns-value]
)

# Canonical form for deduplication during extraction.
_CANONICAL: dict[str, str] = {}
for term in _SORTED_TECH_TERMS:
    _CANONICAL[term.lower()] = term

# Aliases expand matching (not extraction).
MATCH_ALIASES: dict[str, list[str]] = {
    "kubernetes": ["k8s", "kube"],
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "terraform": ["tf", "iac"],
    "github actions": ["gh actions", "gha"],
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp", "google cloud"],
    "microsoft azure": ["azure"],
    "postgresql": ["postgres", "psql"],
    "machine learning": ["ml"],
    "natural language processing": ["nlp"],
    "continuous integration": ["ci"],
    "continuous deployment": ["cd"],
    "ci/cd": ["cicd", "ci cd"],
    "react": ["reactjs", "react.js"],
    "node.js": ["nodejs", "node"],
    "next.js": ["nextjs"],
    "vue.js": ["vuejs"],
    "express.js": ["expressjs"],
    "tensorflow.js": ["tfjs"],
    "hugging face": ["huggingface", "hf"],
    "openai api": ["openai"],
    "mern stack": ["mern"],
    "restful apis": ["rest api", "rest apis", "rest"],
    "github": ["gh"],
    "docker": ["containerization", "containers"],
    "golang": ["go"],
    "tailwind css": ["tailwind"],
    "tailwind": ["tailwind css"],
    "shadcn/ui": ["shadcn", "shadcn ui"],
    "javascript/typescript": ["javascript", "typescript"],
    "openai apis": ["openai api", "openai"],
    "vercel": ["vercel deployment"],
    "github copilot": ["copilot"],
}

# Short tokens that need word-boundary matching.
_WORD_BOUNDARY_TERMS: frozenset[str] = frozenset(
    {"c", "r", "go", "d", "f", "sql", "ml", "ai", "ui", "ux", "os", "io", "js", "ts"}
)

SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "summary": re.compile(r"\bPROFESSIONAL\s+SUMMARY\b", re.IGNORECASE),
    "skills": re.compile(r"\bSKILLS\b", re.IGNORECASE),
    "experience": re.compile(r"\bEXPERIENCE\b", re.IGNORECASE),
    "education": re.compile(r"\bEDUCATION\b", re.IGNORECASE),
    "projects": re.compile(r"\bPROJECTS\b", re.IGNORECASE),
    "certifications": re.compile(r"\bCERTIFICATIONS\b", re.IGNORECASE),
}

EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{4}",
)
LINKEDIN_RE = re.compile(r"linkedin", re.IGNORECASE)
GITHUB_RE = re.compile(r"github", re.IGNORECASE)

PATTERN_TERMS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bC\+\+\b", re.IGNORECASE), "C++"),
    (re.compile(r"\bC#\b", re.IGNORECASE), "C#"),
    (re.compile(r"\b\.NET\b", re.IGNORECASE), ".NET"),
    (re.compile(r"\bPython\s*3(?:\.\d+)?\b", re.IGNORECASE), "Python"),
    (re.compile(r"\bJava\s*(?:8|11|17|21)\b", re.IGNORECASE), "Java"),
    (re.compile(r"\bNode\.js\b", re.IGNORECASE), "Node.js"),
    (re.compile(r"\bReact\.js\b", re.IGNORECASE), "React"),
    (re.compile(r"\bVue\.js\b", re.IGNORECASE), "Vue.js"),
    (re.compile(r"\bNext\.js\b", re.IGNORECASE), "Next.js"),
    (re.compile(r"\bshadcn/ui\b", re.IGNORECASE), "shadcn/ui"),
    (re.compile(r"\bJavaScript/TypeScript\b", re.IGNORECASE), "TypeScript"),
    (re.compile(r"\bJavaScript\s*/\s*TypeScript\b", re.IGNORECASE), "TypeScript"),
)

@dataclass(frozen=True)
class ContactInfo:
    email_found: bool
    phone_found: bool
    linkedin_found: bool
    github_found: bool
    email_matches_expected: bool
    phone_matches_expected: bool
    expected_email: str | None
    expected_phone: str | None


@dataclass(frozen=True)
class PdfFidelity:
    name_in_pdf: bool
    email_in_pdf: bool
    sections_in_pdf: dict[str, bool]
    keyword_parity: dict[str, bool]
    fidelity_score: float
    checks_passed: int
    checks_total: int


@dataclass(frozen=True)
class AtsReport:
    jd_keywords: tuple[str, ...]
    matched_keywords: tuple[str, ...]
    missing_keywords: tuple[str, ...]
    keyword_match_pct: float
    sections_found: tuple[str, ...]
    sections_missing: tuple[str, ...]
    contact: ContactInfo
    pdf_fidelity: PdfFidelity | None
    resume_text_source: Literal["pdf", "flattened"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(text: str) -> str:
    text = text.replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _canonical_keyword(term: str) -> str:
    return _CANONICAL.get(term.lower(), term)


def _term_in_text(term: str, text: str, *, normalized: str | None = None) -> bool:
    """Return True when a curated tech term appears in text (word boundaries for single tokens)."""
    norm = normalized if normalized is not None else _normalize_text(text)
    pattern = re.escape(term.lower())
    if " " in term:
        return bool(re.search(pattern, norm, re.IGNORECASE))
    if term.lower() in _WORD_BOUNDARY_TERMS:
        return bool(re.search(rf"\b{pattern}\b", norm, re.IGNORECASE))
    return bool(re.search(rf"\b{pattern}\b", norm, re.IGNORECASE))


def _extract_tech_terms_from_text(text: str) -> list[str]:
    """Extract curated tech terms from arbitrary text (longest-match-first)."""
    if not text.strip():
        return []

    normalized = _normalize_text(text)
    found: dict[str, str] = {}

    for term in _SORTED_TECH_TERMS:
        if _term_in_text(term, text, normalized=normalized):
            canonical = _canonical_keyword(term)
            found[canonical.lower()] = canonical

    for regex, label in PATTERN_TERMS:
        if regex.search(text):
            canonical = _canonical_keyword(label)
            found[canonical.lower()] = canonical

    return sorted(found.values(), key=str.lower)


def extract_jd_keywords(job_description: str) -> list[str]:
    """Extract tech keywords from a job description deterministically."""
    return _extract_tech_terms_from_text(job_description)


def _prefer_jd_skill_labels(
    skills: list[str],
    jd_keywords: list[str],
    background_text: str,
) -> list[str]:
    """Prefer JD-matching labels when multiple evidenced forms exist (e.g. Tailwind CSS vs Tailwind)."""
    jd_lower = {k.lower() for k in jd_keywords}
    bg_lower = background_text.lower()
    drop: set[str] = set()

    if "tailwind css" in jd_lower and "tailwind css" in bg_lower:
        drop.add("tailwind")

    return [s for s in skills if s.lower() not in drop]


def extract_evidenced_skills(
    background_text: str,
    *,
    jd_keywords: list[str] | None = None,
) -> list[str]:
    """Extract tech terms evidenced in background text using word-boundary matching."""
    skills = _extract_tech_terms_from_text(background_text)
    if jd_keywords:
        skills = _prefer_jd_skill_labels(skills, jd_keywords, background_text)
    return skills


def _expand_aliases(keyword: str) -> list[str]:
    """Return keyword plus known aliases for matching."""
    lower = keyword.lower()
    variants = [lower]
    for canonical, aliases in MATCH_ALIASES.items():
        if lower == canonical or lower in aliases:
            variants.append(canonical)
            variants.extend(aliases)
    return list(dict.fromkeys(variants))


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Check if keyword (or alias) appears in text."""
    normalized = _normalize_text(text)
    for variant in _expand_aliases(keyword):
        escaped = re.escape(variant.lower())
        if variant.lower() in _WORD_BOUNDARY_TERMS or len(variant) <= 2:
            if re.search(rf"\b{escaped}\b", normalized, re.IGNORECASE):
                return True
        elif " " in variant:
            if escaped in normalized:
                return True
        else:
            if re.search(rf"\b{escaped}\b", normalized, re.IGNORECASE):
                return True
    return False


def match_keywords(
    keywords: list[str],
    resume_text: str,
) -> tuple[list[str], list[str]]:
    """Return (matched, missing) keyword lists."""
    if not keywords:
        return [], []

    matched: list[str] = []
    missing: list[str] = []
    for keyword in keywords:
        if _keyword_in_text(keyword, resume_text):
            matched.append(keyword)
        else:
            missing.append(keyword)
    return matched, missing


def detect_sections(
    resume_text: str,
    expected_sections: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Detect section headings in resume text; return (found, missing)."""
    found: list[str] = []
    for section_id, pattern in SECTION_PATTERNS.items():
        if pattern.search(resume_text):
            found.append(section_id)

    if expected_sections is None:
        return found, []

    expected = resolve_resume_sections(expected_sections)
    missing = [s for s in expected if s not in found]
    return found, missing


def _normalize_phone(phone: str) -> str:
    return re.sub(r"\D", "", phone)


def parse_contact_info(
    resume_text: str,
    header: dict[str, Any] | None = None,
) -> ContactInfo:
    """Parse contact fields from resume text and compare to expected header."""
    expected_email = (header or {}).get("email")
    expected_phone = (header or {}).get("phone")

    email_found = bool(EMAIL_RE.search(resume_text))
    phone_found = bool(PHONE_RE.search(resume_text))
    linkedin_found = bool(LINKEDIN_RE.search(resume_text))
    github_found = bool(GITHUB_RE.search(resume_text))

    email_matches = False
    if expected_email and email_found:
        email_matches = expected_email.lower() in resume_text.lower()

    phone_matches = False
    if expected_phone and phone_found:
        expected_digits = _normalize_phone(expected_phone)
        text_digits_chunks = re.findall(r"\d[\d\s().\-]{8,}\d", resume_text)
        phone_matches = any(
            _normalize_phone(chunk) == expected_digits
            or expected_digits in _normalize_phone(chunk)
            or _normalize_phone(chunk) in expected_digits
            for chunk in text_digits_chunks
        )

    return ContactInfo(
        email_found=email_found,
        phone_found=phone_found,
        linkedin_found=linkedin_found,
        github_found=github_found,
        email_matches_expected=email_matches,
        phone_matches_expected=phone_matches,
        expected_email=expected_email,
        expected_phone=expected_phone,
    )


def compare_pdf_fidelity(
    *,
    pdf_text: str,
    flattened_text: str,
    header: dict[str, Any],
    expected_sections: list[str] | None,
    matched_keywords: list[str],
) -> PdfFidelity:
    """Compare PDF-extracted text against expected structure from flattened render."""
    checks: list[bool] = []
    sections_in_pdf: dict[str, bool] = {}
    keyword_parity: dict[str, bool] = {}

    name = header.get("name", "")
    if name:
        name_in_pdf = name.lower() in pdf_text[:500].lower()
    else:
        name_in_pdf = False
    checks.append(name_in_pdf)

    email = header.get("email", "")
    if email:
        email_in_pdf = email.lower() in pdf_text.lower()
    else:
        email_in_pdf = True
    checks.append(email_in_pdf)

    resolved = resolve_resume_sections(expected_sections)
    for section_id in resolved:
        pattern = SECTION_PATTERNS.get(section_id)
        if pattern:
            found = bool(pattern.search(pdf_text))
            sections_in_pdf[section_id] = found
            checks.append(found)

    for keyword in matched_keywords:
        in_pdf = _keyword_in_text(keyword, pdf_text)
        in_flat = _keyword_in_text(keyword, flattened_text)
        parity = in_pdf == in_flat and in_pdf
        keyword_parity[keyword] = parity
        if matched_keywords:
            checks.append(parity)

    passed = sum(1 for c in checks if c)
    total = len(checks)
    score = (passed / total * 100.0) if total else 100.0

    return PdfFidelity(
        name_in_pdf=name_in_pdf,
        email_in_pdf=email_in_pdf,
        sections_in_pdf=sections_in_pdf,
        keyword_parity=keyword_parity,
        fidelity_score=round(score, 1),
        checks_passed=passed,
        checks_total=total,
    )


def analyze_ats_compatibility(
    *,
    job_description: str,
    background_data: dict[str, Any],
    ai_output: dict[str, Any],
    pdf_bytes: bytes | None,
    sections: list[str] | None,
    content_selection: dict[str, Any] | None,
    max_certifications: int | None,
) -> AtsReport | None:
    """Run full ATS analysis. Returns None when job description is empty."""
    if not job_description.strip():
        return None

    flattened = flatten_resume_text(
        background_data,
        ai_output,
        sections=sections,
        content_selection=content_selection,
        max_certifications=max_certifications,
    )

    pdf_text: str | None = None
    if pdf_bytes:
        pdf_text = extract_pdf_text(pdf_bytes)

    if pdf_text:
        resume_text = pdf_text
        text_source: Literal["pdf", "flattened"] = "pdf"
    else:
        resume_text = flattened
        text_source = "flattened"

    jd_keywords = extract_jd_keywords(job_description)
    matched, missing = match_keywords(jd_keywords, resume_text)

    if jd_keywords:
        keyword_match_pct = round(len(matched) / len(jd_keywords) * 100.0, 1)
    else:
        keyword_match_pct = 0.0

    sections_found, sections_missing = detect_sections(resume_text, sections)
    contact = parse_contact_info(resume_text, background_data.get("header"))

    pdf_fidelity: PdfFidelity | None = None
    if pdf_text:
        pdf_fidelity = compare_pdf_fidelity(
            pdf_text=pdf_text,
            flattened_text=flattened,
            header=background_data.get("header", {}),
            expected_sections=sections,
            matched_keywords=matched,
        )

    return AtsReport(
        jd_keywords=tuple(jd_keywords),
        matched_keywords=tuple(matched),
        missing_keywords=tuple(missing),
        keyword_match_pct=keyword_match_pct,
        sections_found=tuple(sections_found),
        sections_missing=tuple(sections_missing),
        contact=contact,
        pdf_fidelity=pdf_fidelity,
        resume_text_source=text_source,
    )
