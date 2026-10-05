"""Build synthetic claim and evidence rows.

Labels are assigned from the rules in docs/labels.md.
The same seed rebuilds the same rows.
"""

from __future__ import annotations

import random
import re

from cesc.labels import LABELS
from cesc.prose import assert_no_dash

PAIRS = [
    ("PostgreSQL", "MySQL"),
    ("AWS", "GCP"),
    ("Docker", "Kubernetes"),
    ("React", "Angular"),
    ("Python", "Java"),
    ("TensorFlow", "PyTorch"),
    ("Redis", "Memcached"),
    ("Kafka", "RabbitMQ"),
    ("Snowflake", "BigQuery"),
    ("Terraform", "CloudFormation"),
    ("Jenkins", "GitHub Actions"),
    ("MongoDB", "DynamoDB"),
    ("Vue", "Svelte"),
    ("Spark", "Flink"),
    ("Elasticsearch", "OpenSearch"),
    ("TypeScript", "JavaScript"),
    ("Go", "Rust"),
    ("Airflow", "Luigi"),
    ("Tableau", "Power BI"),
    ("Nginx", "Apache"),
    ("FastAPI", "Spring Boot"),
    ("Pandas", "Polars"),
    ("Ansible", "Chef"),
    ("Prometheus", "Datadog"),
    ("SageMaker", "Vertex AI"),
    ("Lambda", "Cloud Functions"),
    ("S3", "Cloud Storage"),
    ("Pytest", "JUnit"),
    ("Celery", "Sidekiq"),
    ("dbt", "Dataform"),
    ("GraphQL", "REST"),
    ("Linux", "Windows Server"),
    ("Scala", "Kotlin"),
    ("XGBoost", "LightGBM"),
    ("Redshift", "Azure Synapse"),
]

ALIASES = {
    "PostgreSQL": "Postgres",
    "AWS": "Amazon Web Services",
    "GCP": "Google Cloud",
    "JavaScript": "JS",
    "TypeScript": "TS",
    "MongoDB": "Mongo",
    "Kubernetes": "K8s",
    "DynamoDB": "Dynamo",
    "Elasticsearch": "Elastic Search",
}

COMPANIES = [
    "Northwind Labs",
    "Harbor Analytics",
    "Brightline Health",
    "Copper Street Bank",
    "Lumen Retail",
    "Fieldnote Media",
    "Redwood Logistics",
    "Aster Payroll",
    "Summit Transit",
    "Kindred Insurance",
    "Marble Education",
    "Oakline Energy",
]

ROLES = [
    "backend engineer",
    "data engineer",
    "platform engineer",
    "machine learning engineer",
    "frontend engineer",
    "site reliability engineer",
    "analytics engineer",
    "infrastructure engineer",
]

SKILLS = [
    ("mentoring junior engineers", "mentored junior engineers on the team"),
    ("owning on call rotations", "owned the on call rotation"),
    ("writing design documents for the team", "wrote the design documents the team used"),
    ("leading incident reviews", "led incident reviews after failures"),
    ("building a training program for new hires", "built the training program for new hires"),
    ("setting coding standards for the group", "set the coding standards the group follows"),
    ("leading the migration plan", "led the migration plan off the old stack"),
    ("owning capacity planning", "owned capacity planning for the service"),
]

SKILL_MARKERS = [
    "mentoring",
    "mentored",
    "on call",
    "design document",
    "incident review",
    "training program",
    "coding standard",
    "migration plan",
    "capacity planning",
]

SINGLE_CLAIMS = [
    "The candidate must have production experience with {tech}.",
    "This role requires {tech} on a live product.",
    "Required: operate {tech} under live traffic.",
    "The hire must have shipped {tech} to real users.",
    "We need production ownership of {tech}.",
    "The candidate has to run {tech} for a live product.",
]

YEARS_CLAIMS = [
    "At least {req} years of {tech} experience.",
    "The candidate must have {req} or more years with {tech}.",
    "Required tenure: {req} years using {tech}.",
    "Need {req} years of practical {tech} work.",
]

TWO_PART_CLAIMS = [
    "The candidate needs {tech} experience and {skill}.",
    "Required: {tech} plus {skill}.",
    "This role needs {tech} together with {skill}.",
    "Must have {tech} and {skill}.",
]

WEAK_EVIDENCE = [
    "Finished an online course on {mention} and a short practice lab.",
    "Used {mention} in a university class project with sample data.",
    "Built a weekend prototype with {mention} on sample data.",
    "Read the {mention} docs and passed an internal quiz.",
    "Joined a hackathon team and showed a short {mention} demo.",
]

WEAK_MARKERS = [
    "online course",
    "class project",
    "weekend prototype",
    "internal quiz",
    "hackathon",
]

STRONG_MARKERS = [
    "production",
    "live product",
    "live traffic",
    "shipped",
    "real users",
]

NEGATION_CUES = ["rejected", "turned down", "never used", "did not adopt"]

UNRELATED = [
    "Managed a regional sales quota and renewed six enterprise contracts.",
    "Taught evening painting classes and sold prints at a weekend market.",
    "Prepared monthly payroll for a staff of forty and filed the tax forms.",
    "Led a community garden and scheduled volunteers for Saturday work.",
    "Wrote a weekly food column and edited restaurant notes for the paper.",
    "Coordinated freight deliveries between two warehouses and the shop floor.",
    "Ran the front desk at a clinic and booked patient visits.",
    "Designed costumes for a local theater season and fitted the cast.",
    "Handled donor calls for a charity and logged gifts in a spreadsheet.",
    "Drove a school bus route and completed the daily safety checklist.",
    "Baked bread for three cafes and ordered flour from local farms.",
    "Staffed a hotel reception desk and resolved late booking complaints.",
    "Coached a youth soccer team and set the weekend practice times.",
    "Catalogued donated books for a neighborhood library.",
    "Planned seating and menus for banquet events at a city hall.",
    "Repaired bicycles for commuters and kept the shop schedule.",
]

QUOTAS = [
    ("supported_single", 160),
    ("supported_years", 120),
    ("supported_two_part", 120),
    ("partial_years", 140),
    ("partial_missing_part", 140),
    ("partial_weaker", 120),
    ("not_near_miss", 180),
    ("not_years_on_other_tech", 80),
    ("not_negated", 70),
    ("not_unrelated", 70),
]

LABEL_OF = {
    "supported_single": "Supported",
    "supported_years": "Supported",
    "supported_two_part": "Supported",
    "partial_years": "Partially Supported",
    "partial_missing_part": "Partially Supported",
    "partial_weaker": "Partially Supported",
    "not_near_miss": "Not Evidenced",
    "not_years_on_other_tech": "Not Evidenced",
    "not_negated": "Not Evidenced",
    "not_unrelated": "Not Evidenced",
}


def mentions(text: str, name: str) -> bool:
    if not name:
        return False
    pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
    return re.search(pattern, text) is not None


def mentions_any(text: str, names) -> bool:
    return any(mentions(text, name) for name in names)


def has_number(text: str, number: int) -> bool:
    return re.search(rf"(?<!\d){number}(?!\d)", text) is not None


def vocab_names():
    names = set()
    for left, right in PAIRS:
        names.add(left)
        names.add(right)
    names.update(ALIASES.values())
    return sorted(names, key=len, reverse=True)


VOCAB = vocab_names()


def pick_pair(index: int):
    left, right = PAIRS[index % len(PAIRS)]
    if (index // len(PAIRS)) % 2 == 1:
        return right, left
    return left, right


def surface(rng: random.Random, name: str, rate: float) -> str:
    alias = ALIASES.get(name)
    if alias and rng.random() < rate:
        return alias
    return name


def single_claim_at(index: int, tech: str) -> str:
    return SINGLE_CLAIMS[index % len(SINGLE_CLAIMS)].format(tech=tech)


def years_claim_at(index: int, tech: str, req: int) -> str:
    return YEARS_CLAIMS[index % len(YEARS_CLAIMS)].format(tech=tech, req=req)


def two_part_claim_at(index: int, tech: str, skill: str) -> str:
    return TWO_PART_CLAIMS[index % len(TWO_PART_CLAIMS)].format(tech=tech, skill=skill)


def skill_at(index: int):
    return SKILLS[index % len(SKILLS)]


def year_numbers(index: int, kind: str):
    req = 4 + (index % 5)
    if kind == "supported":
        actual = req + (index % 3)
    elif kind == "partial":
        actual = req - 2 - (index % 2)
        if actual < 2:
            actual = 2
    else:
        actual = req
    return req, actual


def production_evidence(rng, company, role, mention, years) -> str:
    frames = [
        f"At {company}, the {role} ran {mention} in production for {years} years and owned live incidents.",
        f"The {role} at {company} shipped {mention} to real users and owned releases for {years} years.",
        f"{company} used {mention} for live traffic. The {role} handled failures and fixes for {years} years.",
        f"For {years} years, the {role} at {company} kept {mention} running for real users.",
        f"The production stack at {company} centered on {mention}. The {role} owned it for {years} years.",
        f"Daily work for the {role} at {company} was {mention}, including live incidents over {years} years.",
    ]
    return rng.choice(frames)


def years_evidence(rng, company, role, mention, years) -> str:
    frames = [
        f"The {role} used {mention} for {years} years at {company}.",
        f"At {company}, the {role} spent {years} years working with {mention}.",
        f"{mention} was the daily tool for this {role} for {years} years at {company}.",
        f"Resume line from {company}: {years} years with {mention} as the {role}.",
    ]
    return rng.choice(frames)


def two_part_evidence(rng, company, role, mention, years, evidence_phrase) -> str:
    frames = [
        f"At {company}, the {role} used {mention} for {years} years and {evidence_phrase}.",
        f"The {role} at {company} used {mention} in production and {evidence_phrase}.",
        f"{mention} was the core tool at {company}. The {role} {evidence_phrase}.",
    ]
    return rng.choice(frames)


def weak_evidence(rng, mention) -> str:
    return rng.choice(WEAK_EVIDENCE).format(mention=mention)


def negation_evidence(rng, company, role, tech, near) -> str:
    frames = [
        f"The team at {company} reviewed {tech} and rejected it. The {role} stayed on {near}.",
        f"A short trial of {tech} was turned down. The {role} at {company} kept using {near}.",
        f"{tech} was never used in production at {company}. The live stack was {near}.",
        f"The {role} did not adopt {tech}. Work at {company} stayed on {near}.",
    ]
    return rng.choice(frames)


def unrelated_evidence(rng, company) -> str:
    base = rng.choice(UNRELATED)
    return f"{base} The employer was {company}."


def context(rng):
    return rng.choice(COMPANIES), rng.choice(ROLES)


def covers_name(text: str, name: str) -> bool:
    if mentions(text, name):
        return True
    alias = ALIASES.get(name)
    return bool(alias and mentions(text, alias))


def check_common(row: dict) -> None:
    claim = row["claim"]
    evidence = row["evidence"]
    assert_no_dash(claim, "claim")
    assert_no_dash(evidence, "evidence")
    if not claim.endswith(".") or not evidence.endswith("."):
        raise ValueError("missing period: " + claim)
    if len(claim.split()) > 42 or len(evidence.split()) > 48:
        raise ValueError("too many words: " + claim + " || " + evidence)
    if not covers_name(claim, row["tech"]):
        raise ValueError("claim missing tech: " + claim)


def validate_example(row: dict) -> None:
    check_common(row)
    pattern = row["pattern"]
    evidence = row["evidence"]
    claim = row["claim"]
    tech = row["tech"]
    near = row["near"]
    ev_tech = covers_name(evidence, tech)
    ev_near = covers_name(evidence, near)
    positive = {
        "supported_single",
        "supported_years",
        "supported_two_part",
        "partial_years",
        "partial_missing_part",
        "partial_weaker",
    }
    if pattern in positive and not ev_tech:
        raise ValueError("evidence missing claimed tech: " + evidence)
    if pattern in positive and ev_near:
        raise ValueError("evidence also names the near tech: " + evidence)
    if pattern in {"supported_years", "partial_years", "not_years_on_other_tech"}:
        if not has_number(claim, row["req"]):
            raise ValueError("claim missing required years: " + claim)
    if pattern == "supported_years":
        if not has_number(evidence, row["actual"]) or row["actual"] < row["req"]:
            raise ValueError("supported years mismatch: " + evidence)
        if row["actual"] != row["req"] and has_number(evidence, row["req"]):
            raise ValueError("evidence repeats the required years: " + evidence)
    if pattern == "partial_years":
        if not (1 <= row["actual"] < row["req"]):
            raise ValueError("partial years numbers are wrong")
        if not has_number(evidence, row["actual"]) or has_number(evidence, row["req"]):
            raise ValueError("partial years text is wrong: " + evidence)
    if pattern in {"supported_two_part", "partial_missing_part"} or row.get("two_part"):
        if row["claim_phrase"] not in claim:
            raise ValueError("claim missing the second part: " + claim)
    if pattern == "supported_two_part":
        if row["evidence_phrase"] not in evidence:
            raise ValueError("evidence missing the second part: " + evidence)
    if pattern == "partial_missing_part":
        if any(marker in evidence for marker in SKILL_MARKERS):
            raise ValueError("partial evidence leaks a second skill: " + evidence)
    if pattern == "partial_weaker":
        if not any(marker in evidence for marker in WEAK_MARKERS):
            raise ValueError("weak evidence has no weak marker: " + evidence)
        if not any(marker in claim for marker in STRONG_MARKERS):
            raise ValueError("weak claim has no strong marker: " + claim)
        blocked = ["production", "live product", "live traffic", "shipped", "real users"]
        if any(marker in evidence for marker in blocked):
            raise ValueError("weak evidence sounds like production: " + evidence)
    if pattern in {"not_near_miss", "not_years_on_other_tech"}:
        if not ev_near or ev_tech:
            raise ValueError("near miss evidence is not a clean miss: " + evidence)
    if pattern == "not_years_on_other_tech":
        if not has_number(evidence, row["req"]):
            raise ValueError("year trap missing the number: " + evidence)
    if pattern == "not_negated":
        if not ev_tech or not ev_near:
            raise ValueError("negation should name both tools: " + evidence)
        if not any(cue in evidence for cue in NEGATION_CUES):
            raise ValueError("negation missing a cue: " + evidence)
    if pattern == "not_unrelated":
        if mentions_any(evidence, VOCAB):
            raise ValueError("unrelated evidence names a tool: " + evidence)
        if any(marker in evidence for marker in SKILL_MARKERS):
            raise ValueError("unrelated evidence names a skill: " + evidence)


def build_supported_single(rng, index):
    tech, near = pick_pair(index)
    company, role = context(rng)
    years = 2 + (index % 7)
    claim = single_claim_at(index, tech)
    evidence = production_evidence(rng, company, role, surface(rng, tech, 0.7), years)
    return {
        "claim": claim,
        "evidence": evidence,
        "label": "Supported",
        "pattern": "supported_single",
        "tech": tech,
        "near": near,
    }


def build_supported_years(rng, index):
    tech, near = pick_pair(index)
    company, role = context(rng)
    req, actual = year_numbers(index, "supported")
    claim = years_claim_at(index, tech, req)
    evidence = years_evidence(rng, company, role, surface(rng, tech, 0.7), actual)
    return {
        "claim": claim,
        "evidence": evidence,
        "label": "Supported",
        "pattern": "supported_years",
        "tech": tech,
        "near": near,
        "req": req,
        "actual": actual,
    }


def build_supported_two_part(rng, index):
    tech, near = pick_pair(index)
    company, role = context(rng)
    claim_phrase, evidence_phrase = skill_at(index)
    years = 3 + (index % 6)
    claim = two_part_claim_at(index, tech, claim_phrase)
    evidence = two_part_evidence(
        rng, company, role, surface(rng, tech, 0.7), years, evidence_phrase
    )
    return {
        "claim": claim,
        "evidence": evidence,
        "label": "Supported",
        "pattern": "supported_two_part",
        "tech": tech,
        "near": near,
        "claim_phrase": claim_phrase,
        "evidence_phrase": evidence_phrase,
        "two_part": True,
    }


def build_partial_years(rng, index):
    tech, near = pick_pair(index)
    company, role = context(rng)
    req, actual = year_numbers(index, "partial")
    claim = years_claim_at(index, tech, req)
    evidence = years_evidence(rng, company, role, surface(rng, tech, 0.7), actual)
    return {
        "claim": claim,
        "evidence": evidence,
        "label": "Partially Supported",
        "pattern": "partial_years",
        "tech": tech,
        "near": near,
        "req": req,
        "actual": actual,
    }


def build_partial_missing(rng, index):
    tech, near = pick_pair(index)
    company, role = context(rng)
    claim_phrase, evidence_phrase = skill_at(index)
    years = 2 + (index % 7)
    claim = two_part_claim_at(index, tech, claim_phrase)
    evidence = production_evidence(rng, company, role, surface(rng, tech, 0.7), years)
    return {
        "claim": claim,
        "evidence": evidence,
        "label": "Partially Supported",
        "pattern": "partial_missing_part",
        "tech": tech,
        "near": near,
        "claim_phrase": claim_phrase,
        "evidence_phrase": evidence_phrase,
        "two_part": True,
    }


def build_partial_weaker(rng, index):
    tech, near = pick_pair(index)
    claim = single_claim_at(index, tech)
    evidence = weak_evidence(rng, surface(rng, tech, 0.7))
    return {
        "claim": claim,
        "evidence": evidence,
        "label": "Partially Supported",
        "pattern": "partial_weaker",
        "tech": tech,
        "near": near,
    }


def build_not_near(rng, index):
    tech, near = pick_pair(index)
    company, role = context(rng)
    years = 2 + (index % 7)
    claim_phrase = None
    if index % 2 == 0:
        claim = single_claim_at(index, tech)
        two_part = False
    else:
        claim_phrase, _evidence_phrase = skill_at(index)
        claim = two_part_claim_at(index, tech, claim_phrase)
        two_part = True
    evidence = production_evidence(rng, company, role, surface(rng, near, 0.5), years)
    row = {
        "claim": claim,
        "evidence": evidence,
        "label": "Not Evidenced",
        "pattern": "not_near_miss",
        "tech": tech,
        "near": near,
        "two_part": two_part,
    }
    if claim_phrase:
        row["claim_phrase"] = claim_phrase
    return row


def build_not_years_other(rng, index):
    tech, near = pick_pair(index)
    company, role = context(rng)
    req, actual = year_numbers(index, "trap")
    claim = years_claim_at(index, tech, req)
    evidence = years_evidence(rng, company, role, surface(rng, near, 0.5), actual)
    return {
        "claim": claim,
        "evidence": evidence,
        "label": "Not Evidenced",
        "pattern": "not_years_on_other_tech",
        "tech": tech,
        "near": near,
        "req": req,
        "actual": actual,
    }


def build_not_negated(rng, index):
    tech, near = pick_pair(index)
    company, role = context(rng)
    if index % 3 == 0:
        claim_phrase, _evidence_phrase = skill_at(index)
        claim = two_part_claim_at(index, tech, claim_phrase)
        two_part = True
    else:
        claim_phrase = None
        claim = single_claim_at(index, tech)
        two_part = False
    evidence = negation_evidence(rng, company, role, tech, near)
    row = {
        "claim": claim,
        "evidence": evidence,
        "label": "Not Evidenced",
        "pattern": "not_negated",
        "tech": tech,
        "near": near,
        "two_part": two_part,
    }
    if claim_phrase:
        row["claim_phrase"] = claim_phrase
    return row


def build_not_unrelated(rng, index):
    tech, near = pick_pair(index)
    company, _role = context(rng)
    claim_phrase = None
    if index % 2 == 0:
        claim = single_claim_at(index, tech)
        two_part = False
    else:
        claim_phrase, _evidence_phrase = skill_at(index)
        claim = two_part_claim_at(index, tech, claim_phrase)
        two_part = True
    evidence = unrelated_evidence(rng, company)
    row = {
        "claim": claim,
        "evidence": evidence,
        "label": "Not Evidenced",
        "pattern": "not_unrelated",
        "tech": tech,
        "near": near,
        "two_part": two_part,
    }
    if claim_phrase:
        row["claim_phrase"] = claim_phrase
    return row


BUILDERS = {
    "supported_single": build_supported_single,
    "supported_years": build_supported_years,
    "supported_two_part": build_supported_two_part,
    "partial_years": build_partial_years,
    "partial_missing_part": build_partial_missing,
    "partial_weaker": build_partial_weaker,
    "not_near_miss": build_not_near,
    "not_years_on_other_tech": build_not_years_other,
    "not_negated": build_not_negated,
    "not_unrelated": build_not_unrelated,
}


def quotas_for(total: int):
    if total % 3 != 0:
        raise SystemExit("total must be divisible by 3 so the labels can be balanced")
    if total == 1200:
        return list(QUOTAS)
    per = total // 3
    grouped = {label: [] for label in LABELS}
    for name, count in QUOTAS:
        grouped[LABEL_OF[name]].append((name, count))
    chosen = []
    for label in LABELS:
        base = grouped[label]
        base_sum = sum(count for _name, count in base)
        raw = [(name, count * per / base_sum) for name, count in base]
        floors = {name: int(value) for name, value in raw}
        remainder = per - sum(floors.values())
        order = sorted(raw, key=lambda item: item[1] - int(item[1]), reverse=True)
        for offset in range(remainder):
            floors[order[offset][0]] += 1
        for name, _count in base:
            chosen.append((name, floors[name]))
    return chosen


def generate_rows(seed: int = 42, total: int = 1200):
    rng = random.Random(seed)
    rows = []
    seen = set()
    for pattern, count in quotas_for(total):
        builder = BUILDERS[pattern]
        made = 0
        attempt = 0
        while made < count:
            if attempt > count * 40:
                raise RuntimeError(f"Could not build unique rows for {pattern}")
            row = builder(rng, made + attempt)
            validate_example(row)
            key = (row["claim"], row["evidence"])
            attempt += 1
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            made += 1
    return rows


def public_row(row: dict) -> dict:
    return {"claim": row["claim"], "evidence": row["evidence"], "label": row["label"]}


def stratified_split(rows, seed: int = 42):
    rng = random.Random(seed)
    by_label = {label: [] for label in LABELS}
    for row in rows:
        by_label[row["label"]].append(row)
    train, val, test = [], [], []
    for label in LABELS:
        items = by_label[label][:]
        rng.shuffle(items)
        count = len(items)
        n_train = int(round(count * 0.70))
        n_val = int(round(count * 0.15))
        if n_train + n_val > count:
            n_val = count - n_train
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
