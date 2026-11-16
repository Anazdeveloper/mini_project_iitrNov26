"""Step 1 of the pipeline: read the CSV and turn messy rows into clean Lead objects.

Design rule: ingestion NEVER raises on bad data. Every problem becomes a data-quality
flag on the lead, so downstream steps can route it to human review instead of crashing.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

EXPECTED_COLUMNS = ["name", "company", "company_size", "industry", "source", "last_interaction_date"]

# Values that mean "we don't know" in real CRM exports.
MISSING_TOKENS = {"", "na", "n/a", "null", "none", "-", "?", "unknown"}
UNKNOWN_PREFIX = re.compile(r"^unknown\b", re.IGNORECASE)  # "Unknown Sector", "Unknown Source"


@dataclass
class Lead:
    lead_id: str                      # stable id ("L001") used to match LLM answers back to rows
    row_number: int                   # 1-based data row in the source file
    name: str | None
    company: str | None
    company_size: int | None
    industry: str | None
    source: str | None
    last_interaction_date: date | None
    raw: dict = field(default_factory=dict)   # original strings, kept for the report
    flags: list[str] = field(default_factory=list)  # data-quality issues found

    @property
    def display_name(self) -> str:
        return self.name or f"(no name) @ {self.company or 'unknown company'}"


def _clean(value: str | None) -> str | None:
    """Trim whitespace and map every 'missing' spelling to None."""
    if value is None:
        return None
    value = value.strip()
    if value.lower() in MISSING_TOKENS or UNKNOWN_PREFIX.match(value):
        return None
    return value


def _parse_size(value: str | None, flags: list[str]) -> int | None:
    if value is None:
        flags.append("missing_company_size")
        return None
    digits = value.replace(",", "").replace("+", "").strip()
    try:
        size = int(float(digits))
    except ValueError:
        flags.append(f"invalid_company_size:{value}")
        return None
    if size <= 0:
        flags.append(f"invalid_company_size:{value}")
        return None
    return size


def _parse_date(value: str | None, flags: list[str]) -> date | None:
    if value is None:
        flags.append("missing_last_interaction_date")
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    flags.append(f"invalid_date:{value}")
    return None


def load_leads(path: str | Path) -> list[Lead]:
    """Read a leads CSV. Returns cleaned leads; logs (never raises) on row-level problems."""
    path = Path(path)
    leads: list[Lead] = []
    seen: dict[tuple[str, str], str] = {}

    # utf-8-sig strips the BOM Excel adds to CSV exports.
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in header]
        if missing_cols:
            # Not fatal: the affected fields are just treated as missing on every row.
            log.warning("Input is missing columns %s; they will be treated as blank", missing_cols)

        for i, row in enumerate(reader, start=1):
            row = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
            if not any(v.strip() for v in row.values()):
                log.info("Row %d is empty, skipped", i)
                continue

            flags: list[str] = []
            name = _clean(row.get("name"))
            company = _clean(row.get("company"))
            industry = _clean(row.get("industry"))
            source = _clean(row.get("source"))
            if name is None:
                flags.append("missing_name")
            if company is None:
                flags.append("missing_company")
            if industry is None:
                flags.append("missing_industry")
            if source is None:
                flags.append("missing_source")

            lead = Lead(
                lead_id=f"L{i:03d}",
                row_number=i,
                name=name,
                company=company,
                company_size=_parse_size(_clean(row.get("company_size")), flags),
                industry=industry,
                source=source,
                last_interaction_date=_parse_date(_clean(row.get("last_interaction_date")), flags),
                raw={c: row.get(c, "") for c in EXPECTED_COLUMNS},
                flags=flags,
            )

            # Same person at same company twice = CRM duplicate. Keep it, but flag it.
            key = ((name or "").lower(), (company or "").lower())
            if name and company and key in seen:
                lead.flags.append(f"duplicate_of:{seen[key]}")
            else:
                seen[key] = lead.lead_id

            if lead.flags:
                log.info("%s (%s): data flags %s", lead.lead_id, lead.display_name, lead.flags)
            leads.append(lead)

    log.info("Loaded %d leads from %s", len(leads), path)
    return leads


def resolve_reference_date(setting, leads: list[Lead]) -> date:
    """'auto' = latest interaction date in the data (the sample data is historical)."""
    if setting and str(setting).lower() != "auto":
        return datetime.strptime(str(setting), "%Y-%m-%d").date()
    dates = [l.last_interaction_date for l in leads if l.last_interaction_date]
    return max(dates) if dates else date.today()
