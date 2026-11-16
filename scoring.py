"""Step 3: apply the rubric. Pure, deterministic Python: same input -> same score, every run.

Each factor returns (points, max_points, explanation). A factor whose data is missing
contributes a RANGE (0..max) instead of a guess. A lead is auto-decided only when its
whole possible range lands on one side of a threshold; otherwise a human reviews it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ingest import Lead

DECISION_QUALIFIED = "qualified"
DECISION_REJECTED = "rejected"
DECISION_REVIEW = "review"

TIER_POINTS = {"core": 2, "adjacent": 1, "poor": 0}


def norm_key(text: str) -> str:
    """'Non-profit' / 'NonProfit' / 'non profit' -> 'nonprofit' for table lookups."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class FactorScore:
    factor: str
    points: int | None        # None = data missing
    max_points: int
    detail: str

    @property
    def low(self) -> int:
        return self.points if self.points is not None else 0

    @property
    def high(self) -> int:
        return self.points if self.points is not None else self.max_points


@dataclass
class ScoredLead:
    lead: Lead
    factors: list[FactorScore]
    days_since_interaction: int | None
    decision: str = DECISION_REVIEW
    decision_reasons: list[str] = field(default_factory=list)
    priority: str = "NONE"           # HIGH / MEDIUM / LOW for qualified, REVIEW, NONE
    priority_rank: int | None = None
    industry_tier_source: str = ""   # "config" | "llm" | "unclassified"
    llm_concern: str | None = None
    outreach: dict | None = None     # {"angle", "subject", "body", "generated_by"}

    @property
    def score_min(self) -> int:
        return sum(f.low for f in self.factors)

    @property
    def score_max(self) -> int:
        return sum(f.high for f in self.factors)

    @property
    def score(self) -> int:
        """Single 1-10 score for ranking. Missing data -> midpoint of the possible range."""
        return max(1, round((self.score_min + self.score_max) / 2))

    @property
    def is_complete(self) -> bool:
        return all(f.points is not None for f in self.factors)

    def factor(self, name: str) -> FactorScore:
        return next(f for f in self.factors if f.factor == name)


class Scorer:
    def __init__(self, config: dict, reference_date: date):
        rubric = config["rubric"]
        self.size_bands = rubric["size_fit"]
        self.source_points = {norm_key(k): v for k, v in rubric["source_intent"].items()}
        self.recency_bands = rubric["recency"]
        self.stale_after = rubric["stale_after_days"]
        self.thresholds = config["thresholds"]
        self.reference_date = reference_date
        # industry key -> tier, seeded from config; the LLM adds unknown industries later.
        self.industry_tiers: dict[str, str] = {}
        self.industry_tier_origin: dict[str, str] = {}
        for tier, names in rubric["industry_tiers"].items():
            for n in names:
                self.industry_tiers[norm_key(n)] = tier
                self.industry_tier_origin[norm_key(n)] = "config"

    # ----- industry tiers -------------------------------------------------
    def unknown_industries(self, leads: list[Lead]) -> list[str]:
        """Distinct industries not covered by config (these go to the LLM once each)."""
        out, seen = [], set()
        for l in leads:
            if l.industry and norm_key(l.industry) not in self.industry_tiers:
                k = norm_key(l.industry)
                if k not in seen:
                    seen.add(k)
                    out.append(l.industry)
        return out

    def add_industry_tiers(self, mapping: dict[str, str], origin: str = "llm") -> None:
        for industry, tier in mapping.items():
            if tier in TIER_POINTS:
                self.industry_tiers[norm_key(industry)] = tier
                self.industry_tier_origin[norm_key(industry)] = origin

    # ----- factors --------------------------------------------------------
    def _size(self, lead: Lead) -> FactorScore:
        if lead.company_size is None:
            return FactorScore("size_fit", None, 3, "company size unknown")
        for band in self.size_bands:
            hi = band["max"] if band["max"] is not None else float("inf")
            if band["min"] <= lead.company_size <= hi:
                return FactorScore("size_fit", band["points"], 3,
                                   f"{lead.company_size:,} employees: {band['label']}")
        return FactorScore("size_fit", 0, 3, f"{lead.company_size} employees: outside all bands")

    def _source(self, lead: Lead) -> FactorScore:
        if lead.source is None:
            return FactorScore("source_intent", None, 3, "lead source unknown")
        pts = self.source_points.get(norm_key(lead.source))
        if pts is None:
            # Unmapped source: we do not guess intent, we treat it as unknown.
            lead.flags.append(f"unmapped_source:{lead.source}")
            return FactorScore("source_intent", None, 3, f"unrecognised source '{lead.source}'")
        return FactorScore("source_intent", pts, 3, f"source '{lead.source}'")

    def _recency(self, lead: Lead) -> tuple[FactorScore, int | None]:
        if lead.last_interaction_date is None:
            return FactorScore("recency", None, 2, "last interaction date unknown"), None
        days = (self.reference_date - lead.last_interaction_date).days
        if days < 0:
            lead.flags.append("interaction_date_in_future")
            days = 0
        for band in self.recency_bands:
            if band["max_days"] is None or days <= band["max_days"]:
                return FactorScore("recency", band["points"], 2,
                                   f"{days} days since last touch: {band['label']}"), days
        return FactorScore("recency", 0, 2, f"{days} days since last touch"), days

    def _industry(self, lead: Lead) -> tuple[FactorScore, str]:
        if lead.industry is None:
            return FactorScore("industry_fit", None, 2, "industry unknown"), "unclassified"
        key = norm_key(lead.industry)
        tier = self.industry_tiers.get(key)
        if tier is None:
            return (FactorScore("industry_fit", None, 2,
                                f"industry '{lead.industry}' could not be classified"), "unclassified")
        return (FactorScore("industry_fit", TIER_POINTS[tier], 2, f"{lead.industry}: {tier} vertical"),
                self.industry_tier_origin.get(key, "config"))

    # ----- scoring + decision ------------------------------------------------
    def score(self, lead: Lead) -> ScoredLead:
        recency, days = self._recency(lead)
        industry, tier_src = self._industry(lead)
        s = ScoredLead(
            lead=lead,
            factors=[self._size(lead), self._source(lead), recency, industry],
            days_since_interaction=days,
            industry_tier_source=tier_src,
        )
        self.decide(s)
        return s

    def decide(self, s: ScoredLead) -> None:
        q_min, r_max = self.thresholds["qualified_min"], self.thresholds["rejected_max"]
        reasons: list[str] = []
        lo, hi = s.score_min, s.score_max

        if lo >= q_min:
            decision = DECISION_QUALIFIED
            reasons.append(f"score {s.score}/10 meets qualify bar ({q_min})"
                           + ("" if s.is_complete else f" even with missing data (range {lo}-{hi})"))
        elif hi <= r_max:
            decision = DECISION_REJECTED
            reasons.append(f"score {s.score}/10 at or below reject bar ({r_max})"
                           + ("" if s.is_complete else f" even in best case (range {lo}-{hi})"))
            reasons += [f"weak: {f.detail}" for f in s.factors if f.points == 0]
        else:
            decision = DECISION_REVIEW
            if s.is_complete:
                reasons.append(f"borderline score {s.score}/10 (between {r_max} and {q_min})")
            else:
                missing = [f.factor for f in s.factors if f.points is None]
                reasons.append(f"missing {', '.join(missing)}: score could be {lo}-{hi}, too close to call")

        # Guardrails: things a score alone should not auto-approve.
        if decision == DECISION_QUALIFIED:
            if s.days_since_interaction is not None and s.days_since_interaction > self.stale_after:
                decision = DECISION_REVIEW
                reasons.append(f"interaction is {s.days_since_interaction} days old: confirm lead is still active")
            if s.factor("size_fit").points == 0:
                decision = DECISION_REVIEW
                reasons.append("micro team (10 or fewer employees): strong signals but may lack budget, human decides")
            if s.lead.name is None or s.lead.company is None:
                decision = DECISION_REVIEW
                reasons.append("no contact name/company: cannot send personalised outreach")
            if s.llm_concern:
                decision = DECISION_REVIEW
                reasons.append(f"AI reviewer concern: {s.llm_concern}")

        if s.days_since_interaction is not None and s.days_since_interaction > self.stale_after:
            if "stale_interaction" not in s.lead.flags:
                s.lead.flags.append("stale_interaction")

        s.decision, s.decision_reasons = decision, reasons


def assign_priorities(scored: list[ScoredLead], high_min: int, medium_min: int) -> None:
    """Rank qualified leads (the call queue) and review leads (the human queue)."""
    def sort_key(s: ScoredLead):
        src = s.factor("source_intent").low
        days = s.days_since_interaction if s.days_since_interaction is not None else 10**6
        return (-s.score, -src, days, s.lead.row_number)

    qualified = sorted([s for s in scored if s.decision == DECISION_QUALIFIED], key=sort_key)
    for rank, s in enumerate(qualified, start=1):
        s.priority_rank = rank
        s.priority = "HIGH" if s.score >= high_min else "MEDIUM" if s.score >= medium_min else "LOW"

    review = sorted([s for s in scored if s.decision == DECISION_REVIEW],
                    key=lambda s: (-s.score_max, -s.score, s.lead.row_number))
    for rank, s in enumerate(review, start=1):
        s.priority_rank = rank
        s.priority = "REVIEW"

    for s in scored:
        if s.decision == DECISION_REJECTED:
            s.priority, s.priority_rank = "NONE", None
