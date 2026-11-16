"""Step 5: turn scored leads into something a sales team can act on.

Outputs:
  * output_report.json: full report (run info, rubric, stats, queues, samples, every lead)
  * output_report.csv: one row per lead, sorted call-queue first, for Excel / CRM import
  * a short console summary
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from scoring import DECISION_QUALIFIED, DECISION_REJECTED, DECISION_REVIEW, ScoredLead

FACTOR_LABELS = {"size_fit": "size", "source_intent": "intent", "recency": "recency", "industry_fit": "industry"}


def reasoning_text(s: ScoredLead) -> str:
    """One-line, human-readable audit trail: every point is explained."""
    parts = []
    for f in s.factors:
        pts = f"+{f.points}" if f.points is not None else f"?(0-{f.max_points})"
        parts.append(f"{pts} {FACTOR_LABELS[f.factor]}: {f.detail}")
    total = f"{s.score}/10" if s.is_complete else f"{s.score}/10 (range {s.score_min}-{s.score_max})"
    return "; ".join(parts) + f" = {total} -> {s.decision.upper()}. " + " ".join(r.rstrip(".") + "." for r in s.decision_reasons)


def weak_factor_reasons(s: ScoredLead) -> list[str]:
    """Factors that scored under half their max: used for 'common rejection reasons'."""
    out = []
    for f in s.factors:
        if f.points is None or f.points * 2 >= f.max_points:
            continue
        if f.factor == "size_fit":
            out.append("Company size outside target (" + f.detail.split(": ", 1)[-1] + ")")
        elif f.factor == "source_intent":
            out.append(f"Low buying intent ({s.lead.source})")
        elif f.factor == "recency":
            out.append("Cold lead (no interaction in 45+ days)")
        elif f.factor == "industry_fit":
            out.append("Poor industry fit")
    return out


def lead_record(s: ScoredLead) -> dict:
    l = s.lead
    return {
        "lead_id": l.lead_id,
        "row": l.row_number,
        "name": l.name,
        "company": l.company,
        "company_size": l.company_size,
        "industry": l.industry,
        "source": l.source,
        "last_interaction_date": l.last_interaction_date.isoformat() if l.last_interaction_date else None,
        "days_since_interaction": s.days_since_interaction,
        "score": s.score,
        "score_range": [s.score_min, s.score_max],
        "decision": s.decision,
        "priority": s.priority,
        "priority_rank": s.priority_rank,
        "reasoning": reasoning_text(s),
        "factor_breakdown": [
            {"factor": f.factor, "points": f.points, "max_points": f.max_points, "detail": f.detail}
            for f in s.factors
        ],
        "decision_reasons": s.decision_reasons,
        "data_flags": l.flags,
        "industry_tier_source": s.industry_tier_source,
        "ai_concern": s.llm_concern,
        "outreach": s.outreach,
    }


def pick_samples(qualified: list[ScoredLead], n: int = 5) -> list[dict]:
    """3-5 sample messages, round-robin across angles so the variants are visible."""
    by_angle: dict[str, list[ScoredLead]] = {}
    for s in qualified:
        if s.outreach:
            by_angle.setdefault(s.outreach["angle"], []).append(s)
    picks: list[ScoredLead] = []
    while len(picks) < n and any(by_angle.values()):
        for angle in list(by_angle):
            if by_angle[angle] and len(picks) < n:
                picks.append(by_angle[angle].pop(0))
    return [{
        "lead_id": s.lead.lead_id, "to": f"{s.lead.name} ({s.lead.company})", "priority": s.priority,
        "score": s.score, "angle": s.outreach["angle"], "subject": s.outreach["subject"],
        "body": s.outreach["body"], "generated_by": s.outreach["generated_by"],
    } for s in picks]


def build_report(scored: list[ScoredLead], *, config: dict, input_file: str, reference_date,
                 llm_stats, industry_llm: dict[str, dict], duration_s: float) -> dict:
    total = len(scored)
    by = {d: [s for s in scored if s.decision == d] for d in (DECISION_QUALIFIED, DECISION_REVIEW, DECISION_REJECTED)}
    qualified = sorted(by[DECISION_QUALIFIED], key=lambda s: s.priority_rank or 0)
    review = sorted(by[DECISION_REVIEW], key=lambda s: s.priority_rank or 0)
    pct = lambda n: round(100 * n / total, 1) if total else 0.0

    rejection_reasons = Counter(r for s in by[DECISION_REJECTED] for r in weak_factor_reasons(s))
    review_reasons = Counter()
    for s in review:
        for r in s.decision_reasons:
            if r.startswith("borderline"):
                review_reasons["Borderline score (6)"] += 1
            elif r.startswith("missing"):
                review_reasons["Missing data makes score uncertain"] += 1
            elif "days old" in r:
                review_reasons["Stale interaction (over 1 year): confirm still active"] += 1
            elif r.startswith("no contact"):
                review_reasons["No contact name/company for outreach"] += 1
            elif r.startswith("micro team"):
                review_reasons["Micro team with strong signals"] += 1
            elif r.startswith("AI reviewer"):
                review_reasons["AI reviewer raised a concern"] += 1
    flag_counts = Counter(f.split(":")[0] for s in scored for f in s.lead.flags)

    llm_model = config["llm"]["model"]
    llm_status = "ok" if not llm_stats.disabled_reason else llm_stats.disabled_reason
    return {
        "run": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_file": input_file,
            "reference_date": reference_date.isoformat(),
            "model": llm_model,
            "llm_status": llm_status,
            "api_calls": llm_stats.api_calls,
            "failed_api_calls": llm_stats.failed_calls,
            "input_tokens": llm_stats.input_tokens,
            "output_tokens": llm_stats.output_tokens,
            "items_served_by_fallback": llm_stats.fallback_items,
            "duration_seconds": round(duration_s, 1),
            "errors": llm_stats.errors[:50],
        },
        "rubric": {
            "summary": ("10-point score = company size fit (0-3) + source intent (0-3) + recency (0-2) + "
                        "industry fit (0-2). >=7 qualified (9-10 HIGH, 8 MEDIUM, 7 LOW priority), <=5 rejected, 6 or uncertain "
                        "due to missing data -> human review. Micro teams (<=10), stale (>1yr), nameless, or "
                        "AI-flagged leads are never auto-qualified."),
            "factors": config["rubric"],
            "thresholds": config["thresholds"],
        },
        "summary": {
            "total_processed": total,
            "qualified": {"count": len(qualified), "pct": pct(len(qualified))},
            "review": {"count": len(review), "pct": pct(len(review))},
            "rejected": {"count": len(by[DECISION_REJECTED]), "pct": pct(len(by[DECISION_REJECTED]))},
            "high_priority": sum(1 for s in qualified if s.priority == "HIGH"),
            "average_score": round(sum(s.score for s in scored) / total, 2) if total else None,
            "common_rejection_reasons": [{"reason": r, "count": c} for r, c in rejection_reasons.most_common()],
            "review_reasons": [{"reason": r, "count": c} for r, c in review_reasons.most_common()],
            "data_quality_flags": dict(flag_counts.most_common()),
            "industries_classified_by_ai": [
                {"industry": k, "tier": v["tier"], "rationale": v["rationale"]} for k, v in industry_llm.items()
            ],
            "estimated_time_saved_minutes": total * 10,  # brief: 8-12 min per manual decision
        },
        "priority_queue": [{
            "rank": s.priority_rank, "priority": s.priority, "lead_id": s.lead.lead_id,
            "name": s.lead.name, "company": s.lead.company, "score": s.score,
            "why": "; ".join(f.detail for f in s.factors),
            "outreach_subject": s.outreach["subject"] if s.outreach else None,
        } for s in qualified],
        "review_queue": [{
            "rank": s.priority_rank, "lead_id": s.lead.lead_id, "name": s.lead.name,
            "company": s.lead.company, "score": s.score, "score_range": [s.score_min, s.score_max],
            "why_review": s.decision_reasons, "data_flags": s.lead.flags,
        } for s in review],
        "disqualified": [{
            "lead_id": s.lead.lead_id, "name": s.lead.name, "company": s.lead.company,
            "score": s.score, "reasons": weak_factor_reasons(s) or s.decision_reasons,
        } for s in sorted(by[DECISION_REJECTED], key=lambda s: (s.score, s.lead.row_number))],
        "sample_outreach_messages": pick_samples(qualified),
        "leads": [lead_record(s) for s in sorted(scored, key=lambda s: s.lead.row_number)],
    }


def write_outputs(report: dict, json_path: Path, csv_path: Path | None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_path is None:
        return
    order = {"qualified": 0, "review": 1, "rejected": 2}
    rows = sorted(report["leads"], key=lambda r: (order[r["decision"]], r["priority_rank"] or 999, -r["score"]))
    cols = ["priority_rank", "priority", "decision", "score", "score_range", "name", "company", "company_size",
            "industry", "source", "last_interaction_date", "reasoning", "data_flags", "outreach_subject", "outreach_body"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            o = r.get("outreach") or {}
            w.writerow({
                **{c: r.get(c) for c in cols if c in r},
                "score_range": f"{r['score_range'][0]}-{r['score_range'][1]}",
                "data_flags": "; ".join(r["data_flags"]),
                "outreach_subject": o.get("subject", ""),
                "outreach_body": o.get("body", ""),
            })


def print_summary(report: dict) -> None:
    s, run = report["summary"], report["run"]
    print("\n" + "=" * 64)
    print(f" Lead qualification report: {s['total_processed']} leads   (ref date {run['reference_date']})")
    print("=" * 64)
    print(f"  Qualified : {s['qualified']['count']:>3}  ({s['qualified']['pct']}%)   HIGH priority: {s['high_priority']}")
    print(f"  Review    : {s['review']['count']:>3}  ({s['review']['pct']}%)")
    print(f"  Rejected  : {s['rejected']['count']:>3}  ({s['rejected']['pct']}%)")
    print(f"  LLM       : {run['llm_status']} | calls {run['api_calls']} (failed {run['failed_api_calls']}) | "
          f"fallback items {run['items_served_by_fallback']} | {run['duration_seconds']}s")
    if s["common_rejection_reasons"]:
        print("  Top rejection reasons:")
        for r in s["common_rejection_reasons"][:4]:
            print(f"    - {r['reason']}: {r['count']}")
    if report["priority_queue"]:
        print("  Call first:")
        for p in report["priority_queue"][:5]:
            print(f"    #{p['rank']:<3} [{p['priority']}] {p['score']}/10  {p['name']} - {p['company']}")
    print("=" * 64)
