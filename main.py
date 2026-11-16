#!/usr/bin/env python3
"""Lead Qualifier CLI.

Pipeline:  ingest -> classify unknown industries (LLM, 1 call) -> score with rubric (rules)
           -> write outreach for qualified leads (LLM, batches of ~10) -> prioritise -> report

Usage:
  python main.py --input data/leads_full_scale_100.csv --output output_report.json
  python main.py --input data/leads_testing.csv --output outputs/testing_report.json
  python main.py --input leads.csv --no-llm          # rules + template messages, no API key needed
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from ingest import load_leads, resolve_reference_date
from llm import LLMClient, angle_for, template_message
from report import build_report, print_summary, write_outputs
from scoring import DECISION_QUALIFIED, DECISION_REVIEW, FactorScore, Scorer, ScoredLead, assign_priorities

log = logging.getLogger("lead_qualifier")


def setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[file_handler, console])
    for noisy in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_dotenv(path: Path = Path(__file__).with_name(".env")) -> None:
    """Minimal .env support (KEY=VALUE lines) so the API key never lives in code or config."""
    import os
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Auto-qualify leads, prioritise outreach, draft first messages.")
    p.add_argument("--input", "-i", default="leads.csv", help="leads CSV (default: leads.csv)")
    p.add_argument("--output", "-o", default="output_report.json", help="JSON report path")
    p.add_argument("--csv", help="CSV report path (default: same name as --output, .csv)")
    p.add_argument("--config", "-c", default="config.yaml", help="rubric/settings file")
    p.add_argument("--log", default="run.log", help="log file path")
    p.add_argument("--no-llm", action="store_true", help="skip the API: rules-only scoring + template messages")
    p.add_argument("--limit", type=int, help="only process the first N leads (prompt testing)")
    p.add_argument("--verbose", "-v", action="store_true", help="show info logs on the console")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_dotenv()
    setup_logging(Path(args.log), args.verbose)
    started = time.time()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    try:
        leads = load_leads(args.input)
    except FileNotFoundError:
        log.error("Input file not found: %s", args.input)
        return 2
    if args.limit:
        leads = leads[: args.limit]
    if not leads:
        log.error("No leads found in %s", args.input)
        return 2

    ref_date = resolve_reference_date(config.get("reference_date"), leads)
    log.info("Reference date for recency: %s", ref_date)
    scorer = Scorer(config, ref_date)
    llm = LLMClient(config, enabled=not args.no_llm)

    # --- 1. LLM: classify industries the rubric has never seen (one batched call) ---
    unknown = scorer.unknown_industries(leads)
    industry_llm: dict[str, dict] = {}
    if unknown:
        log.info("Industries not in rubric, sending to LLM: %s", unknown)
        industry_llm = llm.classify_industries(unknown, config["rubric"]["industry_tiers"])
        scorer.add_industry_tiers({k: v["tier"] for k, v in industry_llm.items()})
        for ind in unknown:
            if ind not in industry_llm:
                log.warning("Industry '%s' unclassified (LLM unavailable): lead goes to review if it matters", ind)

    # --- 2. Rules: score every lead (one bad row never stops the run) ---
    scored: list[ScoredLead] = []
    for lead in leads:
        try:
            scored.append(scorer.score(lead))
        except Exception as e:  # defensive: should not happen, but never crash the batch
            log.exception("Scoring failed for %s", lead.lead_id)
            lead.flags.append(f"processing_error:{type(e).__name__}")
            s = ScoredLead(lead=lead, factors=[FactorScore("size_fit", None, 3, "not scored"),
                                               FactorScore("source_intent", None, 3, "not scored"),
                                               FactorScore("recency", None, 2, "not scored"),
                                               FactorScore("industry_fit", None, 2, "not scored")],
                           days_since_interaction=None, decision=DECISION_REVIEW,
                           decision_reasons=[f"processing error ({type(e).__name__}): needs manual check"])
            scored.append(s)

    # --- 3. LLM: outreach messages (+ second-opinion concern) for qualified leads ---
    qualified = [s for s in scored if s.decision == DECISION_QUALIFIED]
    log.info("%d qualified leads need outreach messages", len(qualified))
    messages = llm.write_outreach(qualified) if llm.available else {}
    for s in qualified:
        msg = messages.get(s.lead.lead_id)
        if msg:
            concern = msg.pop("concern", "")
            s.outreach = msg
            if concern:
                log.info("%s: AI concern '%s' -> review", s.lead.lead_id, concern)
                s.llm_concern = concern
                scorer.decide(s)  # re-run decision with the concern (moves it to review)
        else:
            s.outreach = template_message(s.lead, angle_for(s.lead.source), config["company"])

    # --- 4. Prioritise + report ---
    th = config["thresholds"]
    assign_priorities(scored, th["high_priority_min"], th["medium_priority_min"])
    report = build_report(scored, config=config, input_file=args.input, reference_date=ref_date,
                          llm_stats=llm.stats, industry_llm=industry_llm, duration_s=time.time() - started)
    out = Path(args.output)
    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")
    write_outputs(report, out, csv_path)
    print_summary(report)
    print(f"  Report: {out}   CSV: {csv_path}   Log: {args.log}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
