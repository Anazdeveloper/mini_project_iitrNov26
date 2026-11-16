"""Offline tests (no API key needed): run with `python -m unittest -v`.

They pin down the behaviour the rubric promises: messy data never crashes, missing
data produces a score range, guardrails stop risky auto-qualifications, and an API
outage degrades to fallbacks instead of failing the run.
"""
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import yaml

import main
from ingest import load_leads
from llm import LLMBatchError, LLMClient
from scoring import Scorer

CONFIG = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
REF = date(2024, 1, 20)
HEADER = "name,company,company_size,industry,source,last_interaction_date\n"


def leads_from(rows: str):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(HEADER + rows)
    return load_leads(f.name)


class IngestTests(unittest.TestCase):
    def test_missing_tokens_and_short_rows(self):
        (l,) = leads_from("James Wilson,NA,NA,NA,NA\n")
        self.assertIsNone(l.company)
        self.assertIsNone(l.company_size)
        self.assertIsNone(l.last_interaction_date)
        self.assertIn("missing_source", l.flags)

    def test_unknown_prefix_is_missing(self):
        (l,) = leads_from("A,B,10,Unknown Sector,Unknown Source,2024-01-01\n")
        self.assertIsNone(l.industry)
        self.assertIsNone(l.source)

    def test_duplicates_flagged(self):
        a, b = leads_from("A,B,10,SaaS,Referral,2024-01-01\nA,B,10,SaaS,Referral,2024-01-01\n")
        self.assertIn("duplicate_of:L001", b.flags)


class ScoringTests(unittest.TestCase):
    def score(self, row):
        (l,) = leads_from(row + "\n")
        return Scorer(CONFIG, REF).score(l)

    def test_perfect_lead(self):
        s = self.score("Alice,CloudCo,200,SaaS,Inbound demo request,2024-01-19")
        self.assertEqual((s.score, s.decision), (10, "qualified"))

    def test_obvious_reject(self):
        s = self.score("Bo,Farm,5,Agriculture,LinkedIn outreach,2023-10-01")
        self.assertEqual(s.decision, "rejected")

    def test_missing_size_still_qualifies_if_worst_case_qualifies(self):
        s = self.score("T,Test Inc,NA,SaaS,Inbound demo request,2024-01-20")
        self.assertEqual((s.score_min, s.score_max, s.decision), (7, 10, "qualified"))

    def test_uncertain_range_goes_to_review(self):
        s = self.score("T,Tech Partial,NA,SaaS,,2024-01-20")
        self.assertEqual(s.decision, "review")

    def test_stale_never_auto_qualifies(self):
        s = self.score("T,Old Co,200,SaaS,Inbound demo request,2022-06-15")
        self.assertNotEqual(s.decision, "qualified")
        self.assertIn("stale_interaction", s.lead.flags)

    def test_micro_team_goes_to_review(self):
        s = self.score("T,Tiny,2,SaaS,Inbound demo request,2024-01-20")
        self.assertEqual(s.decision, "review")

    def test_deterministic(self):
        a = self.score("Alice,CloudCo,450,Retail,Webinar attendee,2023-12-30")
        b = self.score("Alice,CloudCo,450,Retail,Webinar attendee,2023-12-30")
        self.assertEqual((a.score, a.decision), (b.score, b.decision))


class EndToEndTests(unittest.TestCase):
    def run_main(self, extra):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            rc = main.main(["-i", "data/leads_testing.csv", "-o", str(out), "--log", str(Path(d) / "l.log"), *extra])
            return rc, json.loads(out.read_text())

    def test_messy_file_rules_only(self):
        rc, r = self.run_main(["--no-llm"])
        self.assertEqual(rc, 0)
        self.assertEqual(r["summary"]["total_processed"], 20)
        for lead in r["leads"]:
            self.assertIn(lead["decision"], {"qualified", "review", "rejected"})
            self.assertTrue(lead["reasoning"])
            if lead["decision"] == "qualified":
                self.assertTrue(lead["outreach"]["body"])

    def test_api_outage_falls_back(self):
        """Every API call fails -> run still completes, messages come from templates."""
        with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "test"}), \
             mock.patch.object(LLMClient, "_call_json", side_effect=LLMBatchError("simulated 529")):
            rc, r = self.run_main([])
        self.assertEqual(rc, 0)
        q = [l for l in r["leads"] if l["decision"] == "qualified"]
        self.assertTrue(q)
        self.assertTrue(all(l["outreach"]["generated_by"] == "template_fallback" for l in q))
        self.assertGreater(r["run"]["items_served_by_fallback"], 0)


if __name__ == "__main__":
    unittest.main()
