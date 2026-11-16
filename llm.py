"""Steps 2 and 4: the LLM layer.

Provider is set in config.yaml (`llm.provider`): `gemini` (default, free tier, official
`google-genai` SDK) or `anthropic` (Claude API, official `anthropic` SDK). Both return
schema-constrained JSON, and both feed the same retry / split / fallback logic below.

The LLM does the two jobs rules cannot:
  1. classify_industries(): put industries the rubric has never seen into a tier.
     One call per run covers every distinct unknown industry, so it is consistent.
  2. write_outreach(): write a personalised first message for qualified leads,
     about 10 leads per call, and raise a concern if a lead looks like it should not
     be contacted (for example, the company appears to be shut down).

Robustness, from the inside out:
  * The SDK retries 429 / 5xx / timeouts / connection errors with exponential backoff,
    and calls are spaced out to respect free-tier requests-per-minute limits.
  * Responses are forced into a JSON schema (structured outputs) and then validated:
    every lead_id must come back exactly once and messages must name the company.
  * A batch that still fails is retried, then split in half recursively, so one bad
    lead cannot sink the other nine.
  * Fatal errors (bad key, unknown model, bad request) trip a circuit breaker:
    the run continues on deterministic fallbacks instead of crashing or hammering the API.
  * Every lead served by a fallback is marked, so the report says what the AI did and did not do.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

TIERS = ["core", "adjacent", "poor"]
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMBatchError(Exception):
    """A batch failed in a way that may succeed on retry or with a smaller batch."""


class LLMFatalError(Exception):
    """Retrying will not help (auth, model, request shape). Disable the LLM for this run."""


@dataclass
class LLMStats:
    api_calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    fallback_items: int = 0
    disabled_reason: str | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Message angles: 3 variants chosen by how the lead reached us.
# ---------------------------------------------------------------------------
ANGLES = {
    "demo_followup": "They asked for a demo. Thank them, confirm what you will show them for a company like theirs, and offer two concrete time slots this week.",
    "warm_followup": "They came via a referral or have spoken to sales. Reference that warm connection, recap the likely priority for their team, and suggest a short next step.",
    "insight_led": "They engaged lightly (webinar, content, or outreach). Lead with one specific, plausible observation about lead-qualification challenges for their industry and size (framed as a question, not a claim about them), then a low-pressure CTA.",
}


def angle_for(source: str | None) -> str:
    s = (source or "").lower()
    if "demo" in s:
        return "demo_followup"
    if "referral" in s or "sales call" in s:
        return "warm_followup"
    return "insight_led"


def template_message(lead, angle: str, company_cfg: dict) -> dict:
    """Deterministic fallback used when the LLM is unavailable. Still uses lead details."""
    first = (lead.name or "there").split()[0]
    company, industry = lead.company or "your team", lead.industry or "your industry"
    size = f"{lead.company_size:,}-person " if lead.company_size else ""
    product, sender = company_cfg["name"], company_cfg["sender"]
    if angle == "demo_followup":
        body = (f"Hi {first},\n\nThanks for requesting a demo of {product}. For a {size}{industry} team like "
                f"{company}, I'd focus on how we auto-qualify inbound leads and surface the ones worth a call today. "
                f"Would Tuesday or Thursday afternoon work for 30 minutes?\n\n{sender}")
        subject = f"Your {product} demo for {company}"
    elif angle == "warm_followup":
        body = (f"Hi {first},\n\nFollowing up on our recent conversation. Many {industry} teams of {company}'s size "
                f"lose good leads because qualification is manual. {product} automates that triage. "
                f"Open to a 20-minute call next week to see if it fits?\n\n{sender}")
        subject = f"Next step for {company}"
    else:
        body = (f"Hi {first},\n\nHow does {company} decide which inbound leads get a call first? For {size}{industry} "
                f"teams that is often a manual, time-consuming step. {product} scores and prioritises leads automatically. "
                f"Worth a quick look?\n\n{sender}")
        subject = f"Lead triage at {company}"
    return {"angle": angle, "subject": subject, "body": body, "generated_by": "template_fallback"}


class LLMClient:
    def __init__(self, config: dict, enabled: bool = True):
        self.cfg = config["llm"]
        self.company = config["company"]
        self.stats = LLMStats()
        self.provider = self.cfg.get("provider", "gemini").lower()
        self.model_name = self.cfg["gemini_model"] if self.provider == "gemini" else self.cfg["model"]
        self.client = None
        self._last_call = 0.0
        if not enabled:
            self.stats.disabled_reason = "LLM disabled by --no-llm flag"
        elif self.provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                self.stats.disabled_reason = "GEMINI_API_KEY not set: running rules-only with fallbacks"
            else:
                from google import genai
                from google.genai import types
                retry = types.HttpRetryOptions(attempts=int(self.cfg.get("sdk_max_retries", 4)) + 1,
                                               initial_delay=2.0, max_delay=30.0)
                self.client = genai.Client(api_key=key, http_options=types.HttpOptions(
                    timeout=int(self.cfg.get("timeout_seconds", 120)) * 1000, retry_options=retry))
        elif self.provider == "anthropic":
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                self.stats.disabled_reason = "ANTHROPIC_API_KEY not set: running rules-only with fallbacks"
            else:
                import anthropic
                self.client = anthropic.Anthropic(
                    timeout=float(self.cfg.get("timeout_seconds", 120)),
                    max_retries=int(self.cfg.get("sdk_max_retries", 4)),
                )
        else:
            self.stats.disabled_reason = f"unknown llm.provider '{self.provider}'"
        if self.stats.disabled_reason:
            log.warning(self.stats.disabled_reason)

    @property
    def available(self) -> bool:
        return self.client is not None and self.stats.disabled_reason is None

    # ------------------------------------------------------------------
    # Low-level call: one request, schema-constrained JSON back.
    # ------------------------------------------------------------------
    def _call_json(self, system: str, user: str, schema: dict) -> dict:
        assert self.client is not None
        # Free tiers limit requests per minute: space calls out instead of tripping 429s.
        gap = float(self.cfg.get("min_seconds_between_calls", 0))
        wait = self._last_call + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            if self.provider == "gemini":
                return self._call_json_gemini(system, user, schema)
            return self._call_json_anthropic(system, user, schema)
        finally:
            self._last_call = time.time()

    def _with_hard_timeout(self, fn, **kwargs):
        """Run fn in a daemon thread and give up after a wall-clock limit.
        Needed because a hung free-tier request was observed to ignore the SDK timeout."""
        limit = float(self.cfg.get("hard_timeout_seconds", 150))
        box: dict = {}

        def run():
            try:
                box["result"] = fn(**kwargs)
            except BaseException as e:  # re-raised in the caller's thread
                box["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(limit)
        if t.is_alive():
            raise TimeoutError(f"no response after {limit:.0f}s (hard timeout)")
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _call_json_gemini(self, system: str, user: str, schema: dict) -> dict:
        from google.genai import errors, types
        self.stats.api_calls += 1
        started = time.time()
        try:
            resp = self._with_hard_timeout(
                self.client.models.generate_content,
                model=self.model_name,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    temperature=float(self.cfg.get("temperature", 0.2)),
                    max_output_tokens=int(self.cfg.get("max_tokens", 16000)),
                    # Short classification/copy tasks: low thinking keeps calls fast and within quota.
                    thinking_config=types.ThinkingConfig(thinking_level=self.cfg.get("gemini_thinking_level", "low").upper()),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            log.info("Gemini call ok in %.1fs", time.time() - started)
        except errors.ClientError as e:
            self.stats.failed_calls += 1
            if e.code == 429:
                # SDK retries are exhausted: back off once more, then let the batch logic retry.
                log.warning("Gemini rate limit (429) after SDK retries; sleeping 30s")
                time.sleep(30)
                raise LLMBatchError(f"rate limited: {e.message}") from e
            raise LLMFatalError(f"Gemini {e.code}: {e.message}") from e   # bad key / model / request
        except errors.APIError as e:  # ServerError (5xx) and anything else from the API
            self.stats.failed_calls += 1
            raise LLMBatchError(f"Gemini {getattr(e, 'code', '?')}: {getattr(e, 'message', e)}") from e
        except Exception as e:        # network / timeout
            self.stats.failed_calls += 1
            raise LLMBatchError(f"{type(e).__name__}: {e}") from e

        um = getattr(resp, "usage_metadata", None)
        if um:
            self.stats.input_tokens += um.prompt_token_count or 0
            self.stats.output_tokens += um.candidates_token_count or 0
        finish = str(resp.candidates[0].finish_reason) if resp.candidates else "NO_CANDIDATES"
        text = resp.text or ""
        if "MAX_TOKENS" in finish:
            self.stats.failed_calls += 1
            raise LLMBatchError("response truncated (MAX_TOKENS): batch too large")
        if not text.strip():
            self.stats.failed_calls += 1
            raise LLMBatchError(f"empty response (finish_reason={finish})")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.stats.failed_calls += 1
            raise LLMBatchError(f"invalid JSON from model: {e}") from e

    def _call_json_anthropic(self, system: str, user: str, schema: dict) -> dict:
        import anthropic
        kwargs = dict(
            model=self.model_name,
            max_tokens=int(self.cfg.get("max_tokens", 16000)),
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.cfg.get("effort", "low"),
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        self.stats.api_calls += 1
        try:
            if self.cfg.get("use_refusal_fallback", True):
                # If the model declines, the API re-runs the request on a fallback model.
                resp = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError) as e:
            self.stats.failed_calls += 1
            raise LLMFatalError(f"{type(e).__name__}: {getattr(e, 'message', e)}") from e
        except anthropic.RateLimitError as e:
            # The SDK already retried with backoff. Honour retry-after once more at app level.
            self.stats.failed_calls += 1
            wait = min(int(e.response.headers.get("retry-after", "20") or 20), 60)
            log.warning("Rate limited after SDK retries; sleeping %ss", wait)
            time.sleep(wait)
            raise LLMBatchError("rate limited") from e
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # 5xx / overloaded / timeout / network, after the SDK's own retries.
            self.stats.failed_calls += 1
            raise LLMBatchError(f"{type(e).__name__}: {getattr(e, 'message', e)}") from e

        usage = getattr(resp, "usage", None)
        if usage:
            self.stats.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.stats.output_tokens += getattr(usage, "output_tokens", 0) or 0

        if resp.stop_reason == "refusal":
            self.stats.failed_calls += 1
            raise LLMBatchError("model declined the request (refusal)")
        if resp.stop_reason == "max_tokens":
            self.stats.failed_calls += 1
            raise LLMBatchError("response truncated (max_tokens): batch too large")

        # Use the text after the last fallback switch point (if a fallback happened).
        blocks = list(resp.content)
        last_fb = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)
        text = "".join(b.text for b in blocks[last_fb + 1:] if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.stats.failed_calls += 1
            raise LLMBatchError(f"invalid JSON from model: {e}") from e

    # ------------------------------------------------------------------
    # Batch runner: retry -> split in half -> fallback per item.
    # ------------------------------------------------------------------
    def _run_batch(self, items: list, worker: Callable[[list], dict], label: str) -> dict:
        """worker(items) -> {item_key: result}. Returns results for items that succeeded."""
        if not items or not self.available:
            return {}
        for attempt in range(1, int(self.cfg.get("app_retries", 2)) + 2):
            try:
                return worker(items)
            except LLMFatalError as e:
                self.stats.disabled_reason = f"LLM disabled after fatal error: {e}"
                self.stats.errors.append(str(e))
                log.error("%s. Continuing with deterministic fallbacks.", self.stats.disabled_reason)
                return {}
            except LLMBatchError as e:
                self.stats.errors.append(f"{label} (size {len(items)}, attempt {attempt}): {e}")
                log.warning("%s batch of %d failed (attempt %d): %s", label, len(items), attempt, e)
        if len(items) == 1:
            log.error("%s: giving up on item after retries; using fallback", label)
            return {}
        mid = len(items) // 2
        log.info("%s: splitting batch of %d into %d + %d", label, len(items), mid, len(items) - mid)
        out = self._run_batch(items[:mid], worker, label)
        out.update(self._run_batch(items[mid:], worker, label))
        return out

    # ------------------------------------------------------------------
    # Task 1: industry classification (one call for all unknown industries).
    # ------------------------------------------------------------------
    def classify_industries(self, industries: list[str], tier_examples: dict[str, list[str]]) -> dict[str, dict]:
        if not industries:
            return {}
        system = (
            f"You are a B2B sales-ops analyst at {self.company['name']}, which sells {self.company['product']}. "
            "Classify each industry by how well companies in it fit as buyers of our product.\n"
            "Tiers:\n"
            "- core: tech-forward B2B sellers with sales/lead pipelines similar to our best customers\n"
            "- adjacent: viable buyers with real sales teams, but slower or less tech-led adoption\n"
            "- poor: little or no B2B lead pipeline, public-sector procurement, or very low software budgets\n"
            "Anchor examples already decided (stay consistent with them):\n"
            + "\n".join(f"- {t}: {', '.join(v)}" for t, v in tier_examples.items())
            + "\nReturn every input industry exactly once, spelled exactly as given."
        )
        schema = {
            "type": "object",
            "properties": {"classifications": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "industry": {"type": "string"},
                    "tier": {"type": "string", "enum": TIERS},
                    "rationale": {"type": "string"},
                },
                "required": ["industry", "tier", "rationale"],
                "additionalProperties": False,
            }}},
            "required": ["classifications"],
            "additionalProperties": False,
        }

        def worker(batch: list[str]) -> dict:
            data = self._call_json(system, "Industries:\n" + json.dumps(batch), schema)
            got = {c["industry"]: c for c in data.get("classifications", []) if c.get("tier") in TIERS}
            missing = [i for i in batch if i not in got]
            if missing:
                raise LLMBatchError(f"classification missing industries {missing}")
            return {i: got[i] for i in batch}

        result = self._run_batch(industries, worker, "industry-classification")
        self.stats.fallback_items += len(industries) - len(result)
        return result

    # ------------------------------------------------------------------
    # Task 2: outreach messages + sanity check for qualified leads.
    # ------------------------------------------------------------------
    def write_outreach(self, scored_leads: list) -> dict[str, dict]:
        """Returns {lead_id: {"subject","body","concern","angle","generated_by"}}."""
        c = self.company
        system = (
            f"You write first-touch sales emails for {c['name']}, {c['product']}. "
            f"Tone: {c['tone']}. Sign off as \"{c['sender']}\".\n"
            "Rules for every message:\n"
            "- Address the contact by first name and mention their company by name.\n"
            "- Use at least two specific lead details (industry, team size, how they reached us, timing).\n"
            "- Never invent facts about the company (funding, tools, news) or about what was said in past "
            "calls, demos or referrals: we only know THAT the interaction happened. Frame assumptions as questions.\n"
            "- Body under 110 words, one clear call to action, plain text, no emojis. Use short paragraphs "
            "separated by blank lines (greeting, 1-2 body paragraphs, sign-off on its own line).\n"
            "- Follow the angle given per lead; messages in a batch must not read like copies of each other.\n"
            "Also act as a second pair of eyes: set `concern` to one short sentence ONLY if the record itself "
            "suggests we should not contact this lead (for example, the company name implies it shut down, or "
            "fields contradict each other). Otherwise set `concern` to an empty string. Do not raise concerns about "
            "lead quality (scoring is already done) or about names that look like placeholders or labels such as "
            "'Test_...': contact names may be anonymised in this dataset, so judge only the business facts."
        )
        schema = {
            "type": "object",
            "properties": {"messages": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "lead_id": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "concern": {"type": "string"},
                },
                "required": ["lead_id", "subject", "body", "concern"],
                "additionalProperties": False,
            }}},
            "required": ["messages"],
            "additionalProperties": False,
        }

        def describe(s) -> dict:
            l = s.lead
            return {
                "lead_id": l.lead_id, "name": l.name, "company": l.company,
                "company_size": l.company_size, "industry": l.industry, "source": l.source,
                "days_since_last_interaction": s.days_since_interaction,
                "angle": angle_for(l.source), "angle_instructions": ANGLES[angle_for(l.source)],
                "why_qualified": [f.detail for f in s.factors],
            }

        def worker(batch: list) -> dict:
            payload = [describe(s) for s in batch]
            data = self._call_json(system, "Write one message per lead:\n" + json.dumps(payload, indent=1), schema)
            by_id = {m["lead_id"]: m for m in data.get("messages", [])}
            out = {}
            for s in batch:
                m = by_id.get(s.lead.lead_id)
                if not m or not m.get("body", "").strip():
                    raise LLMBatchError(f"no message returned for {s.lead.lead_id}")
                # Personalisation check (punctuation/case-insensitive). A message that fails it
                # is dropped for that lead only (it gets the template), not the whole batch.
                text = re.sub(r"[^a-z0-9]", "", (m["subject"] + " " + m["body"]).lower())
                if s.lead.company and re.sub(r"[^a-z0-9]", "", s.lead.company.lower()) not in text:
                    log.warning("%s: message does not name the company; using template instead", s.lead.lead_id)
                    continue
                out[s.lead.lead_id] = {
                    "angle": angle_for(s.lead.source), "subject": m["subject"].strip(),
                    "body": m["body"].strip(), "concern": m.get("concern", "").strip(),
                    "generated_by": self.model_name,
                }
            return out

        bs = int(self.cfg.get("message_batch_size", 10))
        results: dict[str, dict] = {}
        for i in range(0, len(scored_leads), bs):
            batch = scored_leads[i:i + bs]
            log.info("Writing outreach for leads %d-%d of %d", i + 1, i + len(batch), len(scored_leads))
            results.update(self._run_batch(batch, worker, "outreach"))
        self.stats.fallback_items += len(scored_leads) - len(results)
        return results
