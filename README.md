# Lead Qualifier: auto-qualify, prioritise and draft outreach for 100+ leads

A command-line tool that reads a leads CSV and does four things:

- scores each lead from 1 to 10 against an explicit rubric
- decides whether each lead is **qualified**, **rejected**, or sent to **human review**
- ranks the call queue
- has an LLM (Google Gemini `gemini-4-hydrogen`, free tier) draft a personalised first email for each qualified lead

It outputs `output_report.json` (the full report) and `output_report.csv` (opens in Excel or imports into a CRM).

## How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your-key" > .env           # free key: aistudio.google.com/apikey (never committed)

python main.py -i data/leads_full_scale_100.csv -o output_report.json   # the 100-lead run
python main.py -i leads.csv -o outputs/sample_report.json               # 49-lead sample (clean + messy)
python main.py -i leads.csv --limit 8 -v                                # quick prompt check on 8 leads
python main.py -i leads.csv --no-llm                                    # no API key: rules + template emails
python -m unittest -v                                                   # offline tests, incl. simulated API outage
```
Edit `config.yaml` to change the rubric weights, thresholds, model, batch size or company voice. No code changes needed.
Set `llm.provider: anthropic` (with `ANTHROPIC_API_KEY`) to use Claude instead of Gemini.

## Qualification rubric
Each lead gets 10 points across four factors:

| Factor | Points |
|---|---|
| Company size fit | 0–3. 51–1000 employees is the sweet spot; 10 or fewer gets 0. |
| Source intent | 0–3. Demo request 3, referral or sales call 2, webinar or content 1, cold LinkedIn 0. |
| Recency | 0–2. Within 2 weeks 2, within 45 days 1. |
| Industry fit | 0–2. Core 2, adjacent 1, poor 0. |

**Decisions:** 7 or more is **qualified** (9–10 HIGH, 8 MEDIUM, 7 LOW priority). 5 or less is **rejected**. A score of 6
goes to **review**, and so does any lead whose missing data leaves the outcome undecided. When a field is missing,
the lead gets a *score range* instead of a guess, and it is only auto-decided if the whole range falls on one side
of a threshold.

**Guardrails:** these leads are never auto-qualified and go to review instead:
- micro teams
- interactions older than a year
- leads with no name or company
- leads where the AI reviewer raised a concern

## How rules and the LLM split the work
| Stage | Who | Why |
|---|---|---|
| Clean & validate CSV | Python | `NA`, blanks and `Unknown…` become *missing*, not values. Bad rows are flagged, never crash the run. |
| Industries not in the rubric (e.g. Telecom, Climate Tech) | **LLM, 1 call for all of them** | Judgment call. Each industry is classified once per run, so it scores the same for every lead. |
| Score + decision | Python | Deterministic and auditable. Every point is explained in `reasoning`. |
| First emails + sanity check | **LLM, batches of 10** | 3 angles (demo follow-up, warm follow-up, insight-led) based on lead source. Each email must name the company and use 2+ lead details; an email that doesn't is swapped for a template for that lead only. The LLM can flag leads that shouldn't be contacted, and those go to review. |

**Failure handling:**
- The SDK retries 429, 5xx and timeouts with backoff. Calls are spaced 7s apart to stay under the free-tier rate limit.
- A hard 150s wall-clock timeout per call, enforced by the tool itself (see Pivots).
- Responses are schema-constrained JSON. Every `lead_id` must come back, otherwise the batch is retried and then
  split in half.
- Fatal errors (bad key, wrong model) trip a circuit breaker, and the run finishes on rule scores and template emails.
- A blocked, empty or truncated response counts as a failed batch and goes through the same retry/split path.
- Everything is logged to `run.log`. The report shows API calls, failures and which items used a fallback.

## Results (submitted `output_report.json`, 100 leads)
- **Processed:** 100 leads in 151s using **7 API calls**: 1 to classify 15 unseen industries and 6 outreach batches. 0 failed calls and 0 fallbacks.
- **Qualified:** 59 (14 HIGH, 17 MEDIUM, 28 LOW). Every qualified lead has a Gemini-written email that names the company and uses its details.
- **Review:** 18, all borderline scores of 6. **Rejected:** 23. The top rejection reasons are cold LinkedIn outreach (10), poor industry fit (9), content-download-only intent (8) and no touch in 45+ days (8).
- **49-lead `leads.csv`** (29 clean + 20 messy, `outputs/sample_report.json`): 0 crashes. Missing or unknown fields go to review with a score range. Micro teams, stale leads and nameless leads are not auto-qualified.
- **Rate-limit handling, live:** a later rerun hit Gemini's daily free quota (429). The tool backed off and retried as designed, see `outputs/quota_exhausted_rerun.log`. It was stopped so it wouldn't overwrite the clean report.

## Pivots during the build
- **Provider:** the tool was built on the Claude API first, then switched to Gemini's free tier because no API credits were available. Both providers are still supported (`llm.provider`).
- **Model:** `gemini-2.5-flash` returned 404 (retired for new keys) and `gemini-3.8-flash` took 3 minutes per call, so the tool uses `gemini-4-hydrogen` (5–30s per batch).
- **Hung request:** one call hung for 23 minutes and ignored the SDK timeout, so the tool now enforces its own hard timeout per call. Thinking level was also lowered to `low`, which cut calls from minutes to seconds.
- **All-or-nothing batches:** one email that didn't name the company used to fail its whole batch of 10. Now only that lead falls back to a template.
- **Inconsistent AI concerns:** the AI flagged some `Test_*` contacts as "test data" but not others. Since consistency matters more, the prompt now says to judge only business facts, never name formatting.
- **Thresholds:** the first rubric qualified a 2-person startup, and score 7 was the most common score. So the micro-team guardrail was added, the reject bar was raised to 5 or below, and LOW priority was added for scores of 7.

## Known limitations
- **Narrow inputs.** The data has only 6 columns, with no budget or role signals, so scores reflect *fit and intent*,
  not *ability to buy*.
- **Recency reference date.** Recency is measured against the latest date in the file (`reference_date: auto`),
  because the sample data is from 2024. Set a real date in production.
- **Unrecognised sources are treated as unknown.** Their intent is never guessed, so those leads lean toward review.
- **Free-tier quota:** Gemini's free daily quota covers only a few full runs. Beyond that, emails fall back to templates, which is flagged in the report, until the quota resets or `llm.provider` is switched.
- **Duplicates are flagged, not merged.**
- **Emails are drafts.** A rep should skim them before sending. The LLM is told not to invent facts, but it can't verify
  company details.
- **The weights are business assumptions, not a trained model.** They should be re-tuned against real conversion data
  once it exists.
