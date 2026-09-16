"""
Node: critique

Adversarial LLM pass that reviews the draft thesis for quality issues:
  - Unsupported claims (evidence cited doesn't support the conclusion)
  - Contradictions between technical and sentiment signals not flagged
  - Overconfidence (confidence score too high for the evidence quality)
  - Missing risks or caveats
  - Vague or non-specific evidence

Uses a DIFFERENT prompt framing than synthesis — the critique LLM is
instructed to be skeptical, adversarial, and to look for weaknesses.

Output: CritiqueResult with approved=True/False.
  - If approved=False and critique_count < max_retries → route back to synthesize_draft
  - If approved=True or critique_count >= max_retries → route to human_approval_gate

The routing logic lives in graph.py as a conditional edge; this node only
produces the critique verdict.

Graceful degradation: if the LLM call fails, the critique auto-approves
with a warning flag so the graph can proceed to human review.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, Field

from app.config import get_llm, get_settings
from app.state import CritiqueResult, LogEntry, ResearchState

logger = logging.getLogger("market_research_agent.nodes.critique")


# ---------------------------------------------------------------------------
# Structured output schema for LLM critique
# ---------------------------------------------------------------------------

class CritiqueOutput(BaseModel):
    """LLM output schema for the adversarial critique."""
    approved: bool = Field(
        description=(
            "True ONLY if the thesis meets ALL quality criteria: "
            "evidence is specific and cited, confidence is calibrated, "
            "contradictions are acknowledged, and risks are listed. "
            "Set to False if ANY significant issue is found."
        )
    )
    issues: list[str] = Field(
        description=(
            "List of issues found in the draft. Each issue should be specific "
            "and reference a particular claim or section of the thesis."
        )
    )
    unsupported_claims: list[str] = Field(
        description=(
            "Claims in the thesis that are not adequately supported by the "
            "evidence provided. Quote or paraphrase the specific claim."
        )
    )
    suggested_revisions: list[str] = Field(
        description=(
            "Specific, actionable revisions the analyst should make. "
            "Each revision should address one of the identified issues."
        )
    )
    confidence_adjustment: float = Field(
        description=(
            "Suggested adjustment to the confidence score in points. "
            "Negative = lower confidence, Positive = raise confidence, "
            "0 = no change needed. Range: -50 to +20."
        )
    )


# ---------------------------------------------------------------------------
# Critique prompt — adversarial framing
# ---------------------------------------------------------------------------

_CRITIQUE_SYSTEM_PROMPT = """\
You are a senior research quality reviewer performing an adversarial review \
of a draft equity research thesis. Your job is to find weaknesses, not to \
agree with the analyst.

Your review criteria (in order of severity):

1. **UNSUPPORTED CLAIMS**: Does the thesis make claims not backed by the \
provided evidence? Flag any conclusion that goes beyond what the data shows.

2. **CONFIDENCE CALIBRATION**: Is the confidence score appropriate?
   - Below 30: requires extremely bearish/bearish signals with no contradictions
   - 30-50: mixed signals, significant uncertainty
   - 50-70: moderate conviction with some supporting evidence
   - Above 70: requires strong, corroborating signals across BOTH technicals and sentiment
   - Above 85: almost never justified with limited data
   
   If the score is too high for the evidence quality, flag it and suggest an adjustment.

3. **CONTRADICTION HANDLING**: If technical signals contradict sentiment signals, \
has the thesis explicitly acknowledged this? A bullish technical read with bearish \
sentiment (or vice versa) that isn't discussed is a critical issue.

4. **MISSING RISKS**: Has the thesis identified realistic risks? Every thesis \
should have at least 2-3 meaningful risks. Generic risks ("market could go down") \
don't count.

5. **EVIDENCE SPECIFICITY**: Does the evidence cite specific numbers (RSI values, \
SMA levels, actual sentiment scores) or is it vague? Vague evidence is a quality issue.

IMPORTANT:
- Be tough but fair. Reject drafts that have significant issues.
- Approve drafts that are well-reasoned even if you might disagree with the direction.
- On a first review, you should reject unless the thesis is genuinely strong — \
this forces a revision pass that usually improves quality.
- You are reviewing research quality, NOT giving investment advice."""


def _build_critique_prompt(
    ticker: str,
    draft_thesis: dict,
    technical_signals: dict | None,
    sentiment_summary: dict | None,
    critique_count: int,
) -> str:
    """Build the user prompt for adversarial critique."""
    parts = [f"## Critique Request: Draft Thesis for {ticker}\n"]

    # --- Draft thesis under review ---
    parts.append("### Draft Thesis")
    parts.append(f"- **Direction**: {draft_thesis.get('direction', 'N/A')}")
    parts.append(f"- **Confidence score**: {draft_thesis.get('confidence_score', 'N/A')}/100")
    parts.append(f"- **Revision number**: {draft_thesis.get('revision_number', 0)}")

    summary = draft_thesis.get("summary", "")
    if summary:
        parts.append(f"\n**Summary**: {summary}")

    tech_ev = draft_thesis.get("technical_evidence", [])
    if tech_ev:
        parts.append("\n**Technical evidence**:")
        for t in tech_ev:
            parts.append(f"- {t}")

    sent_ev = draft_thesis.get("sentiment_evidence", [])
    if sent_ev:
        parts.append("\n**Sentiment evidence**:")
        for s in sent_ev:
            parts.append(f"- {s}")

    contras = draft_thesis.get("contradictions", [])
    if contras:
        parts.append("\n**Contradictions identified by analyst**:")
        for c in contras:
            parts.append(f"- {c}")
    else:
        parts.append("\n**Contradictions identified by analyst**: NONE")

    risks = draft_thesis.get("risks_and_caveats", [])
    if risks:
        parts.append("\n**Risks and caveats**:")
        for r in risks:
            parts.append(f"- {r}")
    else:
        parts.append("\n**Risks and caveats**: NONE")

    # --- Raw data for cross-checking ---
    parts.append("\n### Raw Data (for cross-checking)")

    if technical_signals:
        parts.append(f"- SMA-20: {technical_signals.get('sma_20', 'N/A')}")
        parts.append(f"- SMA-50: {technical_signals.get('sma_50', 'N/A')}")
        parts.append(f"- RSI-14: {technical_signals.get('rsi_14', 'N/A')}")
        parts.append(f"- Volume trend: {technical_signals.get('volume_trend', 'N/A')}")
        parts.append(f"- 5d change: {technical_signals.get('price_change_pct_5d', 'N/A')}%")
        parts.append(f"- 30d change: {technical_signals.get('price_change_pct_30d', 'N/A')}%")
    else:
        parts.append("- Technical signals: UNAVAILABLE")

    if sentiment_summary:
        parts.append(f"- Overall sentiment: {sentiment_summary.get('overall_sentiment', 'N/A')}")
        parts.append(f"- Positive/Negative/Neutral: "
                      f"{sentiment_summary.get('positive_count', 0)}/"
                      f"{sentiment_summary.get('negative_count', 0)}/"
                      f"{sentiment_summary.get('neutral_count', 0)}")
        parts.append(f"- Avg confidence: {sentiment_summary.get('average_confidence', 0):.0%}")
    else:
        parts.append("- Sentiment data: UNAVAILABLE")

    # --- Context about revision history ---
    if critique_count > 0:
        parts.append(
            f"\n### Note: This is revision #{draft_thesis.get('revision_number', 0)} "
            f"after {critique_count} previous critique pass(es). "
            "If the analyst has addressed prior issues, you may approve even "
            "if minor imperfections remain."
        )

    parts.append(
        "\nPerform your adversarial review. Be specific about what's wrong "
        "and what needs to change."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def critique(state: ResearchState) -> dict:
    """
    Adversarial LLM review of the draft thesis.

    Produces a CritiqueResult. The routing decision (revise vs proceed)
    is made by the conditional edge in graph.py, not here.

    On LLM failure: auto-approves with a warning so the graph proceeds
    to human review (the human can catch issues the critique missed).
    """
    start = time.time()
    ticker = state["ticker"]
    critique_count = state.get("critique_count", 0)
    draft = state.get("draft_thesis", {})
    technical_signals = state.get("technical_signals")
    sentiment_summary = state.get("sentiment_summary")
    settings = get_settings()

    # --- Check if this is a fallback/placeholder thesis ---
    # Don't waste an LLM call critiquing a thesis that failed to generate
    if draft.get("direction") == "neutral" and draft.get("confidence_score", 0) <= 10:
        elapsed_ms = (time.time() - start) * 1000
        result = CritiqueResult(
            approved=True,  # Let it through to human — they'll see it's a fallback
            issues=["Draft is a fallback/placeholder thesis due to prior LLM failure"],
            unsupported_claims=[],
            suggested_revisions=[],
            confidence_adjustment=None,
        )
        return {
            "critique_notes": result.model_dump(),
            "critique_count": critique_count + 1,
            "run_log": [
                LogEntry(
                    node="critique",
                    status="skipped",
                    duration_ms=elapsed_ms,
                    message=(
                        f"Skipped critique for {ticker}: draft is a fallback thesis. "
                        "Forwarding to human review."
                    ),
                ).model_dump()
            ],
        }

    # --- Build prompt and call LLM ---
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(CritiqueOutput)

        user_prompt = _build_critique_prompt(
            ticker, draft, technical_signals, sentiment_summary, critique_count,
        )

        logger.info(
            "Running adversarial critique for %s (pass %d)...",
            ticker, critique_count + 1,
        )

        llm_result: CritiqueOutput = structured_llm.invoke(
            [
                {"role": "system", "content": _CRITIQUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        result = CritiqueResult(
            approved=llm_result.approved,
            issues=llm_result.issues,
            unsupported_claims=llm_result.unsupported_claims,
            suggested_revisions=llm_result.suggested_revisions,
            confidence_adjustment=llm_result.confidence_adjustment,
        )

        elapsed_ms = (time.time() - start) * 1000
        verdict = "APPROVED" if result.approved else "NEEDS REVISION"
        logger.info(
            "Critique pass %d for %s: %s (%d issues, conf adj: %s) in %.0fms",
            critique_count + 1, ticker, verdict, len(result.issues),
            result.confidence_adjustment, elapsed_ms,
        )

        return {
            "critique_notes": result.model_dump(),
            "critique_count": critique_count + 1,
            "run_log": [
                LogEntry(
                    node="critique",
                    status="completed",
                    duration_ms=elapsed_ms,
                    message=(
                        f"Critique pass {critique_count + 1} for {ticker}: {verdict}. "
                        f"{len(result.issues)} issue(s) found. "
                        f"Confidence adjustment: {result.confidence_adjustment:+.0f}"
                    ),
                    details={
                        "approved": result.approved,
                        "issues_count": len(result.issues),
                        "unsupported_claims_count": len(result.unsupported_claims),
                        "confidence_adjustment": result.confidence_adjustment,
                    },
                ).model_dump()
            ],
        }

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error("Critique LLM call failed for %s: %s", ticker, e)

        # Auto-approve with warning — human reviewer is the final gate
        result = CritiqueResult(
            approved=True,
            issues=[f"Critique could not be performed due to LLM error: {e}"],
            unsupported_claims=[],
            suggested_revisions=[],
            confidence_adjustment=None,
        )

        return {
            "critique_notes": result.model_dump(),
            "critique_count": critique_count + 1,
            "run_log": [
                LogEntry(
                    node="critique",
                    status="failed",
                    duration_ms=elapsed_ms,
                    message=(
                        f"Critique failed for {ticker}: {e}. "
                        "Auto-approving to proceed to human review."
                    ),
                ).model_dump()
            ],
            "error_log": [
                f"[critique] LLM call failed (auto-approved): {e}"
            ],
        }
