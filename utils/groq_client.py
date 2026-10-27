"""
utils/groq_client.py — Groq API Integration (LLaMA 3 70B)
Fake News Detection System | Ayush Deval

"""

import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# ── UPDATED MODEL NAMES ───────────────────────
# llama3-70b-8192 was DEPRECATED by Groq
# Use these instead:
GROQ_MODEL_PRIMARY  = "openai/gpt-oss-120b"
GROQ_MODEL_FALLBACK = "openai/gpt-oss-20b"
MAX_TOKENS          = 400


# ── SYSTEM PROMPT ─────────────────────────────
SYSTEM_PROMPT = """You are an expert fake news detection AI specialized in both Indian and international news.

Your job is to analyze a news statement or article and determine if it is REAL or FAKE.

Rules:
- You have deep knowledge of Indian politics, government, history, science, and current affairs
- You know all Indian Prime Ministers: Narendra Modi (current), Manmohan Singh, Vajpayee, Rajiv Gandhi, Indira Gandhi, Nehru
- You know international leaders, scientific facts, and world events
- Be direct and confident in your verdict
- Keep your explanation under 3 sentences — clear and simple
- Always end your response with either [VERDICT: FAKE] or [VERDICT: REAL]

Response format (follow this exactly every time):
Explanation: <your explanation in 2-3 sentences>
Confidence: <High / Medium / Low>
[VERDICT: FAKE] or [VERDICT: REAL]"""


# ── MAIN FUNCTION ─────────────────────────────

def analyze_with_groq(text: str) -> dict:
    """
    Send news text to Groq LLaMA 3 for analysis.
    Tries primary model first, falls back to secondary if it fails.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    # Check API key exists
    if not api_key:
        logger.warning("GROQ_API_KEY not set in .env file")
        return _groq_unavailable("GROQ_API_KEY not found in environment variables.")

    # Check API key format
    if not api_key.startswith("gsk_"):
        logger.warning("GROQ_API_KEY format looks wrong — should start with gsk_")

    # Try primary model first, then fallback
    for model in [GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK]:
        result = _call_groq(api_key, model, text)
        if result.get("groq_used"):
            logger.info("Groq success with model: %s", model)
            return result
        logger.warning("Model %s failed, trying next...", model)

    # Both models failed
    return _groq_unavailable("Both Groq models failed. Check API key and internet connection.")


def _call_groq(api_key: str, model: str, text: str) -> dict:
    """Make a single API call to Groq with specified model."""
    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Analyze this news: {text}"}
            ]
        )

        # Check response is not empty
        if not response.choices:
            logger.error("Groq returned empty choices list")
            return _groq_unavailable("Empty response from Groq")

        raw = response.choices[0].message.content
        if not raw or not raw.strip():
            logger.error("Groq returned empty content")
            return _groq_unavailable("Empty content from Groq")

        raw = raw.strip()
        logger.info("Groq raw response (%s): %s", model, raw[:200])
        return _parse_groq_response(raw)

    except Exception as exc:
        error_msg = str(exc)
        logger.error("Groq API error with model %s: %s", model, error_msg)

        # Specific error messages for common issues
        if "401" in error_msg or "invalid_api_key" in error_msg.lower():
            return _groq_unavailable("Invalid Groq API key. Check your GROQ_API_KEY in .env file.")
        elif "429" in error_msg or "rate_limit" in error_msg.lower():
            return _groq_unavailable("Groq rate limit reached. Wait 1 minute and try again.")
        elif "model_not_found" in error_msg.lower() or "404" in error_msg:
            return _groq_unavailable(f"Model {model} not found on Groq.")
        else:
            return _groq_unavailable(error_msg)


# ── RESPONSE PARSER ───────────────────────────

def _parse_groq_response(raw: str) -> dict:
    """
    Parse Groq response — handles any format variation.
    Very robust — works even if model doesn't follow format exactly.
    """
    raw_upper  = raw.upper()
    lines      = raw.strip().splitlines()
    verdict    = "Unknown"
    explanation = ""
    confidence  = "Medium"

    # ── Extract verdict ───────────────────────
    # Check for explicit verdict tags first
    if "[VERDICT: FAKE]" in raw_upper:
        verdict = "Fake"
    elif "[VERDICT: REAL]" in raw_upper:
        verdict = "Real"
    # Fallback — check for FAKE/REAL keywords in full response
    elif "FAKE" in raw_upper and "REAL" not in raw_upper:
        verdict = "Fake"
    elif "REAL" in raw_upper and "FAKE" not in raw_upper:
        verdict = "Real"
    elif raw_upper.count("FAKE") > raw_upper.count("REAL"):
        verdict = "Fake"
    else:
        verdict = "Real"

    # ── Extract explanation ───────────────────
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith("explanation:"):
            explanation = line.split(":", 1)[1].strip()
            break

    # If no explanation line found — use first meaningful line
    if not explanation:
        for line in lines:
            line = line.strip()
            # Skip verdict tags and confidence lines
            if (line and
                not line.upper().startswith("[VERDICT") and
                not line.lower().startswith("confidence:") and
                len(line) > 20):
                explanation = line
                break

    # Final fallback — use full raw response trimmed
    if not explanation:
        explanation = raw[:300]

    # ── Extract confidence ────────────────────
    for line in lines:
        if line.lower().strip().startswith("confidence:"):
            conf_text = line.split(":", 1)[1].strip().lower()
            if "high" in conf_text:
                confidence = "High"
            elif "low" in conf_text:
                confidence = "Low"
            else:
                confidence = "Medium"
            break

    return {
        "verdict":     verdict,
        "explanation": explanation,
        "confidence":  confidence,
        "groq_used":   True,
        "raw":         raw[:500]   # store raw for debugging
    }


# ── UNAVAILABLE FALLBACK ──────────────────────

def _groq_unavailable(error: str = "") -> dict:
    return {
        "verdict":     "Unavailable",
        "explanation": f"Groq AI unavailable. Using ML model result only. ({error})" if error else "Groq AI unavailable.",
        "confidence":  "N/A",
        "groq_used":   False,
        "error":       error
    }


# ── COMBINED VERDICT ──────────────────────────

def combined_verdict(ml_prediction: int, ml_confidence: float, groq_result: dict) -> dict:
    """Combines ML model result with Groq result into final verdict."""
    ml_label = "Fake" if ml_prediction == 1 else "Real"

    # Groq not available — use ML only
    if not groq_result.get("groq_used"):
        return {
            "final_verdict":    ml_label,
            "final_confidence": round(ml_confidence * 100),
            "ml_verdict":       ml_label,
            "ml_confidence":    round(ml_confidence * 100),
            "groq_verdict":     "Unavailable",
            "groq_explanation": groq_result.get("explanation", ""),
            "agreement":        "ML Model only (Groq unavailable)",
            "verdict_source":   "ML Model only"
        }

    groq_label = groq_result.get("verdict", "Unknown")
    groq_conf  = groq_result.get("confidence", "Medium")

    # Both agree — boost confidence
    if ml_label == groq_label:
        boosted = min(ml_confidence + 0.05, 0.99)
        return {
            "final_verdict":    ml_label,
            "final_confidence": round(boosted * 100),
            "ml_verdict":       ml_label,
            "ml_confidence":    round(ml_confidence * 100),
            "groq_verdict":     groq_label,
            "groq_explanation": groq_result.get("explanation", ""),
            "groq_confidence":  groq_conf,
            "agreement":        "✅ Both models agree",
            "verdict_source":   "ML + Groq (agreed)"
        }

    # Disagree — trust Groq
    return {
        "final_verdict":    groq_label,
        "final_confidence": round(ml_confidence * 100),
        "ml_verdict":       ml_label,
        "ml_confidence":    round(ml_confidence * 100),
        "groq_verdict":     groq_label,
        "groq_explanation": groq_result.get("explanation", ""),
        "groq_confidence":  groq_conf,
        "agreement":        "⚠️ Models disagree — Groq verdict used",
        "verdict_source":   "Groq (overrode ML)"
    }


# ── DEBUG TEST — run this to verify API key works ──
# python -c "from utils.groq_client import test_groq; test_groq()"

def test_groq():
    """
    Quick test to verify your Groq API key and connection.
    Run from your project root:
        python -c "from utils.groq_client import test_groq; test_groq()"
    """
    print("Testing Groq API connection...")
    print(f"API Key found: {'Yes' if os.environ.get('GROQ_API_KEY') else 'NO — check .env file'}")
    print(f"Primary model: {GROQ_MODEL_PRIMARY}")

    result = analyze_with_groq("Narendra Modi is the Prime Minister of India.")
    print(f"\nResult:")
    print(f"  groq_used   : {result.get('groq_used')}")
    print(f"  verdict     : {result.get('verdict')}")
    print(f"  explanation : {result.get('explanation')}")
    print(f"  confidence  : {result.get('confidence')}")
    if result.get("error"):
        print(f"  error       : {result.get('error')}")