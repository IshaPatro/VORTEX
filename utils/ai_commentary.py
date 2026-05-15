"""
ai_commentary.py — LLM-driven market insight generation.

Uses Hugging Face Inference API to generate market commentary based on
regime data, portfolio metrics, and stress test results.

If the API fails, times out, or the token is missing, falls back to
deterministic template-based summaries so the app NEVER crashes.
"""
from __future__ import annotations
import os
import json
import logging
import requests
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_TOKEN = os.getenv("HG_TOKEN")

TIMEOUT_SEC = 5


def _fallback_commentary(prompt_type: str, context: Dict[str, Any]) -> str:
    """Deterministic fallback templates if AI is unavailable."""
    if prompt_type == "regime":
        regime = context.get("regime", "Unknown")
        return f"Deterministic Insight: The market is currently in a '{regime}' regime. " \
               f"Proceed with the standard risk protocols associated with this state."

    elif prompt_type == "portfolio":
        ret = context.get("return_ann", 0)
        vol = context.get("vol_ann", 0)
        sharpe = context.get("sharpe", 0)
        return f"Deterministic Insight: Portfolio exhibits {vol}% annualized volatility " \
               f"with a Sharpe ratio of {sharpe}. The return is {ret}%."

    elif prompt_type == "stress":
        worst = context.get("worst_scenario", "None")
        loss = context.get("worst_loss", 0)
        return f"Deterministic Insight: The portfolio is most vulnerable to '{worst}', " \
               f"with a projected severe drawdown of {loss}%."

    return "No insights available."


def _call_hf_api(prompt: str) -> str:
    if not HF_TOKEN:
        log.warning("HF_TOKEN missing. Using fallback.")
        raise ValueError("No token")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"[INST] {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.4,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        raise ValueError("Unexpected API response format.")
    except Exception as e:
        log.warning(f"HF API failed: {e}")
        raise


def generate_regime_commentary(current_regime: str, probs: Dict[str, float]) -> str:
    prompt = (
        f"As a quantitative risk manager, briefly summarize the current market environment. "
        f"The primary regime is '{current_regime}'. The probabilities are: {json.dumps(probs)}. "
        f"Keep it under 3 sentences, institutional tone. Focus on risk implications."
    )
    try:
        return _call_hf_api(prompt)
    except Exception:
        return _fallback_commentary("regime", {"regime": current_regime})


def generate_portfolio_commentary(metrics: Dict[str, float]) -> str:
    prompt = (
        f"As a quant risk analyst, interpret these portfolio metrics: {json.dumps(metrics)}. "
        f"Highlight the key risk exposures (Volatility, VaR, Max Drawdown). "
        f"Keep it to 2-3 sentences, objective and institutional."
    )
    try:
        return _call_hf_api(prompt)
    except Exception:
        return _fallback_commentary("portfolio", {
            "return_ann": metrics.get("Ann. Return (%)", 0),
            "vol_ann": metrics.get("Ann. Volatility (%)", 0),
            "sharpe": metrics.get("Sharpe Ratio", 0)
        })


def generate_stress_commentary(worst_scenario: str, loss: float) -> str:
    prompt = (
        f"A portfolio stress test shows maximum vulnerability to the '{worst_scenario}' scenario "
        f"with a projected loss of {loss}%. Provide a 2 sentence risk warning."
    )
    try:
        return _call_hf_api(prompt)
    except Exception:
        return _fallback_commentary("stress", {"worst_scenario": worst_scenario, "worst_loss": loss})
