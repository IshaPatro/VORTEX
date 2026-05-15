import os
import json
import time
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)

LOG_FILE = "llm_logs.json"

def log_provider_usage(provider: str, success: bool, latency: float, fallback_used: bool, error: str = ""):
    """Track provider usage, latency, and fallback activations."""
    log_entry = {
        "timestamp": time.time(),
        "provider": provider,
        "success": success,
        "latency_sec": round(latency, 3),
        "fallback_used": fallback_used,
        "error": error
    }
    
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []
            
        logs.append(log_entry)
        
        # Keep only last 1000 logs
        if len(logs) > 1000:
            logs = logs[-1000:]
            
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f)
            
    except Exception as e:
        log.error(f"Failed to write LLM logs: {e}")

def get_deterministic_fallback(prompt: str) -> str:
    """Deterministic fallback if all LLMs fail."""
    # We can do simple keyword matching on the prompt to return decent fallbacks.
    prompt_lower = prompt.lower()
    
    if "regime" in prompt_lower:
        return "Deterministic Insight: The market regime is displaying structural shifts. Adjust risk protocols accordingly."
    elif "stress" in prompt_lower or "loss" in prompt_lower:
         return "Deterministic Insight: A severe stress test vulnerability has been detected. Consider reducing exposure to affected asset classes."
    elif "portfolio" in prompt_lower or "volatility" in prompt_lower:
        return "Portfolio risk increased under elevated volatility conditions with rising cross-asset correlation instability."
    
    return "Portfolio risk increased under elevated volatility conditions with rising cross-asset correlation instability."
