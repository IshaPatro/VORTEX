"""
response_cache.py — Persistent on-disk cache for LLM responses.

The agent prompts are deterministic for a given portfolio/regime/metrics
snapshot. To avoid regenerating the same commentary on every app restart,
provider responses are cached to a local JSON file keyed by a hash of the
prompt. Rich Claude commentary is pre-generated (via Cursor) and stored here,
so the dashboard serves it instantly.

Only genuine provider responses (Claude / Hugging Face) are cached — the
deterministic template fallback is intentionally NOT cached, so a transient
provider outage never permanently poisons the cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm_cache.json")


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def get_cached_response(prompt: str) -> Optional[Dict[str, Any]]:
    """Return a cached result dict for this prompt, or None if not cached."""
    entry = _load().get(_key(prompt))
    if not entry:
        return None
    return {
        "provider": entry.get("provider", "Cache"),
        "success": True,
        "response": entry.get("response", ""),
        "latency": 0.0,
        "fallback_used": False,
        "cached": True,
    }


def store_response(prompt: str, response: str, provider: str) -> None:
    """Persist a successful provider response to the on-disk cache."""
    if not response:
        return
    cache = _load()
    cache[_key(prompt)] = {"provider": provider, "response": response, "prompt": prompt}
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as exc:
        log.warning("Failed to write LLM cache: %s", exc)
