import logging
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from .huggingface_provider import call_huggingface
from .fallback_handler import log_provider_usage, get_deterministic_fallback
from .response_cache import get_cached_response, store_response

log = logging.getLogger(__name__)


class ResilientProviderRouter(BaseChatModel):
    """
    LangChain compatible chat model with a resilient provider chain:
        0. Persistent response cache  (pre-generated Claude commentary, instant)
        1. Hugging Face Inference API (optional cloud fallback)
        2. Deterministic templates    (guaranteed, never fails)

    Each provider is tried in order; the first success wins. The deterministic
    template guarantees the app never crashes due to LLM unavailability.
    """

    @property
    def _llm_type(self) -> str:
        return "resilient_router"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:

        # Combine messages into a single prompt for the simpler providers.
        prompt = "\n".join(
            [msg.content for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
        )
        if not prompt:
            prompt = str(messages[0].content)

        # 0. Persistent disk cache — serve pre-generated Claude commentary
        #    instantly so the model doesn't run again and again.
        cached = get_cached_response(prompt)
        if cached is not None:
            log_provider_usage(cached.get("provider", "Cache"), True, 0.0, False)
            return self._wrap(cached)

        # 1. Hugging Face (optional cloud fallback)
        try:
            result = call_huggingface(prompt)
            log_provider_usage(result["provider"], True, result["latency"], True)
            store_response(prompt, result["response"], result["provider"])
            return self._wrap(result)
        except Exception as hf_err:
            log.warning("Hugging Face failed: %s. Using deterministic fallback.", hf_err)
            log_provider_usage("Hugging Face", False, 0.0, True, str(hf_err))

        # 2. Deterministic fallback (always succeeds)
        fallback_resp = get_deterministic_fallback(prompt)
        meta = {
            "provider": "Local Template",
            "success": True,
            "response": fallback_resp,
            "latency": 0.0,
            "fallback_used": True,
        }
        log_provider_usage("Local Template", True, 0.0, True)
        return self._wrap(meta)

    @staticmethod
    def _wrap(result: dict) -> ChatResult:
        message = AIMessage(content=result["response"], response_metadata=result)
        return ChatResult(generations=[ChatGeneration(message=message)])
