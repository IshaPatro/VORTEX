import time
import logging
from typing import Any, List, Optional, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from .claude_provider import call_claude
from .huggingface_provider import call_huggingface
from .fallback_handler import log_provider_usage, get_deterministic_fallback

log = logging.getLogger(__name__)

class ResilientProviderRouter(BaseChatModel):
    """
    LangChain compatible chat model that automatically falls back from 
    Claude to Hugging Face, and finally to deterministic templates.
    """
    
    # Required for Langchain BaseChatModel
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
        
        # Combine messages into a single prompt for simpler providers
        prompt = "\n".join([msg.content for msg in messages if isinstance(msg, (HumanMessage, AIMessage))])
        if not prompt:
            prompt = str(messages[0].content)

        # 1. Try Claude
        try:
            result = call_claude(prompt)
            log_provider_usage(result["provider"], True, result["latency"], False)
            
            message = AIMessage(
                content=result["response"],
                response_metadata=result
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
            
        except Exception as claude_err:
            log.warning(f"Claude failed: {claude_err}. Attempting Hugging Face fallback.")
            log_provider_usage("Claude", False, 0.0, False, str(claude_err))
            
        # 2. Try Hugging Face
        try:
            start = time.time()
            result = call_huggingface(prompt)
            log_provider_usage(result["provider"], True, result["latency"], True)
            
            message = AIMessage(
                content=result["response"],
                response_metadata=result
            )
            return ChatResult(generations=[ChatGeneration(message=message)])
            
        except Exception as hf_err:
            log.warning(f"Hugging Face failed: {hf_err}. Attempting deterministic fallback.")
            log_provider_usage("Hugging Face", False, 0.0, True, str(hf_err))
            
        # 3. Deterministic Fallback
        fallback_resp = get_deterministic_fallback(prompt)
        meta = {
            "provider": "Local Template",
            "success": True,
            "response": fallback_resp,
            "latency": 0.0,
            "fallback_used": True
        }
        log_provider_usage("Local Template", True, 0.0, True)
        
        message = AIMessage(
            content=fallback_resp,
            response_metadata=meta
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
