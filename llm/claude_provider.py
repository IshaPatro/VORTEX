import os
import time
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

def call_claude(prompt: str, model: str = "claude-3-haiku-20240307") -> Dict[str, Any]:
    """Primary provider: Anthropic Claude."""
    start_time = time.time()
    api_key = os.getenv("CLAUDE_TOKEN")
    
    if not api_key:
        raise ValueError("CLAUDE_TOKEN is not set.")
        
    try:
        # We use haiku by default for speed, can be upgraded
        chat = ChatAnthropic(temperature=0.2, model=model, max_tokens=256, anthropic_api_key=api_key)
        response = chat.invoke([HumanMessage(content=prompt)])
        latency = time.time() - start_time
        
        return {
            "provider": "Claude",
            "success": True,
            "response": response.content,
            "latency": latency,
            "fallback_used": False
        }
    except Exception as e:
        latency = time.time() - start_time
        raise RuntimeError(f"Claude API Error: {str(e)}")
