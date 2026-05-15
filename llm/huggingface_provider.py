import os
import time
import requests
from typing import Dict, Any

HF_API_BASE = "https://api-inference.huggingface.co/models/"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

def call_huggingface(prompt: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Fallback provider: Hugging Face Inference API."""
    start_time = time.time()
    api_key = os.getenv("HG_TOKEN")
    
    if not api_key:
        raise ValueError("HG_TOKEN is not set.")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": f"[INST] {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.2,
            "return_full_text": False
        }
    }
    
    url = f"{HF_API_BASE}{model}"
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            content = data[0]["generated_text"].strip()
            latency = time.time() - start_time
            return {
                "provider": "Hugging Face",
                "success": True,
                "response": content,
                "latency": latency,
                "fallback_used": True
            }
        else:
            raise ValueError("Unexpected Hugging Face response format.")
            
    except Exception as e:
        raise RuntimeError(f"Hugging Face API Error: {str(e)}")
