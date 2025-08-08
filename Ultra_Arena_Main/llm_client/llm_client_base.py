"""
Base LLM client system supporting multiple providers.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get("model")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4000)
        self.timeout = config.get("timeout", 60)
    
    @abstractmethod
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call the LLM with user prompt, optional system prompt, and optional files."""
        pass
    
    @abstractmethod
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call the LLM asynchronously with user prompt, optional system prompt, and optional files."""
        pass
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse LLM response into standardized format."""
        try:
            if isinstance(response, str):
                # Handle empty or whitespace-only responses
                if not response or response.strip() == "":
                    logging.warning("Received empty response from LLM")
                    return {"error": "Empty response from LLM"}
                return json.loads(response)
            elif hasattr(response, 'text'):
                # Handle empty or whitespace-only responses
                if not response.text or response.text.strip() == "":
                    logging.warning("Received empty response from LLM")
                    return {"error": "Empty response from LLM"}
                return json.loads(response.text)
            elif isinstance(response, dict):
                return response
            else:
                logging.error(f"Unexpected response type: {type(response)}")
                return {"error": f"Unexpected response type: {type(response)}"}
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return {"error": f"JSON parsing failed: {e}"} 