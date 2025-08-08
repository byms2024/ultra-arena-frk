"""
Ollama client implementation.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
import os # Added for os.path.basename

# Import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

from ..llm_client_base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """Ollama client for local LLM processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not available")
        
        self.model_name = config["model"]
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Ollama with user prompt, optional system prompt (text-only for now)."""
        try:
            # Ollama doesn't support file uploads in the same way
            # For text-first processing, we pass the extracted text as part of the prompt
            messages = []
            
            # Add system prompt if provided, otherwise use default
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": "You are a helpful assistant that extracts information from documents."})
            
            # Add user prompt
            messages.append({"role": "user", "content": user_prompt})
            
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("Ollama", response)
            
            result = self._parse_ollama_response(response.message.content)
            
            # Add file_name_llm mapping for proper file tracking
            if files and len(files) > 0:
                # Extract original filename from FILE_PATH information in the prompt
                original_filename = None
                if "FILE_PATH:" in user_prompt:
                    # Look for FILE_PATH information in the prompt
                    import re
                    file_path_matches = re.findall(r'FILE_PATH:\s*(.+?)(?:\n|$)', user_prompt)
                    if file_path_matches:
                        # Use the first FILE_PATH found
                        original_filename = os.path.basename(file_path_matches[0].strip())
                        logging.debug(f"🔍 Extracted original filename from FILE_PATH: {original_filename}")
                
                # If no FILE_PATH found, use the processed file name
                if not original_filename:
                    original_filename = os.path.basename(files[0])
                    logging.debug(f"🔍 Using processed filename as fallback: {original_filename}")
                
                if isinstance(result, list):
                    # Multi-file response - add file_name_llm to each result
                    for i, item in enumerate(result):
                        if isinstance(item, dict) and i < len(files):
                            item["file_name_llm"] = original_filename
                elif isinstance(result, dict):
                    # Single file response - add file_name_llm
                    result["file_name_llm"] = original_filename
            
            return result
            
        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            return {"error": str(e)}
    
    def _parse_ollama_response(self, content: str) -> Dict[str, Any]:
        """Parse Ollama response, handling thinking tags and JSON extraction."""
        try:
            # Remove thinking tags if present
            if "<think>" in content and "</think>" in content:
                # Extract content after </think>
                parts = content.split("</think>")
                if len(parts) > 1:
                    content = parts[1].strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content.split("```json", 1)[1]
            elif content.startswith("```"):
                content = content.split("```", 1)[1]
            
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            
            content = content.strip()
            
            # Handle empty content
            if not content:
                logging.warning("Received empty response from Ollama")
                return {"error": "Empty response from Ollama"}
            
            # Try to parse as JSON
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Ollama JSON response: {e}")
            # Truncate content to prevent large log files
            truncated_content = content[:500] + "..." if len(content) > 500 else content
            logging.error(f"Content (truncated): {truncated_content}")
            return {"error": f"JSON parsing failed: {e}"}
        except Exception as e:
            logging.error(f"Error parsing Ollama response: {e}")
            return {"error": f"Response parsing failed: {e}"}
    
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Ollama asynchronously."""
        # Run in thread pool since Ollama doesn't have async API
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_llm, files=files, system_prompt=system_prompt, user_prompt=user_prompt) 