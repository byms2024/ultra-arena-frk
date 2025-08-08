"""
Base token counting functionality supporting multiple providers.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod


class BaseTokenCounter(ABC):
    """Abstract base class for token counting."""
    
    def __init__(self, client):
        """Initialize with a client."""
        self.client = client
    
    @abstractmethod
    def count_text_tokens(self, text: str) -> int:
        """Count tokens in plain text."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name used for token counting."""
        pass


class TokenCounter:
    """Unified token counting utility supporting multiple providers."""
    
    def __init__(self, client, provider: str = "google", model_name: str = None):
        """Initialize with appropriate token counter based on provider.
        
        Args:
            client: The API client (Google GenAI or OpenAI)
            provider: "google" or "openai"
            model_name: Model name (for OpenAI, defaults to "gpt-4")
        """
        self.client = client
        self.provider = provider.lower()
        
        # Import provider-specific counters
        if self.provider == "google":
            from .providers.google_genai_token_counter import GoogleGenAITokenCounter
            self.counter = GoogleGenAITokenCounter(client)
        elif self.provider == "openai":
            from .providers.openai_token_counter import OpenAITokenCounter
            model_name = model_name or "gpt-4"
            self.counter = OpenAITokenCounter(client, model_name)
        elif self.provider == "deepseek":
            from .providers.deepseek_token_counter import DeepSeekTokenCounter
            model_name = model_name or "deepseek-chat"
            self.counter = DeepSeekTokenCounter(client)
        elif self.provider == "claude":
            from .providers.claude_token_counter import ClaudeTokenCounter
            model_name = model_name or "claude-sonnet-4-20250514"
            self.counter = ClaudeTokenCounter(client)
        elif self.provider == "huggingface":
            from .providers.huggingface_token_counter import HuggingFaceTokenCounter
            model_name = model_name or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.counter = HuggingFaceTokenCounter(client, model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'google', 'openai', 'deepseek', 'claude', or 'huggingface'")
        
        logging.info(f"Initialized {self.provider} token counter with model: {self.counter.get_model_name()}")
    
    def count_text_tokens(self, text: str) -> int:
        """Count tokens in plain text."""
        return self.counter.count_text_tokens(text)
    
    def count_prompt_tokens(self, prompt: str, file_paths: List[str] = None) -> int:
        """Count tokens in a complete prompt including file paths."""
        try:
            # Build the complete prompt as it would be sent to the API
            complete_prompt = prompt
            
            if file_paths:
                file_paths_text = "\n".join(f"FILE_PATH: {fp}" for fp in file_paths)
                complete_prompt += f"\n\n{file_paths_text}"
            
            return self.count_text_tokens(complete_prompt)
        except Exception as e:
            logging.warning(f"Failed to count prompt tokens: {e}")
            return len(complete_prompt.split()) * 1.3
    
    def count_file_content_tokens(self, file_path: str) -> int:
        """Count tokens in file content by extracting text."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            total_text = ""
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                total_text += text + " "
            
            doc.close()
            
            return self.count_text_tokens(total_text)
        except ImportError:
            logging.warning("PyMuPDF not available, using fallback estimation")
            return self._fallback_file_token_estimation(file_path)
        except Exception as e:
            logging.warning(f"Failed to count file tokens for {file_path}: {e}")
            return self._fallback_file_token_estimation(file_path)
    
    def count_multi_file_request_tokens(self, prompt: str, file_paths: List[str]) -> Dict[str, int]:
        """Count tokens for a multi-file request including prompt and all file contents."""
        try:
            # Count prompt tokens
            prompt_tokens = self.count_prompt_tokens(prompt, file_paths)
            
            # Count file content tokens
            file_tokens = {}
            total_file_tokens = 0
            
            for file_path in file_paths:
                file_token_count = self.count_file_content_tokens(file_path)
                file_tokens[file_path] = file_token_count
                total_file_tokens += file_token_count
            
            # Estimate request structure overhead (JSON structure, file_data wrappers, etc.)
            request_overhead = len(file_paths) * 50  # Rough estimate per file
            
            total_tokens = prompt_tokens + total_file_tokens + request_overhead
            
            return {
                'prompt_tokens': prompt_tokens,
                'file_tokens': file_tokens,
                'request_overhead': request_overhead,
                'total_input_tokens': total_tokens,
                'estimated_response_tokens': int(total_file_tokens * 0.8)  # LLM typically generates substantial response
            }
            
        except Exception as e:
            logging.warning(f"Failed to count multi-file request tokens: {e}")
            return self._fallback_multi_file_estimation(prompt, file_paths)
    
    def count_response_tokens(self, response_text: str) -> int:
        """Count tokens in LLM response."""
        return self.count_text_tokens(response_text)
    
    def estimate_total_tokens_for_group(self, prompt: str, file_paths: List[str]) -> Dict[str, int]:
        """Estimate total tokens for processing a group of files."""
        try:
            # Get detailed breakdown
            breakdown = self.count_multi_file_request_tokens(prompt, file_paths)
            
            # Estimate total tokens (input + output)
            total_tokens = breakdown['total_input_tokens'] + breakdown['estimated_response_tokens']
            
            return {
                'input_tokens': breakdown['total_input_tokens'],
                'estimated_output_tokens': breakdown['estimated_response_tokens'],
                'total_estimated_tokens': total_tokens,
                'breakdown': breakdown
            }
            
        except Exception as e:
            logging.warning(f"Failed to estimate total tokens for group: {e}")
            return self._fallback_group_estimation(prompt, file_paths)
    
    def _fallback_file_token_estimation(self, file_path: str) -> int:
        """Fallback token estimation for files."""
        try:
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            return int(file_size_mb * 1200)  # Conservative estimate
        except:
            return 1000  # Default fallback
    
    def _fallback_multi_file_estimation(self, prompt: str, file_paths: List[str]) -> Dict[str, int]:
        """Fallback estimation for multi-file requests."""
        prompt_tokens = len(prompt.split()) * 1.3
        file_tokens = {fp: self._fallback_file_token_estimation(fp) for fp in file_paths}
        total_file_tokens = sum(file_tokens.values())
        
        return {
            'prompt_tokens': int(prompt_tokens),
            'file_tokens': file_tokens,
            'request_overhead': len(file_paths) * 50,
            'total_input_tokens': int(prompt_tokens + total_file_tokens + len(file_paths) * 50),
            'estimated_response_tokens': int(total_file_tokens * 0.8)
        }
    
    def _fallback_group_estimation(self, prompt: str, file_paths: List[str]) -> Dict[str, int]:
        """Fallback estimation for group processing."""
        breakdown = self._fallback_multi_file_estimation(prompt, file_paths)
        total_tokens = breakdown['total_input_tokens'] + breakdown['estimated_response_tokens']
        
        return {
            'input_tokens': breakdown['total_input_tokens'],
            'estimated_output_tokens': breakdown['estimated_response_tokens'],
            'total_estimated_tokens': total_tokens,
            'breakdown': breakdown
        }
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current token counter configuration."""
        return {
            'provider': self.provider,
            'model_name': self.counter.get_model_name(),
            'counter_type': type(self.counter).__name__
        } 