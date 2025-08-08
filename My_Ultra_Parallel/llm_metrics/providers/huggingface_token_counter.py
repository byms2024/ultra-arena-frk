"""
HuggingFace token counter implementation.
"""

import logging
from ..llm_token_counter_base import BaseTokenCounter


class HuggingFaceTokenCounter(BaseTokenCounter):
    """Token counter for HuggingFace models using tiktoken."""
    
    def __init__(self, client, model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"):
        super().__init__(client)
        self.model_name = model_name
        try:
            import tiktoken
            # Use GPT-4 encoding as fallback for HuggingFace models
            # This is a reasonable approximation for most modern models
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except ImportError:
            logging.warning("tiktoken not available, using fallback estimation")
            self.encoding = None
        except Exception as e:
            logging.warning(f"Failed to initialize tiktoken for {model_name}: {e}")
            self.encoding = None
    
    def count_text_tokens(self, text: str) -> int:
        """Count tokens in plain text using tiktoken."""
        try:
            if self.encoding:
                return len(self.encoding.encode(text))
            else:
                # Fallback: rough estimation
                return len(text.split()) * 1.3
        except Exception as e:
            logging.warning(f"Failed to count tokens for text: {e}")
            return len(text.split()) * 1.3
    
    def get_model_name(self) -> str:
        return self.model_name 