"""
OpenAI token counter implementation.
"""

import logging
from ..llm_token_counter_base import BaseTokenCounter


class OpenAITokenCounter(BaseTokenCounter):
    """Token counter for OpenAI models using tiktoken."""
    
    def __init__(self, client, model_name: str = "gpt-4.1"):
        super().__init__(client)
        self.model_name = model_name
        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model(model_name)
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