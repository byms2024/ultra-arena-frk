"""
DeepSeek token counter implementation.
"""

import logging
from ..llm_token_counter_base import BaseTokenCounter


class DeepSeekTokenCounter(BaseTokenCounter):
    """Token counter for DeepSeek models."""
    
    def __init__(self, client):
        super().__init__(client)
        self.model_id = "deepseek-chat"  # Default model for token counting
    
    def count_text_tokens(self, text: str) -> int:
        """Count tokens in plain text using DeepSeek's tokenizer."""
        try:
            # Handle both direct client and wrapped client objects
            if hasattr(self.client, 'client'):
                # Wrapped client case
                actual_client = self.client.client
            else:
                # Direct client case
                actual_client = self.client
            
            # Try to use OpenAI-compatible token counting
            if hasattr(actual_client, 'models') and hasattr(actual_client.models, 'count_tokens'):
                # Try OpenAI-like API structure
                response = actual_client.models.count_tokens(
                    model=self.model_id,
                    contents=text
                )
                return response.total_tokens
            else:
                # Fallback to tiktoken estimation (DeepSeek uses similar tokenizer to OpenAI)
                try:
                    import tiktoken
                    encoding = tiktoken.get_encoding("cl100k_base")  # DeepSeek uses similar tokenizer
                    return len(encoding.encode(text))
                except ImportError:
                    logging.warning("tiktoken not available, using fallback estimation")
                    return len(text.split()) * 1.3
                
        except Exception as e:
            logging.warning(f"Failed to count tokens for text: {e}")
            # Fallback: rough estimation
            return len(text.split()) * 1.3
    
    def get_model_name(self) -> str:
        return self.model_id 