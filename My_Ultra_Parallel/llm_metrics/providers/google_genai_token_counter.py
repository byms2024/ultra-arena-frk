"""
Google GenAI token counter implementation.
"""

import logging
from ..llm_token_counter_base import BaseTokenCounter


class GoogleGenAITokenCounter(BaseTokenCounter):
    """Token counter for Google GenAI models."""
    
    def __init__(self, client):
        super().__init__(client)
        self.model_id = "gemini-2.0-flash-001"  # Model for token counting
    
    def count_text_tokens(self, text: str) -> int:
        """Count tokens in plain text using Google GenAI API."""
        try:
            # Handle both direct genai.Client and wrapped client objects
            if hasattr(self.client, 'client'):
                # Our GoogleGenAIClient wrapper
                actual_client = self.client.client
            else:
                # Direct genai.Client
                actual_client = self.client
            
            response = actual_client.models.count_tokens(
                model=self.model_id,
                contents=text
            )
            return response.total_tokens
        except Exception as e:
            logging.warning(f"Failed to count tokens for text: {e}")
            # Fallback: rough estimation
            return len(text.split()) * 1.3
    
    def get_model_name(self) -> str:
        return self.model_id 