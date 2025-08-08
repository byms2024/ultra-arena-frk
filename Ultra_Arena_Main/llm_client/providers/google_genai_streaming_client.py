"""
Google GenAI streaming client implementation.

This module contains the streaming-specific implementation for Google GenAI,
separated from the main client to improve code organization.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from .google_genai_client import GoogleGenAIClientBase


class GoogleGenAIStreamingClient(GoogleGenAIClientBase):
    """Google GenAI client for streaming requests."""
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Google GenAI with streaming API."""
        try:
            # Handle image_first strategy differently
            if strategy_type == "image_first" and files:
                return self._call_llm_image_first_streaming(files=files, system_prompt=system_prompt, user_prompt=user_prompt)
            
            # Standard direct_file processing
            # Upload files
            uploaded_files, original_filenames = self._upload_files(files or [])
            
            # Prepare content parts
            parts = self.create_parts_with_embedded_names(uploaded_files, original_filenames, system_prompt, user_prompt)
            
            # Log request details
            logging.debug(f"Making streaming request to {self.model_id} with {len(parts)} content parts")
            if files:
                logging.debug(f"Files included: {[os.path.basename(f) for f in files]}")
            
            # Use streaming API
            logging.info(f"🔄 Using streaming API for large response handling")
            response_stream = self.client.models.generate_content_stream(
                model=self.model_id,
                contents=parts
            )
            
            # Collect all chunks
            full_text = ""
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
            
            # Create a mock response object for consistency
            class StreamingResponse:
                def __init__(self, text):
                    self.text = text
                    self.candidates = [type('obj', (object,), {'content': [type('obj', (object,), {'parts': [type('obj', (object,), {'text': text})()]})()]})()]
                
                def __str__(self):
                    return f"StreamingResponse(text='{self.text[:100]}...')"
            
            response = StreamingResponse(full_text)
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("Google GenAI (Streaming)", response)
            
            # Parse Google GenAI specific response
            result = self._parse_google_response(response)
            
            # For streaming, we need to estimate tokens since usage_metadata is not available
            # Estimate prompt tokens based on the full prompt
            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
            full_prompt += user_prompt
            
            # Estimate tokens using tiktoken (rough approximation)
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")  # Use the same encoding as GPT models
                
                # Estimate prompt tokens
                prompt_tokens = len(encoding.encode(full_prompt))
                
                # Estimate response tokens
                response_tokens = len(encoding.encode(full_text))
                
                # Total tokens
                total_tokens = prompt_tokens + response_tokens
                
                # Use common token population function
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=prompt_tokens,
                    candidate_tokens=response_tokens,
                    total_tokens=total_tokens,
                    provider_name="Google GenAI (Streaming)"
                )
                
                logging.debug(f"🔍 Streaming Token Estimation: Prompt={prompt_tokens}, Response={response_tokens}, Total={total_tokens}")
                
            except ImportError:
                logging.warning("⚠️ tiktoken not available, using rough token estimation for streaming")
                # Fallback rough estimation
                prompt_tokens = len(full_prompt.split()) * 1.3  # Rough approximation
                response_tokens = len(full_text.split()) * 1.3
                total_tokens = prompt_tokens + response_tokens
                
                # Use common token population function for fallback estimation
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=int(prompt_tokens),
                    candidate_tokens=int(response_tokens),
                    total_tokens=int(total_tokens),
                    provider_name="Google GenAI (Streaming - Fallback)"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Google GenAI Streaming API error: {e}")
            logging.debug(f"Request details - Model: {self.model_id}, Files: {files}, Prompt length: {len(user_prompt)}")
            raise
        finally:
            # Clean up uploaded files
            self._cleanup_files()

    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call Google GenAI asynchronously (runs sync method in thread pool)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_llm, files=files, system_prompt=system_prompt, user_prompt=user_prompt)

    def _call_llm_image_first_streaming(self, *, files: List[str], system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """Handle image_first strategy for Google GenAI with streaming."""
        try:
            logging.info(f"🖼️ Using Google GenAI image_first streaming strategy with {len(files)} image files")
            
            # For image_first strategy, files are already image files (PNG, JPG, etc.)
            # We don't need to upload them since we'll use inline_data
            original_filenames = [os.path.basename(f) for f in files]
            
            # Prepare content parts for images
            parts = self.create_parts_with_embedded_names_for_images(files, original_filenames, system_prompt, user_prompt)
            
            # Log request details
            logging.debug(f"Making image_first streaming request to {self.model_id} with {len(parts)} content parts")
            logging.debug(f"Image files included: {[os.path.basename(f) for f in files]}")
            
            # Use streaming API
            logging.info(f"🔄 Using streaming API for image_first strategy")
            response_stream = self.client.models.generate_content_stream(
                model=self.model_id,
                contents=parts
            )
            
            # Collect all chunks
            full_text = ""
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
            
            # Create a mock response object for consistency
            class StreamingResponse:
                def __init__(self, text):
                    self.text = text
                    self.candidates = [type('obj', (object,), {'content': [type('obj', (object,), {'parts': [type('obj', (object,), {'text': text})()]})()]})()]
                
                def __str__(self):
                    return f"StreamingResponse(text='{self.text[:100]}...')"
            
            response = StreamingResponse(full_text)
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("Google GenAI (Image First Streaming)", response)
            
            # Parse Google GenAI specific response
            result = self._parse_google_response(response)
            
            # For streaming, we need to estimate tokens since usage_metadata is not available
            # Estimate prompt tokens based on the full prompt
            full_prompt = ""
            if system_prompt:
                full_prompt += f"{system_prompt}\n\n"
            full_prompt += user_prompt
            
            # Estimate tokens using tiktoken (rough approximation)
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")  # Use the same encoding as GPT models
                
                # Estimate prompt tokens
                prompt_tokens = len(encoding.encode(full_prompt))
                
                # Estimate response tokens
                response_tokens = len(encoding.encode(full_text))
                
                # Total tokens
                total_tokens = prompt_tokens + response_tokens
                
                # Use common token population function
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=prompt_tokens,
                    candidate_tokens=response_tokens,
                    total_tokens=total_tokens,
                    provider_name="Google GenAI (Image First Streaming)"
                )
                
                logging.debug(f"🔍 Streaming Token Estimation (Image First): Prompt={prompt_tokens}, Response={response_tokens}, Total={total_tokens}")
                
            except ImportError:
                logging.warning("⚠️ tiktoken not available, using rough token estimation for streaming")
                # Fallback rough estimation
                prompt_tokens = len(full_prompt.split()) * 1.3  # Rough approximation
                response_tokens = len(full_text.split()) * 1.3
                total_tokens = prompt_tokens + response_tokens
                
                # Use common token population function for fallback estimation
                from llm_client.token_utils import populate_requestgroup_actual_tokens
                populate_requestgroup_actual_tokens(
                    result=result,
                    prompt_tokens=int(prompt_tokens),
                    candidate_tokens=int(response_tokens),
                    total_tokens=int(total_tokens),
                    provider_name="Google GenAI (Image First Streaming - Fallback)"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Google GenAI Image First Streaming API error: {e}")
            logging.debug(f"Request details - Model: {self.model_id}, Image files: {files}, Prompt length: {len(user_prompt)}")
            raise
        finally:
            # Clean up uploaded files
            self._cleanup_files() 