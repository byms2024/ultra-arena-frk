"""
OpenAI client implementation.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional

# Import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

from ..llm_client_base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI client for direct file processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available")
        
        self.client = OpenAI(api_key=config["api_key"])
        self.model_id = config["model"]
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call OpenAI with user prompt, optional system prompt, and files."""
        
        # OpenAI only supports image_first strategy - other strategies don't work with OpenAI's API limitations
        if strategy_type in ["direct_file", "text_first"]:
            error_msg = f"OpenAI does not support {strategy_type} strategy. Use image_first strategy instead."
            logging.error(f"❌ {error_msg}")
            return {"error": error_msg}
        
        # Check if this is image first strategy with OpenAI - apply special treatment
        if strategy_type == "image_first" and files:
            return self._call_llm_image_first_special(files=files, system_prompt=system_prompt, user_prompt=user_prompt)
        
        # Standard OpenAI processing for other strategies
        return self._call_llm_standard(files=files, system_prompt=system_prompt, user_prompt=user_prompt)
    
    def _call_llm_image_first_special(self, *, files: List[str], system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """
        Special treatment for OpenAI with image first strategy.
        
        This method implements the specific logic needed for OpenAI when processing images
        in the image first strategy, which may differ from standard file processing.
        """
        try:
            logging.info(f"🔄 Using OpenAI special treatment for image first strategy with {len(files)} files")
            
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message with special image first handling
            user_message = {"role": "user", "content": user_prompt}
            
            # For image first strategy with OpenAI, we need to handle images differently
            # Use base64 encoding for images instead of file uploads
            import base64
            
            # Add file references to the user message
            user_message["content"] = [
                {"type": "text", "text": user_prompt}
            ]
            
            for file_path in files:
                # Read image file and encode as base64
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    
                    # Determine image type from file extension
                    import os
                    file_ext = os.path.splitext(file_path)[1].lower()
                    if file_ext == '.png':
                        mime_type = 'image/png'
                    elif file_ext == '.jpg' or file_ext == '.jpeg':
                        mime_type = 'image/jpeg'
                    elif file_ext == '.gif':
                        mime_type = 'image/gif'
                    elif file_ext == '.webp':
                        mime_type = 'image/webp'
                    else:
                        mime_type = 'image/png'  # default
                    
                    # Add image to message content
                    user_message["content"].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
                    
                    logging.debug(f"📎 Added base64 image: {file_path} ({mime_type})")
            
            messages.append(user_message)
            
            # Use specific model and parameters optimized for image processing
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("OpenAI", response)
            
            result = self._parse_response(response.choices[0].message.content)
            
            # Add token usage info
            if response.usage:
                result['prompt_token_count'] = response.usage.prompt_tokens
                result['candidates_token_count'] = response.usage.completion_tokens
                result['total_token_count'] = response.usage.total_tokens
            

            
            return result
            
        except Exception as e:
            logging.error(f"OpenAI image first special treatment error: {e}")
            return {"error": str(e)}
    
    def _call_llm_standard(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """Standard OpenAI processing for non-image-first strategies."""
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            user_message = {"role": "user", "content": user_prompt}
            
            if files:
                # For OpenAI, we need to handle files using base64 encoding
                import base64
                
                # Add file references to the user message
                user_message["content"] = [
                    {"type": "text", "text": user_prompt}
                ]
                
                for file_path in files:
                    # Read file and encode as base64
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                        base64_data = base64.b64encode(file_data).decode('utf-8')
                        
                        # Determine file type from file extension
                        import os
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext == '.pdf':
                            mime_type = 'application/pdf'
                        elif file_ext == '.png':
                            mime_type = 'image/png'
                        elif file_ext == '.jpg' or file_ext == '.jpeg':
                            mime_type = 'image/jpeg'
                        elif file_ext == '.gif':
                            mime_type = 'image/gif'
                        elif file_ext == '.webp':
                            mime_type = 'image/webp'
                        else:
                            mime_type = 'application/octet-stream'  # default
                        
                        # Add file to message content
                        user_message["content"].append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_data}"
                            }
                        })
                        
                        logging.debug(f"📎 Added base64 file: {file_path} ({mime_type})")
            
            messages.append(user_message)
            
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("OpenAI", response)
            
            result = self._parse_response(response.choices[0].message.content)
            
            # Add token usage info
            if response.usage:
                result['prompt_token_count'] = response.usage.prompt_tokens
                result['candidates_token_count'] = response.usage.completion_tokens
                result['total_token_count'] = response.usage.total_tokens
            
            return result
            
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return {"error": str(e)}
    
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call OpenAI asynchronously."""
        # Run in thread pool since OpenAI doesn't have async API in this context
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_llm, files=files, system_prompt=system_prompt, user_prompt=user_prompt, strategy_type=strategy_type) 