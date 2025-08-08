"""
DeepSeek client implementation.
"""

import json
import logging
import time
import os
import asyncio
from typing import Dict, List, Any, Optional

# Import OpenAI (DeepSeek uses OpenAI-compatible API)
try:
    from openai import OpenAI
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logging.warning("DeepSeek not available. Install with: pip install openai")

from ..llm_client_base import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    """DeepSeek client for direct file processing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not DEEPSEEK_AVAILABLE:
            raise ImportError("DeepSeek not available")
        
        # DeepSeek uses OpenAI-compatible API with custom base URL
        self.client = OpenAI(
            api_key=config["api_key"],
            base_url="https://api.deepseek.com/v1"
        )
        self.model_id = config.get("model", "deepseek-chat")
    
    def call_llm(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str, 
                 strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call DeepSeek with user prompt, optional system prompt, and files."""
        try:
            # Build messages list (DeepSeek uses OpenAI-like format)
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user prompt
            messages.append({"role": "user", "content": user_prompt})
            
            # Handle files (DeepSeek doesn't support file uploads, so we'll embed content in prompt)
            file_content = ""
            if files:
                for i, file_path in enumerate(files):
                    logging.info(f"📤 Reading file {i+1}/{len(files)}: {os.path.basename(file_path)}")
                    try:
                        # Check file extension to determine how to read it
                        if file_path.lower().endswith('.pdf'):
                            # Handle PDF files with PyPDF2
                            import PyPDF2
                            with open(file_path, 'rb') as file:
                                pdf_reader = PyPDF2.PdfReader(file)
                                text_content = ""
                                for page_num, page in enumerate(pdf_reader.pages):
                                    text_content += f"\n--- Page {page_num + 1} ---\n"
                                    text_content += page.extract_text()
                        elif file_path.lower().endswith('.txt'):
                            # Handle text files directly
                            with open(file_path, 'r', encoding='utf-8') as file:
                                text_content = file.read()
                        else:
                            # Handle other file types or skip
                            logging.warning(f"Unsupported file type: {file_path}")
                            continue
                        
                        file_content += f"\n[File {i+1}: {os.path.basename(file_path)}]\n{text_content}\n"
                        logging.info(f"✅ File content extracted: {os.path.basename(file_path)} ({len(text_content)} chars)")
                    except ImportError:
                        logging.warning("PyPDF2 not available, using placeholder content")
                        file_content += f"\n[File {i+1}: {os.path.basename(file_path)} - PyPDF2 not installed, content would be extracted here]\n"
                    except Exception as e:
                        logging.error(f"Failed to read file {file_path}: {e}")
                        # Continue with other files
                        continue
            
            # Add file content to the user message if available
            if file_content:
                messages[-1]["content"] += f"\n\nFile Content:\n{file_content}"
            
            # Log the complete request being sent to DeepSeek
            logging.info("🔍 DEEPSEEK REQUEST:")
            logging.info(f"  Model: {self.model_id}")
            logging.info(f"  Temperature: {self.temperature}")
            logging.info(f"  Max Tokens: {self.max_tokens}")
            # Truncate message content to prevent large log files (but keep prompts)
            messages_json = json.dumps(messages, indent=2, ensure_ascii=False)
            truncated_messages = messages_json[:500] + "..." if len(messages_json) > 500 else messages_json
            logging.info(f"  Messages (truncated): {truncated_messages}")
            
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Debug: Log the raw response using centralized logging utility
            from llm_client.llm_response_logging import log_llm_response
            log_llm_response("DeepSeek", response)
            
            # Log the complete response from DeepSeek
            logging.info("🔍 DEEPSEEK RESPONSE:")
            logging.info(f"  Response Object: {response}")
            logging.info(f"  Response Type: {type(response)}")
            logging.info(f"  Response Attributes: {dir(response)}")
            
            if hasattr(response, 'choices') and response.choices:
                logging.info(f"  Choices Count: {len(response.choices)}")
                for i, choice in enumerate(response.choices):
                    logging.info(f"  Choice {i}: {choice}")
                    if hasattr(choice, 'message'):
                        logging.info(f"  Choice {i} Message: {choice.message}")
                        if hasattr(choice.message, 'content'):
                            # Truncate content to prevent large log files
                            content = choice.message.content
                            truncated_content = content[:500] + "..." if len(content) > 500 else content
                            logging.info(f"  Choice {i} Content (truncated): {truncated_content}")
            
            if hasattr(response, 'usage'):
                logging.info(f"  Usage: {response.usage}")
            
            # Parse response
            result = self._parse_deepseek_response(response)
            # Truncate JSON result to prevent large log files
            result_json = json.dumps(result, indent=2, ensure_ascii=False)
            truncated_result = result_json[:500] + "..." if len(result_json) > 500 else result_json
            logging.info(f"🔍 PARSED RESULT (truncated): {truncated_result}")
            
            # For single-file processing, return the first item if it's a list
            if isinstance(result, list) and len(result) == 1:
                result = result[0]
            
            # Return the parsed result directly (not wrapped in text field)
            return result
            
        except Exception as e:
            logging.error(f"DeepSeek API error: {e}")
            return {"error": str(e)}
    
    async def call_llm_async(self, *, files: Optional[List[str]] = None, system_prompt: Optional[str] = None, user_prompt: str,
                           strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Call DeepSeek asynchronously."""
        # Run in thread pool since DeepSeek might not have native async support
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_llm, files=files, system_prompt=system_prompt, user_prompt=user_prompt)
    
    def _parse_deepseek_response(self, response) -> Dict[str, Any]:
        """Parse DeepSeek specific response format."""
        try:
            # Extract text from response
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    content = choice.message.content
                    
                    # Try to parse as JSON (handle markdown code blocks)
                    try:
                        # Remove markdown code blocks if present
                        clean_content = content.strip()
                        if clean_content.startswith('```json'):
                            clean_content = clean_content[7:]  # Remove ```json
                        if clean_content.endswith('```'):
                            clean_content = clean_content[:-3]  # Remove ```
                        clean_content = clean_content.strip()
                        
                        parsed_content = json.loads(clean_content)
                        
                        # Handle different response formats
                        if isinstance(parsed_content, list):
                            # If it's already a list, return as is
                            result = parsed_content
                        elif isinstance(parsed_content, dict):
                            # If it's a single object, wrap it in a list
                            result = [parsed_content]
                        else:
                            # Fallback to text format
                            result = {"text": content}
                    except json.JSONDecodeError:
                        # If not JSON, return as plain text
                        result = {"text": content}
                else:
                    result = {"text": ""}
            else:
                result = {"text": ""}
            
            # Add token usage info if result is a list
            if isinstance(result, list) and result and hasattr(response, 'usage'):
                # Distribute token usage across all results
                total_results = len(result)
                for item in result:
                    if isinstance(item, dict):
                        item.update({
                            'prompt_token_count': response.usage.prompt_tokens // total_results,
                            'candidates_token_count': response.usage.completion_tokens // total_results,
                            'total_token_count': response.usage.total_tokens // total_results
                        })
            elif isinstance(result, dict) and hasattr(response, 'usage'):
                # Single result, add token usage directly
                result.update({
                    'prompt_token_count': response.usage.prompt_tokens,
                    'candidates_token_count': response.usage.completion_tokens,
                    'total_token_count': response.usage.total_tokens
                })
            
            return result
            
        except Exception as e:
            logging.error(f"Error parsing DeepSeek response: {e}")
            return {"error": str(e)} 