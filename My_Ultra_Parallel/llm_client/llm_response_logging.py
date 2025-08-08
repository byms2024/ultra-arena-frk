"""
Centralized logging utility for LLM responses.
Handles different response formats and pretty prints them appropriately.
"""

import json
import logging
from typing import Any, Dict, Union


def log_llm_response(provider_name: str, response: Any) -> None:
    """
    Log LLM response in a pretty format.
    
    Args:
        provider_name: Name of the LLM provider (e.g., "Google GenAI", "OpenAI", etc.)
        response: The raw response object from the LLM client
    """
    try:
        # Try to convert response to a comprehensive dictionary
        response_dict = _extract_response_data(response)
        
        # Try to pretty print as JSON
        try:
            json_str = json.dumps(response_dict, indent=2, ensure_ascii=False, default=str)
            logging.debug(f"🔍 Raw {provider_name} Response:\n{json_str}")
        except (TypeError, ValueError) as e:
            # If JSON serialization fails, fall back to string representation
            logging.debug(f"🔍 Raw {provider_name} Response (JSON serialization failed):\n{str(response_dict)}")
            
    except Exception as e:
        # If all else fails, just log the raw response object
        logging.debug(f"🔍 Raw {provider_name} Response (fallback):\n{str(response)}")
        logging.debug(f"🔍 Failed to format {provider_name} response: {str(e)}")


def _extract_response_data(response: Any) -> Dict[str, Any]:
    """
    Extract comprehensive data from various LLM response objects.
    
    Args:
        response: The raw response object from any LLM client
        
    Returns:
        Dictionary containing the response data in a structured format
    """
    try:
        # Handle different response types
        if hasattr(response, '__dict__'):
            # Object with attributes - extract common fields
            data = {}
            
            # Common fields across most LLM APIs
            common_fields = [
                'id', 'object', 'created', 'model', 'choices', 'usage',
                'candidates', 'content', 'role', 'message', 'finish_reason',
                'stop_reason', 'stop_sequence', 'done', 'total_duration',
                'load_duration', 'prompt_eval_count', 'prompt_eval_duration',
                'eval_count', 'eval_duration', 'created_at'
            ]
            
            for field in common_fields:
                if hasattr(response, field):
                    value = getattr(response, field)
                    if value is not None:
                        data[field] = value
            
            # Handle nested objects
            if hasattr(response, 'choices') and response.choices:
                data['choices'] = []
                for choice in response.choices:
                    choice_data = {}
                    if hasattr(choice, '__dict__'):
                        for attr in ['index', 'message', 'finish_reason']:
                            if hasattr(choice, attr):
                                choice_data[attr] = getattr(choice, attr)
                    else:
                        choice_data = str(choice)
                    data['choices'].append(choice_data)
            
            if hasattr(response, 'candidates') and response.candidates:
                data['candidates'] = []
                for candidate in response.candidates:
                    candidate_data = {}
                    if hasattr(candidate, '__dict__'):
                        for attr in ['content', 'index', 'finish_reason']:
                            if hasattr(candidate, attr):
                                candidate_data[attr] = getattr(candidate, attr)
                    else:
                        candidate_data = str(candidate)
                    data['candidates'].append(candidate_data)
            
            if hasattr(response, 'content') and response.content:
                data['content'] = []
                for content_block in response.content:
                    content_data = {}
                    if hasattr(content_block, '__dict__'):
                        for attr in ['type', 'text', 'inline_data']:
                            if hasattr(content_block, attr):
                                content_data[attr] = getattr(content_block, attr)
                    else:
                        content_data = str(content_block)
                    data['content'].append(content_data)
            
            if hasattr(response, 'message') and response.message:
                message_data = {}
                if hasattr(response.message, '__dict__'):
                    for attr in ['role', 'content']:
                        if hasattr(response.message, attr):
                            message_data[attr] = getattr(response.message, attr)
                else:
                    message_data = str(response.message)
                data['message'] = message_data
            
            if hasattr(response, 'usage') and response.usage:
                usage_data = {}
                if hasattr(response.usage, '__dict__'):
                    for attr in ['prompt_tokens', 'completion_tokens', 'total_tokens', 
                                'input_tokens', 'output_tokens']:
                        if hasattr(response.usage, attr):
                            usage_data[attr] = getattr(response.usage, attr)
                else:
                    usage_data = str(response.usage)
                data['usage'] = usage_data
            
            return data
            
        elif isinstance(response, (dict, list)):
            # Already a dictionary or list
            return response
            
        else:
            # Fallback - convert to string representation
            return {"raw_response": str(response)}
            
    except Exception as e:
        # If extraction fails, return basic info
        return {
            "error": f"Failed to extract response data: {str(e)}",
            "raw_response": str(response),
            "response_type": type(response).__name__
        } 