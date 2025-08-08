"""
Common token utilities for LLM clients.

This module contains shared logic for populating token counts in LLM responses.
"""

import logging
from typing import Dict, Any, List, Union


def populate_requestgroup_actual_tokens(result: Union[Dict[str, Any], List[Dict[str, Any]]], 
                                      prompt_tokens: int, 
                                      candidate_tokens: int, 
                                      total_tokens: int,
                                      provider_name: str = "Unknown") -> None:
    """
    Populate actual token counts in LLM response results.
    
    This function handles the common logic for distributing token counts across
    single or multiple results, ensuring proper token attribution.
    
    Args:
        result: Single result dict or list of result dicts
        prompt_tokens: Total prompt/input tokens used
        candidate_tokens: Total candidate/output tokens generated
        total_tokens: Total tokens (prompt + candidate)
        provider_name: Name of the LLM provider for logging
    """
    # Handle None values for token counts
    if prompt_tokens is None:
        prompt_tokens = 0
    if candidate_tokens is None:
        candidate_tokens = 0
    if total_tokens is None:
        total_tokens = 0
    
    logging.debug(f"🔍 {provider_name} Token Usage: Prompt={prompt_tokens}, Candidates={candidate_tokens}, Total={total_tokens}")
    
    if isinstance(result, dict):
        # Single result - use all tokens
        result.update({
            'prompt_token_count': prompt_tokens,
            'candidates_token_count': candidate_tokens,
            'total_token_count': total_tokens
        })
    elif isinstance(result, list):
        # Multiple results - distribute tokens proportionally
        num_items = len(result)
        if num_items > 0:
            # Distribute prompt tokens (mostly shared across items)
            prompt_per_item = prompt_tokens // num_items if prompt_tokens > 0 else 0
            remaining_prompt = prompt_tokens % num_items if prompt_tokens > 0 else 0
            
            # Distribute candidate tokens (response tokens)
            candidates_per_item = candidate_tokens // num_items if candidate_tokens > 0 else 0
            remaining_candidates = candidate_tokens % num_items if candidate_tokens > 0 else 0
            
            # Distribute total tokens
            total_per_item = total_tokens // num_items if total_tokens > 0 else 0
            remaining_total = total_tokens % num_items if total_tokens > 0 else 0
            
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    # Add extra tokens to first few items to handle remainder
                    extra_prompt = 1 if i < remaining_prompt else 0
                    extra_candidates = 1 if i < remaining_candidates else 0
                    extra_total = 1 if i < remaining_total else 0
                    
                    item.update({
                        'prompt_token_count': prompt_per_item + extra_prompt,
                        'candidates_token_count': candidates_per_item + extra_candidates,
                        'total_token_count': total_per_item + extra_total
                    })
    else:
        logging.warning(f"⚠️ Unexpected result type for token population: {type(result)}")


# def populate_requestgroup_actual_tokens_simple(result: Union[Dict[str, Any], List[Dict[str, Any]]], 
#                                              total_tokens: int,
#                                              provider_name: str = "Unknown") -> None:
#     """
#     Populate total token count only (simplified version for providers that don't separate prompt/candidate tokens).
    
#     Args:
#         result: Single result dict or list of result dicts
#         total_tokens: Total tokens used
#         provider_name: Name of the LLM provider for logging
#     """
#     if total_tokens is None:
#         total_tokens = 0
    
#     logging.debug(f"🔍 {provider_name} Token Usage: Total={total_tokens}")
    
#     if isinstance(result, dict):
#         result['total_token_count'] = total_tokens
#     elif isinstance(result, list):
#         for item in result:
#             if isinstance(item, dict):
#                 item['total_token_count'] = total_tokens
#     else:
#         logging.warning(f"⚠️ Unexpected result type for token population: {type(result)}") 