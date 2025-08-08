"""
Strategy-agnostic grouping utilities for consistent JSON and CSV output.
This module provides common functionality that can be used by all processing strategies.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from fuzzywuzzy import fuzz


class GroupProcessingUtils:
    """Utility class for common group processing operations."""
    
    @staticmethod
    def create_group_stats_template(file_count: int) -> Dict[str, Any]:
        """Create a standardized group stats template."""
        return {
            "total_files": file_count,
            "successful_files": 0,
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": 0
        }
    
    @staticmethod
    def calculate_processing_time(start_time: float) -> float:
        """Calculate processing time in seconds."""
        return time.time() - start_time
    
    @staticmethod
    def distribute_tokens_proportionally(
        total_prompt_tokens: int,
        total_candidates_tokens: int, 
        total_actual_tokens: int,
        num_files: int,
        token_counter: Any,
        prompt_text: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, int]]:
        """
        Distribute token counts proportionally across files in a group.
        
        Args:
            total_prompt_tokens: Total prompt tokens from LLM response
            total_candidates_tokens: Total completion tokens from LLM response
            total_actual_tokens: Total actual tokens from LLM response
            num_files: Number of files in the group
            token_counter: Token counter instance for estimation
            prompt_text: The prompt text used
            system_prompt: Optional system prompt
            
        Returns:
            List of token dictionaries for each file
        """
        if num_files <= 0:
            return []
        
        # Calculate estimated tokens per file
        estimated_prompt_per_file = token_counter.count_text_tokens(prompt_text) // num_files
        if system_prompt:
            estimated_prompt_per_file += token_counter.count_text_tokens(system_prompt) // num_files
        
        # Distribute actual tokens proportionally
        prompt_per_file = total_prompt_tokens // num_files
        candidates_per_file = total_candidates_tokens // num_files
        actual_per_file = total_actual_tokens // num_files
        
        # Handle remainders
        prompt_remainder = total_prompt_tokens % num_files
        candidates_remainder = total_candidates_tokens % num_files
        actual_remainder = total_actual_tokens % num_files
        
        file_tokens = []
        for i in range(num_files):
            # Add remainder to first few files
            extra_prompt = 1 if i < prompt_remainder else 0
            extra_candidates = 1 if i < candidates_remainder else 0
            extra_actual = 1 if i < actual_remainder else 0
            
            file_tokens.append({
                "estimated_tokens": estimated_prompt_per_file,
                "prompt_tokens": prompt_per_file + extra_prompt,
                "candidates_tokens": candidates_per_file + extra_candidates,
                "actual_tokens": actual_per_file + extra_actual,
                "other_tokens": 0  # Calculated as actual - prompt - candidates
            })
        
        return file_tokens
    
    @staticmethod
    def map_llm_outputs_to_files(
        file_results: List[Dict[str, Any]],
        file_paths: List[str],
        image_paths: Optional[List[Tuple[str, str]]] = None,
        group_index: int = 0
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Map LLM outputs to files using multi-strategy matching.
        
        Args:
            file_results: List of results from LLM
            file_paths: List of original file paths
            image_paths: Optional list of (file_path, image_path) tuples for image strategies
            group_index: Group index for logging
            
        Returns:
            List of (file_path, result) tuples
        """
        logging.info(f"🔗 Mapping outputs to files for group {group_index}...")
        
        # Create mapping from file_name_llm to output
        file_name_llm_to_output = {}
        for file_result in file_results:
            if isinstance(file_result, dict) and "file_name_llm" in file_result:
                file_name_llm_to_output[file_result["file_name_llm"]] = file_result
        
        logging.info(f"🔍 Debug: Found {len(file_name_llm_to_output)} results with file_name_llm")
        logging.info(f"🔍 Debug: Available file_name_llm keys: {list(file_name_llm_to_output.keys())}")
        
        results = []
        
        for i, file_path in enumerate(file_paths):
            file_result = None
            
            # Strategy 1: Exact Image Filename Match (for image strategies)
            if image_paths:
                image_path = image_paths[i][1]  # Get image path
                image_filename = Path(image_path).name
                logging.info(f"🔍 Debug: Looking for image filename '{image_filename}' in mapped outputs")
                
                if image_filename in file_name_llm_to_output:
                    file_result = file_name_llm_to_output[image_filename]
                    logging.info(f"✅ Found exact match for image '{image_filename}'")
                else:
                    # Strategy 2: Fuzzy Matching for image filenames
                    best_match = None
                    best_score = 0
                    for llm_filename, output in file_name_llm_to_output.items():
                        score = fuzz.ratio(image_filename.lower(), llm_filename.lower())
                        if score > best_score and score >= 85:  # 85% similarity threshold
                            best_score = score
                            best_match = output
                    
                    if best_match:
                        file_result = best_match
                        logging.info(f"🔍 Found fuzzy match for image '{image_filename}' with score {best_score}")
            
            # Strategy 3: Index-based fallback
            if file_result is None and i < len(file_results):
                file_result = file_results[i]
                logging.info(f"⚠️ Using index-based mapping for file {i}")
            
            # Strategy 4: Last resort - use first available result
            if file_result is None and file_name_llm_to_output:
                first_key = list(file_name_llm_to_output.keys())[0]
                file_result = file_name_llm_to_output[first_key]
                logging.warning(f"⚠️ Using first available result for file {i}")
            
            # Ensure file_name_llm reflects original PDF filename for consistency
            if file_result and "file_name_llm" in file_result:
                file_result["file_name_llm"] = Path(file_path).name
            
            results.append((file_path, file_result or {"error": "No result returned for this file"}))
        
        return results
    
    @staticmethod
    def update_group_stats(
        group_stats: Dict[str, Any],
        results: List[Tuple[str, Dict[str, Any]]],
        processing_time: float,
        total_tokens: int = 0,
        estimated_tokens: int = 0
    ) -> Dict[str, Any]:
        """
        Update group statistics based on processing results.
        
        Args:
            group_stats: Current group stats
            results: List of (file_path, result) tuples
            processing_time: Processing time in seconds
            total_tokens: Total tokens used
            estimated_tokens: Estimated tokens
            
        Returns:
            Updated group stats
        """
        successful_files = sum(1 for _, result in results if "error" not in result)
        failed_files = len(results) - successful_files
        
        group_stats.update({
            "successful_files": successful_files,
            "failed_files": failed_files,
            "processing_time": processing_time,
            "total_tokens": total_tokens,
            "estimated_tokens": estimated_tokens
        })
        
        return group_stats
    
    @staticmethod
    def create_file_result_structure(
        file_path: str,
        llm_result: Dict[str, Any],
        token_stats: Dict[str, int],
        success: bool = True,
        retry_round: Optional[int] = None,
        failure_reason: Optional[str] = None,
        group_ids: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create standardized file result structure for JSON output.
        
        Args:
            file_path: Path to the file
            llm_result: Result from LLM
            token_stats: Token statistics
            success: Whether processing was successful
            retry_round: Retry round number (if applicable)
            failure_reason: Reason for failure (if applicable)
            group_ids: List of group IDs this file was processed in
            
        Returns:
            Standardized file result structure
        """
        file_info = {
            "file_name": Path(file_path).name,
            "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024)
        }
        
        file_process_result = {
            "success": success,
            "retry_round": retry_round,
            "failure_reason": failure_reason,
            "proc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "group_ids_incl_retries": group_ids or []
        }
        
        file_token_stats = {
            "estimated_tokens": token_stats.get("estimated_tokens", 0),
            "prompt_tokens": token_stats.get("prompt_tokens", 0),
            "candidates_tokens": token_stats.get("candidates_tokens", 0),
            "actual_tokens": token_stats.get("actual_tokens", 0),
            "other_tokens": token_stats.get("other_tokens", 0)
        }
        
        return {
            "file_process_result": file_process_result,
            "file_model_output": llm_result,
            "file_token_stats": file_token_stats,
            "file_info": file_info
        }
    
    @staticmethod
    def process_group_with_strategy(
        file_group: List[str],
        group_index: int,
        group_id: str,
        strategy_func: Callable,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """
        Generic group processing function that can be used by any strategy.
        
        Args:
            file_group: List of file paths to process
            group_index: Index of the group
            group_id: Unique ID for the group
            strategy_func: Function that implements the actual processing strategy
            user_prompt: User prompt for LLM
            system_prompt: Optional system prompt
            **kwargs: Additional arguments for strategy_func
            
        Returns:
            Tuple of (results, group_stats, group_id)
        """
        group_start_time = time.time()
        logging.info(f"🔄 Starting processing for group {group_index} ({group_id}): {len(file_group)} files")
        
        # Create initial group stats
        group_stats = GroupProcessingUtils.create_group_stats_template(len(file_group))
        
        try:
            # Call the strategy-specific processing function
            results = strategy_func(
                file_group=file_group,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            
            # Calculate processing time
            processing_time = GroupProcessingUtils.calculate_processing_time(group_start_time)
            
            # Update group stats
            group_stats = GroupProcessingUtils.update_group_stats(
                group_stats, results, processing_time
            )
            
            logging.info(f"✅ Completed processing for group {group_index}: {group_stats['successful_files']} successful, {group_stats['failed_files']} failed")
            
        except Exception as e:
            logging.error(f"❌ Error processing group {group_index}: {e}")
            # Create error results for all files
            results = [(file_path, {"error": str(e)}) for file_path in file_group]
            group_stats.update({
                "successful_files": 0,
                "failed_files": len(file_group),
                "processing_time": GroupProcessingUtils.calculate_processing_time(group_start_time)
            })
        
        return results, group_stats, group_id 