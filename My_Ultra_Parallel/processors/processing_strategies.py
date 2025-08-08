"""
Modular processing strategies for different PDF processing approaches.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from llm_client import LLMClientFactory
from common.text_extractor import TextExtractor, RegexExtractor
from common.base_monitor import BasePerformanceMonitor
from .group_processing_utils import GroupProcessingUtils
from .file_mapping_utils import FileMappingFactory


class BaseProcessingStrategy(ABC):
    """Base class for processing strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = BasePerformanceMonitor(self.__class__.__name__)
        self.mandatory_keys = config.get("mandatory_keys", [])
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay_seconds", 1)
    
    @abstractmethod
    def process_file_group(self, *, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process a group of files and return results, stats, and group ID."""
        pass
    
    def check_mandatory_keys(self, result: Dict[str, Any], file_path: str = None, benchmark_comparator = None) -> Tuple[bool, List[str]]:
        """Check if all mandatory keys are present and non-empty, and optionally compare with benchmark values."""
        if not result or not isinstance(result, dict):
            return False, self.mandatory_keys
        
        # Skip validation for 'Outros' documents
        if result.get('DOC_TYPE') == 'Outros':
            return True, []  # Skip validation for 'Outros' documents
        
        # Filter out empty strings and whitespace-only strings from mandatory keys
        filtered_mandatory_keys = [key for key in self.mandatory_keys if key and key.strip()]
        
        # If no valid mandatory keys, return success
        if not filtered_mandatory_keys:
            logging.info("✅ No valid mandatory keys to validate - skipping validation")
            return True, []
        
        missing_keys = []
        present_keys = []
        matching_keys = []
        mismatching_keys = []
        
        for key in filtered_mandatory_keys:
            value = result.get(key)
            if value is None or value == "" or value == "Not found":
                missing_keys.append(key)
            else:
                present_keys.append(key)
                
                # Check benchmark match if comparator is available and file_path is provided
                if benchmark_comparator and file_path:
                    benchmark_record = benchmark_comparator._find_benchmark_record(Path(file_path).name)
                    if benchmark_record:
                        benchmark_value = benchmark_record.get(key)
                        if benchmark_comparator._values_match(benchmark_value, value):
                            matching_keys.append(key)
                        else:
                            mismatching_keys.append(key)
                            logging.info(f"🔍 Value mismatch for {key} in {Path(file_path).name}: "
                                       f"benchmark='{benchmark_value}' vs extracted='{value}'")
        
        # Log the status of mandatory keys
        if len(missing_keys) == 0:
            logging.info(f"✅ All mandatory keys present: {present_keys}")
            
            # Log benchmark matching results if available
            if benchmark_comparator and file_path:
                if len(matching_keys) == len(present_keys):
                    logging.info(f"🎯 All mandatory key values match benchmark: {matching_keys}")
                elif len(matching_keys) > 0:
                    logging.info(f"🎯 Some mandatory key values match benchmark: {matching_keys}")
                    if mismatching_keys:
                        logging.warning(f"⚠️ Some mandatory key values don't match benchmark: {mismatching_keys}")
                else:
                    logging.warning(f"⚠️ No mandatory key values match benchmark. Mismatches: {mismatching_keys}")
        else:
            logging.warning(f"⚠️ Missing mandatory keys: {missing_keys}. Present keys: {present_keys}")
            
            # Log benchmark matching results for present keys if available
            if benchmark_comparator and file_path and matching_keys:
                logging.info(f"🎯 Some present key values match benchmark: {matching_keys}")
                if mismatching_keys:
                    logging.warning(f"⚠️ Some present key values don't match benchmark: {mismatching_keys}")
        
        return len(missing_keys) == 0, missing_keys
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                wait_time = self.retry_delay * (2 ** attempt)
                logging.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)


class DirectFileProcessingStrategy(BaseProcessingStrategy):
    """Strategy for direct file processing with LLM."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        self.provider = config.get("provider", "google")
        self.provider_configs = config.get("provider_configs", {})
        self.max_num_file_parts_per_request = config.get("max_num_file_parts_per_request", 4)
        self.max_num_file_parts_per_batch = config.get("max_num_file_parts_per_batch", 100)
        self.streaming = streaming  # Store streaming flag
        
        # Initialize LLM client
        self.llm_client = LLMClientFactory.create_client(self.provider, self.provider_configs.get(self.provider, {}), streaming=self.streaming)
        
        # Initialize token counter for accurate estimation
        from llm_metrics import TokenCounter
        self.token_counter = TokenCounter(self.llm_client, provider=self.provider)
        
        # Use hybrid retry limits if specified
        if "file_direct_max_retry" in config:
            self.max_retries = config["file_direct_max_retry"]
            logging.info(f"🌐 Direct file strategy using hybrid retry limit: {self.max_retries}")
    
    def process_file_group(self, *, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files by sending them directly to LLM."""
        
        group_start_time = time.time()
        logging.info(f"🔄 Starting direct file processing for group {group_index} ({group_id}): {len(file_group)} files")
        
        results = []
        group_stats = {
            "total_files": len(file_group),
            "successful_files": 0,
            "failed_files": 0,
            "actual_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": 0
        }
        
        try:
            # Enhance the prompt to require file_name_llm field in each response (like backup project)
            enhanced_user_prompt = (
                f"{user_prompt}\n\n"
                "Return a JSON array, one object per file. "
                "Each object MUST include a 'file_name_llm' field that identifies the document. "
                "If you do not include the file_name_llm, your answer will be ignored."
            )
            
            # Process all files in the group together with enhanced prompt structure
            logging.debug(f"🔍 Streaming debug: self.streaming={self.streaming}, self.provider={self.provider}, hasattr={hasattr(self.llm_client, 'call_llm_streaming')}")
            response = self._retry_with_backoff(
                self.llm_client.call_llm, files=file_group, system_prompt=system_prompt, user_prompt=enhanced_user_prompt
            )
            
            if "error" in response:
                logging.error(f"LLM API error for group {group_index}: {response['error']}")
                # Create failed results for all files
                for file_path in file_group:
                    results.append((file_path, {"error": response["error"]}))
                    group_stats["failed_files"] += 1
            else:
                # Parse response and match with files
                if isinstance(response, list):
                    # Response is already a list of results
                    file_results = response
                else:
                    # Single result, wrap in list
                    file_results = [response]
                
                # Map outputs to files using provider-specific file mapping strategy
                file_mapping_strategy = FileMappingFactory.create_strategy(self.provider)
                mapped_results = file_mapping_strategy.map_outputs_to_files(file_results, file_group, group_index)
                
                # Process mapped results and add token estimates
                for file_path, file_result in mapped_results:
                    filename = Path(file_path).name
                    
                    if file_result is not None and "error" not in file_result:
                        # Calculate individual file token estimate using Google token counter
                        try:
                            file_token_estimate = self.token_counter.count_file_content_tokens(file_path)
                            file_result["estimated_tokens"] = file_token_estimate
                        except Exception as e:
                            logging.warning(f"Failed to estimate tokens for {filename}: {e}")
                            file_result["estimated_tokens"] = 0
                        
                        results.append((file_path, file_result))
                        group_stats["successful_files"] += 1
                        
                        # Add token usage if available
                        if "total_token_count" in file_result:
                            group_stats["actual_tokens"] += file_result["total_token_count"]
                    else:
                        # Handle error cases
                        if "error" not in file_result:
                            file_result = {"error": "No result returned for this file"}
                        
                        # Calculate individual file token estimate even for errors
                        try:
                            file_token_estimate = self.token_counter.count_file_content_tokens(file_path)
                            file_result["estimated_tokens"] = file_token_estimate
                        except Exception as e:
                            logging.warning(f"Failed to estimate tokens for {filename}: {e}")
                            file_result["estimated_tokens"] = 0
                        
                        results.append((file_path, file_result))
                        group_stats["failed_files"] += 1
        
        except Exception as e:
            logging.error(f"Error processing group {group_index}: {e}")
            # Create failed results for all files
            for file_path in file_group:
                results.append((file_path, {"error": str(e)}))
                group_stats["failed_files"] += 1
        
        group_stats["processing_time"] = int(time.time() - group_start_time)
        
        # Calculate estimated tokens for the group using proper token counter
        try:
            estimation = self.token_counter.estimate_total_tokens_for_group(user_prompt, file_group)
            group_stats["estimated_tokens"] = estimation["total_estimated_tokens"]
        except Exception as e:
            logging.warning(f"Failed to calculate accurate token estimation: {e}")
            # Fallback to simple estimation
            estimated_tokens = 0
            for file_path in file_group:
                try:
                    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    if file_size_mb < 0.05:
                        base_estimate = 4500
                    elif file_size_mb < 0.15:
                        base_estimate = 5000
                    else:
                        base_estimate = 5500
                    estimated_tokens += int(base_estimate * 1.2)  # Add 20% buffer
                except:
                    estimated_tokens += 6000  # Default fallback
            
            # Add prompt tokens and response tokens
            if system_prompt:
                estimated_tokens += len(system_prompt.split()) * 1.3
            if user_prompt:
                estimated_tokens += len(user_prompt.split()) * 1.3
            estimated_tokens += len(file_group) * 300  # Response tokens
            estimated_tokens += len(file_group) * 100  # Request overhead
            
            group_stats["estimated_tokens"] = int(estimated_tokens)
        
        logging.info(f"✅ Completed direct file processing for group {group_index}: {group_stats['successful_files']} successful, {group_stats['failed_files']} failed")
        
        return results, group_stats, group_id


class TextFirstProcessingStrategy(BaseProcessingStrategy):
    """Strategy for text-first processing with local or remote LLM."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        self.extractor_lib = config.get("pdf_extractor_lib", "pymupdf")
        self.max_text_length = config.get("max_text_length", 50000)
        
        # Initialize text extractor
        self.text_extractor = TextExtractor(self.extractor_lib)
        self.regex_extractor = RegexExtractor()
        
        # Initialize LLM client
        self.llm_provider = config.get("llm_provider", "ollama")
        self.llm_config = config.get("provider_configs", {}).get(self.llm_provider, {})
        self.llm_client = LLMClientFactory.create_client(self.llm_provider, self.llm_config)
        
        # Use hybrid retry limits if specified
        if "text_first_max_retry" in config:
            self.max_retries = config["text_first_max_retry"]
            logging.info(f"📝 Text-first strategy using hybrid retry limit: {self.max_retries}")
    
    def process_file_group(self, *, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files by extracting text first, then sending to LLM."""
        
        group_start_time = time.time()
        logging.info(f"🔄 Starting text-first processing for group {group_index} ({group_id}): {len(file_group)} files")
        
        results = []
        group_stats = {
            "total_files": len(file_group),
            "successful_files": 0,
            "failed_files": 0,
            "total_tokens": 0,
            "processing_time": 0
        }
        
        # Process each file individually
        for file_path in file_group:
            try:
                file_result = self._process_single_file(file_path=file_path, user_prompt=user_prompt, system_prompt=system_prompt)
                results.append((file_path, file_result))
                
                if "error" not in file_result:
                    group_stats["successful_files"] += 1
                    if "total_token_count" in file_result:
                        group_stats["total_tokens"] += file_result["total_token_count"]
                else:
                    group_stats["failed_files"] += 1
                    
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                results.append((file_path, {"error": str(e)}))
                group_stats["failed_files"] += 1
        
        group_stats["processing_time"] = int(time.time() - group_start_time)
        logging.info(f"✅ Completed text-first processing for group {group_index}: {group_stats['successful_files']} successful, {group_stats['failed_files']} failed")
        
        return results, group_stats, group_id
    
    def _process_single_file(self, *, file_path: str, user_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process a single file using text-first approach."""
        
        # Extract text from PDF
        text_content = self.text_extractor.extract_text(file_path, self.max_text_length)
        
        if not text_content:
            return {"error": "No text extracted from PDF"}
        
        # Extract fields using regex first
        regex_fields = self.regex_extractor.extract_all_fields(text_content)
        
        # Create enhanced prompt with extracted text
        enhanced_user_prompt = f"{user_prompt}\n\nExtracted text from document:\n{text_content[:10000]}..."  # Limit text length
        
        # Call LLM
        response = self._retry_with_backoff(
            self.llm_client.call_llm, files=None, system_prompt=system_prompt, user_prompt=enhanced_user_prompt
        )
        
        if "error" in response:
            return response
        
        # Merge regex results with LLM results
        result = response.copy()
        
        # Use regex results ONLY as fallback when LLM didn't provide valid data
        # Map regex field names to expected field names
        field_mapping = {
            "CNPJ": "CNPJ_1",  # Regex returns "CNPJ", but we need "CNPJ_1"
        }
        
        for field, value in regex_fields.items():
            if value and value != "Not found":
                # Map field name if needed
                target_field = field_mapping.get(field, field)
                
                # Only use regex if LLM didn't provide a valid value for this field
                llm_value = result.get(target_field)
                if not llm_value or llm_value == "Not found" or llm_value == "":
                    result[target_field] = value
        
        # Ensure all mandatory keys are present
        for key in self.mandatory_keys:
            if key not in result:
                result[key] = "Not found"
        
        # Add file identification
        result["file_name_llm"] = Path(file_path).name
        
        return result


class HybridProcessingStrategy(BaseProcessingStrategy):
    """Strategy that combines text-first (local) and direct file (remote) processing in two phases."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        
        # Initialize both strategies
        self.text_strategy = TextFirstProcessingStrategy(config)
        self.direct_strategy = DirectFileProcessingStrategy(config)
        
        # Retry limits for hybrid approach
        self.text_first_max_retry = config.get("text_first_max_retry", 3)
        self.file_direct_max_retry = config.get("file_direct_max_retry", 2)
        
        logging.info(f"🔄 Initialized hybrid strategy: text-first max retries={self.text_first_max_retry}, direct file max retries={self.file_direct_max_retry}")
    
    def process_file_group(self, *, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files using two-phase hybrid approach: local text-first, then remote direct file for failures."""
        
        group_start_time = time.time()
        logging.info(f"🔄 Starting hybrid processing for group {group_index} ({group_id}): {len(file_group)} files")
        
        results = []
        group_stats = {
            "total_files": len(file_group),
            "text_first_successful": 0,
            "text_first_failed": 0,
            "direct_file_successful": 0,
            "direct_file_failed": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_tokens": 0,
            "processing_time": 0
        }
        
        # Phase 1: Process all files with text-first (local) strategy
        logging.info(f"📝 Phase 1: Processing {len(file_group)} files with text-first (local) strategy...")
        text_first_results, text_first_stats, _ = self.text_strategy.process_file_group(
            file_group=file_group, group_index=group_index, user_prompt=user_prompt, system_prompt=system_prompt, group_id=f"{group_id}_text_first"
        )
        
        # Update stats for text-first phase
        group_stats["text_first_successful"] = text_first_stats["successful_files"]
        group_stats["text_first_failed"] = text_first_stats["failed_files"]
        group_stats["total_tokens"] += text_first_stats["total_tokens"]
        
        # Identify files that need retry (missing mandatory keys)
        files_needing_retry = []
        successful_files = []
        
        for file_path, result in text_first_results:
            if "error" in result:
                files_needing_retry.append(file_path)
                results.append((file_path, result))
            else:
                # Check if all mandatory keys are present
                has_all_keys, missing_keys = self.check_mandatory_keys(result)
                if has_all_keys:
                    successful_files.append(file_path)
                    results.append((file_path, result))
                    group_stats["total_successful"] += 1
                else:
                    files_needing_retry.append(file_path)
                    results.append((file_path, result))
                    group_stats["text_first_failed"] += 1
        
        logging.info(f"📊 Phase 1 results: {len(successful_files)} successful, {len(files_needing_retry)} need retry")
        
        # Phase 2: Process failed files with direct file (remote) strategy
        if files_needing_retry:
            logging.info(f"🌐 Phase 2: Processing {len(files_needing_retry)} failed files with direct file (remote) strategy...")
            
            # Create a temporary config with reduced retry limits for direct file processing
            direct_config = self.direct_strategy.config.copy()
            direct_config["max_retries"] = self.file_direct_max_retry
            
            # Create a temporary direct strategy with reduced retries
            temp_direct_strategy = DirectFileProcessingStrategy(direct_config)
            
            direct_results, direct_stats, _ = temp_direct_strategy.process_file_group(
                file_group=files_needing_retry, group_index=group_index, user_prompt=user_prompt, system_prompt=system_prompt, group_id=f"{group_id}_direct_file"
            )
            
            # Update results and stats
            for i, (file_path, result) in enumerate(direct_results):
                # Replace the failed result with the new result
                for j, (existing_path, existing_result) in enumerate(results):
                    if existing_path == file_path:
                        results[j] = (file_path, result)
                        break
                
                # Update stats
                if "error" not in result:
                    has_all_keys, _ = self.check_mandatory_keys(result)
                    if has_all_keys:
                        group_stats["direct_file_successful"] += 1
                        group_stats["total_successful"] += 1
                        group_stats["text_first_failed"] -= 1  # Adjust previous count
                    else:
                        group_stats["direct_file_failed"] += 1
                        group_stats["total_failed"] += 1
                else:
                    group_stats["direct_file_failed"] += 1
                    group_stats["total_failed"] += 1
            
            group_stats["total_tokens"] += direct_stats["total_tokens"]
            
            logging.info(f"📊 Phase 2 results: {group_stats['direct_file_successful']} successful, {group_stats['direct_file_failed']} failed")
        else:
            logging.info("✅ No files need Phase 2 processing - all successful in Phase 1")
        
        # Calculate final stats
        group_stats["total_failed"] = group_stats["text_first_failed"] + group_stats["direct_file_failed"]
        group_stats["processing_time"] = int(time.time() - group_start_time)
        
        logging.info(f"✅ Completed hybrid processing for group {group_index}: {group_stats['total_successful']} successful, {group_stats['total_failed']} failed")
        logging.info(f"📊 Breakdown: Text-first: {group_stats['text_first_successful']} successful, Direct file: {group_stats['direct_file_successful']} successful")
        
        return results, group_stats, group_id


class ImageFirstProcessingStrategy(BaseProcessingStrategy):
    """Strategy for image-first processing: convert PDFs to images, then process with LLM."""
    
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        self.llm_provider = config.get("llm_provider", "openai")
        self.provider_config = config.get("provider_configs", {}).get(self.llm_provider, {})
        
        # Initialize LLM client
        self.llm_client = LLMClientFactory.create_client(self.llm_provider, self.provider_config)
        
        # Initialize token counter for accurate estimation
        from llm_metrics import TokenCounter
        self.token_counter = TokenCounter(self.llm_client, provider=self.llm_provider)
        
        # PDF to image conversion settings
        self.dpi = config.get("pdf_to_image_dpi", 300)
        self.image_format = config.get("pdf_to_image_format", "PNG")
        self.image_quality = config.get("pdf_to_image_quality", 95)
        
        # Use hybrid retry limits if specified
        if "file_direct_max_retry" in config:
            self.max_retries = config["file_direct_max_retry"]
            logging.info(f"🖼️ Image-first strategy using hybrid retry limit: {self.max_retries}")
    
    def process_file_group_with_utils(self, *, file_group: List[str], group_index: int, 
                                     group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files by converting them to images first, then sending to LLM in batches (using utilities)."""
        
        logging.info(f"🖼️ Starting image-first processing for group {group_index} ({group_id}): {len(file_group)} files")
        
        try:
            # Convert all PDFs to images first
            image_paths = []
            for file_path in file_group:
                image_path = self._convert_pdf_to_image(file_path)
                if image_path:
                    image_paths.append((file_path, image_path))
                else:
                    logging.error(f"❌ Failed to convert PDF to image: {file_path}")
            
            if not image_paths:
                logging.error(f"❌ No images could be converted for group {group_index}")
                # Create error results for all files
                results = [(file_path, {"error": "Failed to convert PDF to image"}) for file_path in file_group]
                group_stats = GroupProcessingUtils.create_group_stats_template(len(file_group))
                group_stats.update({"failed_files": len(file_group)})
                return results, group_stats, group_id
            
            # Enhance the prompt to require file_name_llm field in each response
            image_filenames = [Path(image_path).name for _, image_path in image_paths]
            image_filename_list = ", ".join([f"'{name}'" for name in image_filenames])
            
            enhanced_user_prompt = (
                f"{user_prompt}\n\n"
                f"You are processing {len(image_paths)} images with filenames: {image_filename_list}\n\n"
                "Return a JSON array, one object per image. "
                "Each object MUST include a 'file_name_llm' field that identifies the document. "
                "IMPORTANT: The 'file_name_llm' field MUST contain the EXACT image filename from the list above "
                f"(e.g., one of: {image_filename_list}). "
                "This is critical for proper file mapping. "
                "If you do not include the correct file_name_llm, your answer will be ignored."
            )
            
            # Process all images in the group together
            response = self._process_image_group(
                image_paths=image_paths,
                user_prompt=enhanced_user_prompt,
                system_prompt=system_prompt
            )
            
            if "error" in response:
                logging.error(f"LLM API error for group {group_index}: {response['error']}")
                # Create failed results for all files
                results = [(file_path, {"error": response["error"]}) for file_path, _ in image_paths]
                group_stats = GroupProcessingUtils.create_group_stats_template(len(image_paths))
                group_stats.update({"failed_files": len(image_paths)})
                return results, group_stats, group_id
            
            # Parse response and match with files
            if isinstance(response, list):
                file_results = response
            elif isinstance(response, dict) and "results" in response:
                file_results = response["results"]
            else:
                file_results = [response]
            
            # Map outputs to files using utility function
            file_paths = [file_path for file_path, _ in image_paths]
            results = GroupProcessingUtils.map_llm_outputs_to_files(
                file_results=file_results,
                file_paths=file_paths,
                image_paths=image_paths,
                group_index=group_index
            )
            
            # Get token usage from individual results and sum them up
            actual_prompt_tokens = 0
            actual_candidates_tokens = 0
            actual_total_tokens = 0
            
            # Extract token information from individual results
            for result in file_results:
                if isinstance(result, dict):
                    actual_prompt_tokens += result.get('prompt_token_count', 0)
                    actual_candidates_tokens += result.get('candidates_token_count', 0)
                    actual_total_tokens += result.get('total_token_count', 0)
            
            # Distribute tokens using utility function
            file_tokens = GroupProcessingUtils.distribute_tokens_proportionally(
                total_prompt_tokens=actual_prompt_tokens,
                total_candidates_tokens=actual_candidates_tokens,
                total_actual_tokens=actual_total_tokens,
                num_files=len(image_paths),
                token_counter=self.token_counter,
                prompt_text=enhanced_user_prompt,
                system_prompt=system_prompt
            )
            
            # Update results with token information
            for i, (file_path, file_result) in enumerate(results):
                if i < len(file_tokens) and "error" not in file_result:
                    token_stats = file_tokens[i]
                    file_result.update({
                        "estimated_tokens": token_stats["estimated_tokens"],
                        "prompt_token_count": token_stats["prompt_tokens"],
                        "candidates_token_count": token_stats["candidates_tokens"],
                        "total_token_count": token_stats["actual_tokens"]
                    })
            
            # Create group stats using utility function
            group_stats = GroupProcessingUtils.create_group_stats_template(len(image_paths))
            group_stats = GroupProcessingUtils.update_group_stats(
                group_stats=group_stats,
                results=results,
                processing_time=0,  # Will be calculated by caller
                total_tokens=actual_total_tokens,
                estimated_tokens=sum(token["estimated_tokens"] for token in file_tokens)
            )
            
            return results, group_stats, group_id
            
        except Exception as e:
            logging.error(f"❌ Error processing image-first group {group_index}: {e}")
            # Create error results for all files
            results = [(file_path, {"error": str(e)}) for file_path in file_group]
            group_stats = GroupProcessingUtils.create_group_stats_template(len(file_group))
            group_stats.update({"failed_files": len(file_group)})
            return results, group_stats, group_id

    def process_file_group(self, *, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files by converting them to images first, then sending to LLM in batches."""
        return self.process_file_group_with_utils(
            file_group=file_group,
            group_index=group_index,
            group_id=group_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
    
    def _process_image_group(self, *, image_paths: List[Tuple[str, str]], user_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process multiple images in a single API call."""
        try:
            # Convert all images to base64
            import base64
            image_contents = []
            
            for file_path, image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    image_contents.append({
                        "file_path": file_path,
                        "image_data": image_data
                    })
            
            # Create messages array with all images
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message with all images
            user_message_content = [{"type": "text", "text": user_prompt}]
            
            # Add each image to the message
            for img_content in image_contents:
                user_message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_content['image_data']}"
                    }
                })
            
            user_message = {
                "role": "user",
                "content": user_message_content
            }
            messages.append(user_message)
            
            # Use the LLM client's standard method instead of hardcoded OpenAI API
            # Convert image paths to file paths for the LLM client
            image_files = [image_path for _, image_path in image_paths]
            
            # Call the LLM client's standard method
            result = self.llm_client.call_llm(
                files=image_files,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                strategy_type="image_first"
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Error in image group processing: {e}")
            return {"error": str(e)}
    
    def _process_single_file(self, *, file_path: str, user_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process a single file by converting to image and sending to LLM."""
        try:
            # Convert PDF to image
            image_path = self._convert_pdf_to_image(file_path)
            
            if not image_path:
                return {"error": "Failed to convert PDF to image"}
            
            # Convert image to base64 for vision API
            import base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create a custom message with base64 image
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message with image
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    }
                ]
            }
            messages.append(user_message)
            
            # Use the LLM client's standard method instead of hardcoded OpenAI API
            result = self.llm_client.call_llm(
                files=[image_path],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                strategy_type="image_first"
            )
            
            # Clean up temporary image file
            try:
                Path(image_path).unlink()
            except Exception as e:
                logging.warning(f"Failed to clean up temporary image file {image_path}: {e}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in image-first processing for {file_path}: {e}")
            return {"error": str(e)}
    
    def _convert_pdf_to_image(self, pdf_path: str) -> Optional[str]:
        """Convert PDF to PNG image using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            # Create output directory if it doesn't exist
            output_dir = Path("temp_images")
            output_dir.mkdir(exist_ok=True)
            
            # Generate anonymous output filename to prevent information leakage
            # Example: Instead of "xxx.png" (contains sensitive info)
            # We use "image_3423ffcc.png" (anonymous UUID) so LLM cannot extract sensitive data from filename
            import uuid
            anonymous_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
            image_path = output_dir / f"image_{anonymous_id}.png"
            
            # Open PDF and convert first page to image
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                doc.close()
                return None
            
            # Get the first page
            page = doc.load_page(0)
            
            # Convert to image with specified DPI
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 is the default DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            pix.save(str(image_path))
            
            doc.close()
            
            logging.info(f"🖼️ Converted {pdf_path} to {image_path}")
            return str(image_path)
            
        except ImportError:
            logging.error("PyMuPDF (fitz) not available. Install with: pip install PyMuPDF")
            return None
        except Exception as e:
            logging.error(f"Error converting PDF to image: {e}")
            return None


class ProcessingStrategyFactory:
    """Factory for creating processing strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any], streaming: bool = False) -> BaseProcessingStrategy:
        """Create a processing strategy based on type."""
        if strategy_type == "direct_file":
            return DirectFileProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "text_first":
            return TextFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "image_first":
            return ImageFirstProcessingStrategy(config, streaming=streaming)
        elif strategy_type == "hybrid":
            return HybridProcessingStrategy(config, streaming=streaming)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategy types."""
        return ["direct_file", "text_first", "image_first", "hybrid"] 