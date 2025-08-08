"""
Text-first processing strategy.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .base_strategy import BaseProcessingStrategy
from llm_client.llm_client_factory import LLMClientFactory
from llm_metrics import TokenCounter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processors.file_mapping_utils import FileMappingFactory, create_text_first_file_path_mapper, FilePathAwareLLMClient


class TextFirstProcessingStrategy(BaseProcessingStrategy):
    """Strategy for processing files by extracting text first, then sending to LLM."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_provider = config.get("llm_provider", "ollama")
        self.llm_config = config.get("provider_configs", {}).get(self.llm_provider, {})
        self.llm_client = LLMClientFactory.create_client(self.llm_provider, self.llm_config)
        
        # Initialize token counter for accurate estimation
        self.token_counter = TokenCounter(self.llm_client, provider=self.llm_provider)
    
    def process_file_group(self, *, file_group: List[str], group_index: int, 
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process files by extracting text first, then sending to LLM."""
        
        group_start_time = time.time()
        logging.info(f"📝 Starting text-first processing for group {group_index} ({group_id}): {len(file_group)} files")
        
        # Convert PDFs to text files first and maintain mapping
        text_file_group = []
        pdf_to_text_mapping = {}  # Map text file path -> original PDF path
        
        for file_path in file_group:
            text_path = self._convert_pdf_to_text(file_path)
            if text_path:
                text_file_group.append(text_path)
                pdf_to_text_mapping[text_path] = file_path  # Store mapping
            else:
                logging.error(f"❌ Failed to convert PDF to text: {file_path}")
        
        if not text_file_group:
            logging.error(f"❌ No text files could be created for group {group_index}")
            # Create error results for all files
            results = [(file_path, {"error": "Failed to convert PDF to text"}) for file_path in file_group]
            group_stats = {
                "total_files": len(file_group),
                "successful_files": 0,
                "failed_files": len(file_group),
                "total_tokens": 0,
                "estimated_tokens": 0,
                "processing_time": int(time.time() - group_start_time)
            }
            return results, group_stats, group_id
        
        # Process text files using the direct file processor with text_first strategy
        from llm_strategies.direct_file_strategy import DirectFileProcessingStrategy
        
        # Create a direct file processor for text first strategy
        direct_file_processor = DirectFileProcessingStrategy(self.config)
        direct_file_processor.llm_client = self.llm_client
        
        # Create text first file path mapper
        file_path_mapper = create_text_first_file_path_mapper()
        for text_path, pdf_path in pdf_to_text_mapping.items():
            file_path_mapper.add_mapping(pdf_path, text_path)  # original_path, processed_path
        
        # Call the direct file processor with text_first strategy to use TextFirstFilePathMapper
        results, group_stats, _ = direct_file_processor.process_file_group(
            file_group=text_file_group,
            group_index=group_index,
            group_id=group_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            strategy_type="text_first",
            file_path_mapper=file_path_mapper
        )
        
        return results, group_stats, group_id
    
    def _process_text_files_with_mapping(self, *, text_file_group: List[str], pdf_to_text_mapping: Dict[str, str],
                                       group_index: int, group_id: str = "", system_prompt: Optional[str] = None, 
                                       user_prompt: str, group_start_time: float) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """
        Process text files using proper file path mapping.
        
        This method ensures that the LLM receives the original PDF paths in FILE_PATH information
        while still processing the converted text files.
        """
        # Create text first file path mapper
        file_path_mapper = create_text_first_file_path_mapper()
        for text_path, pdf_path in pdf_to_text_mapping.items():
            file_path_mapper.add_mapping(pdf_path, text_path)  # original_path, converted_path
        
        # Create a file path aware LLM client wrapper
        file_path_aware_client = FilePathAwareLLMClient(self.llm_client, file_path_mapper)
        
        # Process each text file individually with proper mapping
        results = []
        group_stats = {
            "total_files": len(text_file_group),
            "successful_files": 0,
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "processing_time": 0
        }
        
        for text_path in text_file_group:
            try:
                # Enhance the prompt to require file_name_llm field in each response (like direct file strategy)
                enhanced_user_prompt = (
                    f"{user_prompt}\n\n"
                    "IMPORTANT: Use the FILE_PATH information provided to identify each document. "
                    "Return a JSON array, one object per file. "
                    "Each object MUST include a 'file_name_llm' field that contains the original filename from the FILE_PATH. "
                    "For example, if FILE_PATH shows 'bbbb/aaaa/xxxx.pdf', "
                    "then file_name_llm should be 'xxxx.pdf'. "
                    "If you do not include the correct file_name_llm, your answer will be ignored."
                )
                
                # Send text file to LLM with proper FILE_PATH mapping
                response = self._retry_with_backoff(
                    file_path_aware_client.call_llm, 
                    files=[text_path], 
                    system_prompt=system_prompt, 
                    user_prompt=enhanced_user_prompt
                )
                
                if "error" in response:
                    pdf_path = pdf_to_text_mapping[text_path]
                    results.append((pdf_path, {"error": response["error"]}))
                    group_stats["failed_files"] += 1
                else:
                    # Parse response using the same logic as direct file strategy
                    if isinstance(response, list):
                        # Response is already a list of results
                        file_results = response
                    else:
                        # Single result, wrap in list
                        file_results = [response]
                    
                    # Map outputs to files using provider-specific file mapping strategy
                    # For text first strategy, we need to map the original PDF paths, not the temporary text paths
                    pdf_path = pdf_to_text_mapping[text_path]
                    file_mapping_strategy = FileMappingFactory.create_strategy(self.llm_provider)
                    mapped_results = file_mapping_strategy.map_outputs_to_files(file_results, [pdf_path], group_index)
                    
                    # Get the mapped result for this file
                    if mapped_results and len(mapped_results) > 0:
                        pdf_path = pdf_to_text_mapping[text_path]
                        file_result = mapped_results[0][1]  # Get the result part
                        
                        if "error" not in file_result:
                            # Calculate token estimate
                            try:
                                file_token_estimate = self.token_counter.count_file_content_tokens(pdf_path)
                                file_result["estimated_tokens"] = file_token_estimate
                            except Exception as e:
                                logging.warning(f"Failed to estimate tokens for {Path(pdf_path).name}: {e}")
                                file_result["estimated_tokens"] = 0
                            
                            results.append((pdf_path, file_result))
                            group_stats["successful_files"] += 1
                            
                            # Add token usage if available
                            if "total_token_count" in file_result:
                                group_stats["total_tokens"] += file_result["total_token_count"]
                        else:
                            results.append((pdf_path, file_result))
                            group_stats["failed_files"] += 1
                    else:
                        pdf_path = pdf_to_text_mapping[text_path]
                        results.append((pdf_path, {"error": "No result returned for this file"}))
                        group_stats["failed_files"] += 1
                        
            except Exception as e:
                logging.error(f"Error processing text file {text_path}: {e}")
                pdf_path = pdf_to_text_mapping[text_path]
                results.append((pdf_path, {"error": str(e)}))
                group_stats["failed_files"] += 1
        
        group_stats["processing_time"] = int(time.time() - group_start_time)
        
        # Calculate estimated tokens for the group
        try:
            original_file_group = list(pdf_to_text_mapping.values())
            estimation = self.token_counter.estimate_total_tokens_for_group(user_prompt, original_file_group)
            group_stats["estimated_tokens"] = estimation["total_estimated_tokens"]
        except Exception as e:
            logging.warning(f"Failed to calculate accurate token estimation: {e}")
            group_stats["estimated_tokens"] = 0
        
        logging.info(f"✅ Completed text-first processing for group {group_index}: {group_stats['successful_files']} successful, {group_stats['failed_files']} failed")
        
        return results, group_stats, group_id
    
    def _convert_pdf_to_text(self, pdf_path: str) -> Optional[str]:
        """Convert PDF to text file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            import tempfile
            import os
            
            doc = fitz.open(pdf_path)
            text_content = ""
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            
            doc.close()
            
            # Create a temporary text file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(text_content)
                temp_file_path = temp_file.name
            
            logging.info(f"📝 Converted {pdf_path} to {temp_file_path}")
            return temp_file_path
            
        except ImportError:
            logging.error("PyMuPDF (fitz) not available. Install with: pip install PyMuPDF")
            return None
        except Exception as e:
            logging.error(f"Error converting PDF to text: {e}")
            return None
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry function with exponential backoff."""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
    
    def _process_single_file(self, *, file_path: str, user_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process a single file by extracting text first, then sending to LLM."""
        try:
            # Extract text from PDF
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            text_content = ""
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            
            doc.close()
            
            # Create a temporary text file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(text_content)
                temp_file_path = temp_file.name
            
            try:
                # Send text file to LLM
                response = self._retry_with_backoff(
                    self.llm_client.call_llm, 
                    files=[temp_file_path], 
                    system_prompt=system_prompt, 
                    user_prompt=user_prompt
                )
                
                # Clean up temporary file
                os.unlink(temp_file_path)
                
                if "error" in response:
                    return {"error": response["error"]}
                
                # Parse response
                if isinstance(response, list) and len(response) > 0:
                    return response[0]  # Return first result
                elif isinstance(response, dict):
                    return response
                else:
                    return {"error": "Unexpected response format"}
                    
            except Exception as e:
                # Clean up temporary file on error
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                raise e
                
        except ImportError:
            return {"error": "PyMuPDF (fitz) not available. Install with: pip install PyMuPDF"}
        except Exception as e:
            logging.error(f"Error in text-first processing for {file_path}: {e}")
            return {"error": str(e)} 