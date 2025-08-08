"""
Simplified modular parallel processor using modular components.
"""

import copy
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_strategies import ProcessingStrategyFactory
from common.base_monitor import BasePerformanceMonitor
from common.csv_dumper import CSVResultDumper
from config.config_base import STRATEGY_DIRECT_FILE, STRATEGY_TEXT_FIRST, STRATEGY_HYBRID, MODE_PARALLEL, MODE_BATCH

from .checkpoint_manager import CheckpointManager
from .benchmark_tracker import BenchmarkTracker
from .statistics_calculator import StatisticsCalculator


class ModularParallelProcessor:
    """Simplified modular parallel processor with support for multiple processing strategies."""
    
    def __init__(self, config: Dict[str, Any], strategy_type: str = "direct_file", 
                 mode: str = "parallel", max_workers: int = 5,
                 checkpoint_file: str = "modular_checkpoint.pkl", 
                 output_file: str = "modular_results.json",
                 real_time_save: bool = True, run_settings: Dict[str, str] = None,
                 csv_output_file: str = None, benchmark_comparator = None):
        """Initialize the modular parallel processor."""
        self.config = config
        self.strategy_type = strategy_type
        self.mode = mode
        self.max_workers = max_workers
        self.output_file = output_file
        self.real_time_save = real_time_save
        self.run_settings = run_settings or {}
        
        # Initialize modular components
        self.checkpoint_manager = CheckpointManager(checkpoint_file)
        self.benchmark_tracker = BenchmarkTracker(benchmark_comparator, csv_output_file)
        self.csv_output_file = csv_output_file
        
        # Initialize processing strategy
        self.strategy = ProcessingStrategyFactory.create_strategy(strategy_type, config)
        
        # Initialize components
        self.monitor = BasePerformanceMonitor("modular_parallel_processor")
        
        # Initialize CSV dumper
        if csv_output_file:
            csv_output_dir = os.path.dirname(csv_output_file)
            csv_filename = os.path.basename(csv_output_file)
            self.csv_dumper = CSVResultDumper(output_dir=csv_output_dir, custom_filename=csv_filename)
        else:
            csv_filename = f"{self.run_settings.get('strategy', strategy_type)}_{self.run_settings.get('mode', mode)}_{self.run_settings.get('llm_provider', 'unknown')}_{self.run_settings.get('llm_model', 'unknown')}_{datetime.now().strftime('%m-%d-%H-%M-%S')}.csv"
            csv_filename = csv_filename.replace(":", "_").replace("/", "_").replace("-", "_")
            csv_output_dir = "output/results/csv"
            self.csv_dumper = CSVResultDumper(output_dir=csv_output_dir, custom_filename=csv_filename)
        
        # Initialize structured output
        self.structured_output = {
            'run_settings': {
                'strategy': self.run_settings.get('strategy', strategy_type),
                'mode': self.run_settings.get('mode', mode),
                'llm_provider': self.run_settings.get('llm_provider', 'unknown'),
                'llm_model': self.run_settings.get('llm_model', 'unknown')
            },
            'file_stats': {},
            'group_stats': {},
            'retry_stats': {
                'num_files_may_need_retry': 0,
                'num_files_had_retry': 0,
                'percentage_files_had_retry': 0.0,
                'num_file_failed_after_max_retries': 0,
                'actual_tokens_for_retries': 0,
                'retry_prompt_tokens': 0,
                'retry_success_rate': 0.0
            },
            'overall_stats': {},
            'benchmark_errors': {}
        }
        
        # Initialize statistics calculator
        self.stats_calculator = StatisticsCalculator(self.structured_output)
    
    def process_files(self, *, pdf_files: List[str], system_prompt: Optional[str] = None, user_prompt: str) -> Dict[str, Any]:
        """Main method to process files."""
        start_time = time.time()
        
        logging.info(f"🚀 Starting modular batch processing with parameters:")
        logging.info(f"   - Input Directory: {pdf_files[0] if pdf_files else 'None'}")
        logging.info(f"   - Strategy Type: {self.strategy_type}")
        logging.info(f"   - Mode: {self.mode}")
        logging.info(f"   - PDF File Paths provided: {len(pdf_files)}")
        logging.info(f"   - Max Workers: {self.max_workers}")
        logging.info(f"   - Output File: {self.output_file}")
        logging.info(f"   - System Prompt provided: {system_prompt is not None}")
        logging.info(f"   - User Prompt provided: {user_prompt is not None}")
        
        # Load checkpoint if available
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            logging.info("📂 Resuming from checkpoint")
            # TODO: Implement checkpoint resumption logic
        
        # Group files based on mode
        if self.mode == MODE_PARALLEL:
            file_groups = self._group_files_parallel(pdf_files)
        else:
            file_groups = self._group_files_batch(pdf_files)
        
        logging.info(f"Processing {len(pdf_files)} PDF files using {self.strategy_type} strategy in {self.mode} mode")
        
        # Process groups
        all_results = []
        file_dict_for_retries = {}
        
        if self.mode == MODE_PARALLEL:
            all_results = self._process_groups_parallel(
                file_groups=file_groups, user_prompt=user_prompt, system_prompt=system_prompt,
                lot_timestamp_hash="", file_dict_for_retries=file_dict_for_retries
            )
        else:
            all_results = self._process_groups_batch(
                file_groups=file_groups, user_prompt=user_prompt, system_prompt=system_prompt,
                lot_timestamp_hash="", file_dict_for_retries=file_dict_for_retries
            )
        
        # Process retries if needed
        if file_dict_for_retries:
            self._process_retries(
                file_dict_for_retries=file_dict_for_retries, user_prompt=user_prompt,
                system_prompt=system_prompt, lot_timestamp_hash=""
            )
        
        # Calculate final statistics
        self.stats_calculator.calculate_final_statistics(start_time)
        self.stats_calculator.calculate_retry_statistics()
        self.stats_calculator.calculate_token_statistics()
        
        # Generate benchmark error CSV
        self.benchmark_tracker.generate_error_csv()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.stats_calculator.print_summary()
        
        logging.info(f"✅ Processing complete! Results saved to: {self.output_file}")
        
        return self.structured_output
    
    def _group_files_parallel(self, pdf_files: List[str]) -> List[List[str]]:
        """Group files for parallel processing."""
        # Use config parameter to determine files per group
        max_files_per_group = self.config.get("max_num_file_parts_per_request", 8)
        
        file_groups = []
        for i in range(0, len(pdf_files), max_files_per_group):
            group = pdf_files[i:i + max_files_per_group]
            file_groups.append(group)
        
        logging.info(f"📦 Grouped {len(pdf_files)} files into {len(file_groups)} groups (max {max_files_per_group} files per group)")
        return file_groups
    
    def _group_files_batch(self, pdf_files: List[str]) -> List[List[str]]:
        """Group files for batch processing."""
        # Use config parameter to determine files per group
        max_files_per_group = self.config.get("max_num_file_parts_per_request", 8)
        
        file_groups = []
        for i in range(0, len(pdf_files), max_files_per_group):
            group = pdf_files[i:i + max_files_per_group]
            file_groups.append(group)
        
        logging.info(f"📦 Grouped {len(pdf_files)} files into {len(file_groups)} groups (max {max_files_per_group} files per group)")
        return file_groups
    
    def _process_groups_parallel(self, *, file_groups: List[List[str]], user_prompt: str, system_prompt: Optional[str],
                                lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """Process groups in parallel mode."""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_group = {
                executor.submit(self._process_single_group, file_group=group, group_index=i,
                              user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash): i
                for i, group in enumerate(file_groups)
            }
            
            for future in as_completed(future_to_group):
                group_index = future_to_group[future]
                try:
                    results, stats, group_id = future.result()
                    all_results.extend(results)
                    
                    # Store group stats
                    self.structured_output['group_stats'][group_id] = stats
                    
                    # Check for retries
                    self._check_group_for_retries(results, file_dict_for_retries)
                    
                except Exception as e:
                    logging.error(f"❌ Error processing group {group_index}: {e}")
        
        return all_results
    
    def _process_groups_batch(self, *, file_groups: List[List[str]], user_prompt: str, system_prompt: Optional[str],
                             lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """Process groups in batch mode."""
        all_results = []
        
        for group_index, file_group in enumerate(file_groups):
            try:
                results, stats, group_id = self._process_single_group(
                    file_group=file_group, group_index=group_index,
                    user_prompt=user_prompt, system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash
                )
                
                all_results.extend(results)
                self.structured_output['group_stats'][group_id] = stats
                
                # Check for retries
                self._check_group_for_retries(results, file_dict_for_retries)
                
            except Exception as e:
                logging.error(f"❌ Error processing group {group_index}: {e}")
        
        return all_results
    
    def _process_single_group(self, *, file_group: List[str], group_index: int, 
                             user_prompt: str, system_prompt: Optional[str], lot_timestamp_hash: str) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """Process a single group of files."""
        group_id = f"{lot_timestamp_hash}_group_{group_index}"
        
        logging.info(f"🔄 Processing group {group_index} ({group_id}): {len(file_group)} files")
        
        # Process the group using the strategy
        results, stats, _ = self.strategy.process_file_group(
            file_group=file_group,
            group_index=group_index,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            group_id=group_id
        )
        
        # Check benchmark errors for each result
        for file_path, result in results:
            self.benchmark_tracker.check_file_benchmark_errors(file_path, result)
        
        # Save results incrementally if enabled
        if self.real_time_save:
            for file_path, result in results:
                self._save_single_file_result(file_path, result)
        
        logging.info(f"✅ Completed group {group_index}: {stats.get('successful_files', 0)} successful, {stats.get('failed_files', 0)} failed")
        
        return results, stats, group_id
    
    def _check_group_for_retries(self, group_results: List[Tuple[str, Dict]], file_dict_for_retries: Dict[str, Dict]):
        """Check if any files in the group need retries."""
        for file_path, result in group_results:
            if "error" in result or result.get('file_process_result', {}).get('success') == False:
                file_dict_for_retries[file_path] = result
                self.structured_output['retry_stats']['num_files_may_need_retry'] += 1
    
    def _process_retries(self, *, file_dict_for_retries: Dict[str, Dict], user_prompt: str, system_prompt: Optional[str], lot_timestamp_hash: str):
        """Process retries for failed files."""
        if not file_dict_for_retries:
            return
        
        # Get max retries from config
        max_retry_rounds = self.config.get("max_retries", 2)
        retry_round = 1
        
        while file_dict_for_retries and retry_round <= max_retry_rounds:
            logging.info(f"🔄 Processing retries for {len(file_dict_for_retries)} files (round {retry_round}/{max_retry_rounds})")
            
            # Group retry files
            retry_files = list(file_dict_for_retries.keys())
            if self.mode == MODE_PARALLEL:
                retry_groups = self._group_files_parallel(retry_files)
            else:
                retry_groups = self._group_files_batch(retry_files)
            
            # Process retry groups
            new_retry_dict = {}
            
            if self.mode == MODE_PARALLEL:
                self._process_retry_groups_parallel(
                    retry_groups=retry_groups, user_prompt=user_prompt, system_prompt=system_prompt,
                    lot_timestamp_hash=lot_timestamp_hash, file_dict_for_retries=file_dict_for_retries,
                    new_retry_dict=new_retry_dict, retry_round=retry_round
                )
            else:
                self._process_retry_groups_batch(
                    retry_groups=retry_groups, user_prompt=user_prompt, system_prompt=system_prompt,
                    lot_timestamp_hash=lot_timestamp_hash, file_dict_for_retries=file_dict_for_retries,
                    new_retry_dict=new_retry_dict, retry_round=retry_round
                )
            
            # Update file_dict_for_retries with new failures
            file_dict_for_retries.clear()
            file_dict_for_retries.update(new_retry_dict)
            
            retry_round += 1
        
        # Log final retry status
        if file_dict_for_retries:
            logging.warning(f"⚠️ {len(file_dict_for_retries)} files failed after {max_retry_rounds} retry rounds")
        else:
            logging.info(f"✅ All files processed successfully after retries")
    
    def _process_retry_groups_parallel(self, *, retry_groups: List[List[str]], user_prompt: str, system_prompt: Optional[str],
                                      lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict],
                                      new_retry_dict: Dict[str, Dict], retry_round: int):
        """Process retry groups in parallel mode."""
        # Simplified retry processing - just process each group
        for group_index, retry_group in enumerate(retry_groups):
            self._process_retry_group(
                file_group=retry_group, group_index=group_index, user_prompt=user_prompt,
                system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash,
                file_dict_for_retries=file_dict_for_retries, new_retry_dict=new_retry_dict,
                retry_round=retry_round
            )
    
    def _process_retry_groups_batch(self, *, retry_groups: List[List[str]], user_prompt: str, system_prompt: Optional[str],
                                   lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict],
                                   new_retry_dict: Dict[str, Dict], retry_round: int):
        """Process retry groups in batch mode."""
        # Simplified retry processing - just process each group
        for group_index, retry_group in enumerate(retry_groups):
            self._process_retry_group(
                file_group=retry_group, group_index=group_index, user_prompt=user_prompt,
                system_prompt=system_prompt, lot_timestamp_hash=lot_timestamp_hash,
                file_dict_for_retries=file_dict_for_retries, new_retry_dict=new_retry_dict,
                retry_round=retry_round
            )
    
    def _process_retry_group(self, *, file_group: List[str], group_index: int, user_prompt: str, system_prompt: Optional[str],
                            lot_timestamp_hash: str, file_dict_for_retries: Dict[str, Dict],
                            new_retry_dict: Dict[str, Dict], retry_round: int):
        """Process a single retry group."""
        group_id = f"{lot_timestamp_hash}_retry_{retry_round}_group_{group_index}"
        
        logging.info(f"🔄 Processing retry group {group_index} (round {retry_round}): {len(file_group)} files")
        
        # Process the retry group
        results, stats, _ = self.strategy.process_file_group(
            file_group=file_group,
            group_index=group_index,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            group_id=group_id
        )
        
        # Update retry statistics
        self.structured_output['retry_stats']['num_files_had_retry'] += len(file_group)
        self.structured_output['retry_stats']['actual_tokens_for_retries'] += stats.get('total_tokens', 0)
        
        # Check results and update retry dict
        for file_path, result in results:
            if "error" in result or result.get('file_process_result', {}).get('success') == False:
                new_retry_dict[file_path] = result
                self.structured_output['retry_stats']['num_file_failed_after_max_retries'] += 1
            else:
                # Success - remove from retry dict
                if file_path in file_dict_for_retries:
                    del file_dict_for_retries[file_path]
        
        # Store retry group stats
        self.structured_output['group_stats'][group_id] = stats
        
        # Check benchmark errors
        for file_path, result in results:
            self.benchmark_tracker.check_file_benchmark_errors(file_path, result)
        
        logging.info(f"✅ Completed retry group {group_index}: {stats.get('successful_files', 0)} successful, {stats.get('failed_files', 0)} failed")
    
    def _save_single_file_result(self, file_path: str, result: dict):
        """Save a single file result to structured output."""
        self.structured_output['file_stats'][file_path] = result
    
    def save_results(self):
        """Save results to output file."""
        try:
            # Ensure output directory exists
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Add benchmark errors to structured output
            self.structured_output['benchmark_errors'] = self.benchmark_tracker.get_error_stats()
            
            # Save to JSON
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.structured_output, f, indent=2, ensure_ascii=False)
            
            logging.info(f"💾 Results saved to {self.output_file}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save results: {e}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get the structured output results."""
        return self.structured_output 