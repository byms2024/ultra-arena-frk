#!/usr/bin/env python3
"""
Main script for the modular PDF processing system.

This script demonstrates how to use the new modular processing system
with different strategies and configurations.
"""

import argparse
import logging
import os
import sys
import time
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Suppress the specific warning about BATCH_STATE_RUNNING from Google GenAI SDK
warnings.filterwarnings("ignore", message="BATCH_STATE_RUNNING is not a valid JobState")

# Import configurations
import config.config_direct_file as config_direct_file
import config.config_txt_first as config_txt_first
import config.config_image_first as config_image_first
from config.config_base import (
    SYSTEM_PROMPT, USER_PROMPT,
    DEFAULT_STRATEGY_TYPE, DEFAULT_MODE, DEFAULT_MAX_WORKERS,
    DEFAULT_OUTPUT_FILE, DEFAULT_CHECKPOINT_FILE,
    STRATEGY_DIRECT_FILE, STRATEGY_TEXT_FIRST, STRATEGY_IMAGE_FIRST, STRATEGY_HYBRID,
    MODE_PARALLEL, MODE_BATCH
)
import config.config_base as config_base

# Import the modular processor
from processors.modular_parallel_processor import ModularParallelProcessor

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('modular_processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_pdf_files(input_path: str) -> List[str]:
    """Get list of PDF files from input path (file or directory)."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        if input_path.suffix.lower() == '.pdf':
            return [str(input_path)]
        else:
            raise ValueError(f"Input file {input_path} is not a PDF")
    
    elif input_path.is_dir():
        pdf_files = []
        for file_path in input_path.rglob("*.pdf"):
            pdf_files.append(str(file_path))
        return sorted(pdf_files)
    
    else:
        raise ValueError(f"Input path {input_path} does not exist")


def generate_timestamped_filename(strategy: str, mode: str, llm_provider: str, llm_model: str, extension: str = "json") -> str:
    """Generate a timestamped filename with the specified format."""
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    # Clean the model name for filename (remove special characters)
    clean_model = llm_model.replace(":", "_").replace("/", "_").replace("-", "_")
    return f"{strategy}_{mode}_{llm_provider}_{clean_model}_{timestamp}.{extension}"


def get_config_for_strategy(strategy_type: str, llm_provider: str = None, llm_model: str = None, streaming: bool = False) -> Dict[str, Any]:
    """
    Get configuration for a specific strategy type.
    
    Args:
        strategy_type (str): The strategy type
        llm_provider (str, optional): Override LLM provider
        llm_model (str, optional): Override LLM model
        streaming (bool): Whether to use streaming mode
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if strategy_type == STRATEGY_DIRECT_FILE:
        config = {
            "llm_provider": llm_provider or config_base.DEFAULT_LLM_PROVIDER,
            "provider_configs": config_direct_file.PROVIDER_CONFIGS,
            "mandatory_keys": config_direct_file.MANDATORY_KEYS,
            "num_retry_for_mandatory_keys": config_direct_file.NUM_RETRY_FOR_MANDATORY_KEYS,
            "max_num_file_parts_per_request": config_direct_file.MAX_NUM_FILE_PARTS_PER_REQUEST,
            "max_num_file_parts_per_batch": config_direct_file.MAX_NUM_FILE_PARTS_PER_BATCH,
            "max_retries": config_direct_file.API_INFRA_MAX_RETRIES,
            "retry_delay_seconds": config_direct_file.API_INFRA_RETRY_DELAY_SECONDS
        }
        # Override provider and model if specified
        if llm_provider:
            config["llm_provider"] = llm_provider
        if llm_model and llm_provider in config["provider_configs"]:
            config["provider_configs"][llm_provider]["model"] = llm_model
        
        # Add streaming configuration to provider configs
        if streaming and llm_provider in config["provider_configs"]:
            config["provider_configs"][llm_provider]["streaming"] = streaming
        
        return config
    elif strategy_type == STRATEGY_TEXT_FIRST:
        config = {
            "llm_provider": config_txt_first.LOCAL_LLM_PROVIDER,
            "provider_configs": config_txt_first.TEXT_PROVIDER_CONFIGS,
            "pdf_extractor_lib": config_txt_first.PDF_EXTRACTOR_LIB,
            "max_text_length": config_txt_first.MAX_TEXT_LENGTH,
            "mandatory_keys": config_txt_first.MANDATORY_KEYS,
            "num_retry_for_mandatory_keys": config_txt_first.NUM_RETRY_FOR_MANDATORY_KEYS,
            "max_num_file_parts_per_request": config_txt_first.MAX_NUM_FILE_PARTS_PER_REQUEST,
            "max_num_file_parts_per_batch": config_txt_first.MAX_NUM_FILE_PARTS_PER_BATCH,
            "max_retries": config_txt_first.API_INFRA_MAX_RETRIES,
            "retry_delay_seconds": config_txt_first.API_INFRA_RETRY_DELAY_SECONDS
        }
        # Override provider and model if specified
        if llm_provider:
            config["llm_provider"] = llm_provider
        if llm_model and llm_provider in config["provider_configs"]:
            config["provider_configs"][llm_provider]["model"] = llm_model
        return config
    elif strategy_type == STRATEGY_IMAGE_FIRST:
        config = {
            "llm_provider": llm_provider or config_base.DEFAULT_LLM_PROVIDER,
            "provider_configs": config_image_first.IMAGE_PROVIDER_CONFIGS,
            "pdf_to_image_dpi": config_image_first.PDF_TO_IMAGE_DPI,
            "pdf_to_image_format": config_image_first.PDF_TO_IMAGE_FORMAT,
            "pdf_to_image_quality": config_image_first.PDF_TO_IMAGE_QUALITY,
            "mandatory_keys": config_image_first.MANDATORY_KEYS,
            "num_retry_for_mandatory_keys": config_image_first.NUM_RETRY_FOR_MANDATORY_KEYS,
            "max_num_file_parts_per_request": config_image_first.MAX_NUM_FILE_PARTS_PER_REQUEST,
            "max_num_file_parts_per_batch": config_image_first.MAX_NUM_FILE_PARTS_PER_BATCH,
            "max_retries": config_image_first.API_INFRA_MAX_RETRIES,
            "retry_delay_seconds": config_image_first.API_INFRA_RETRY_DELAY_SECONDS,
            "max_file_size_mb": config_image_first.MAX_FILE_SIZE_MB
        }
        # Override provider and model if specified
        if llm_provider:
            config["llm_provider"] = llm_provider
        if llm_model and llm_provider in config["provider_configs"]:
            config["provider_configs"][llm_provider]["model"] = llm_model
        return config
    elif strategy_type == STRATEGY_HYBRID:
        # Combine both configurations for hybrid strategy
        direct_config = get_config_for_strategy(STRATEGY_DIRECT_FILE, streaming=streaming)
        text_config = get_config_for_strategy(STRATEGY_TEXT_FIRST, streaming=streaming)
        # Merge configurations, preferring direct_file settings for conflicts
        hybrid_config = {**text_config, **direct_config}
        return hybrid_config
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")


def run_file_processing_full(*, input_pdf_dir_path: Path, pdf_file_paths: List[Path] = [], 
                       strategy_type: str = STRATEGY_DIRECT_FILE, mode: str = MODE_PARALLEL,
                       system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                       max_workers: int = 5, output_file: str = "modular_results.json",
                       checkpoint_file: str = "modular_checkpoint.pkl", 
                       llm_provider: str = None, llm_model: str = None,
                       csv_output_file: str = None, test_match: bool = False,
                       streaming: bool = False) -> Dict[str, Any]:
    """
    Full-featured synchronous main entry point for batch processing using the modular system.
    
    Args:
        input_pdf_dir_path (Path): Directory containing PDF files to process (used only if pdf_file_paths is empty)
        pdf_file_paths (List[Path]): Optional list of specific PDF file paths to process
        strategy_type (str): Processing strategy - STRATEGY_DIRECT_FILE, STRATEGY_TEXT_FIRST, or STRATEGY_HYBRID
        mode (str): Processing mode - MODE_PARALLEL or MODE_BATCH
        system_prompt (Optional[str]): Optional system prompt for LLM configuration
        user_prompt (Optional[str]): User prompt for processing (if None, uses strategy default)
        max_workers (int): Maximum number of concurrent workers
        output_file (str): Output file path for results
        checkpoint_file (str): Checkpoint file path
    
    Returns:
        Dict[str, Any]: Structured output containing processing results and statistics
    """
    overall_start_time = time.time()
    
    # Log the function parameters
    logging.info(f"🚀 Starting modular batch processing with parameters:")
    logging.info(f"   - Input Directory: {input_pdf_dir_path}")
    logging.info(f"   - Strategy Type: {strategy_type}")
    logging.info(f"   - Mode: {mode}")
    logging.info(f"   - PDF File Paths provided: {len(pdf_file_paths)}")
    logging.info(f"   - Max Workers: {max_workers}")
    logging.info(f"   - Output File: {output_file}")
    logging.info(f"   - System Prompt provided: {system_prompt is not None}")
    logging.info(f"   - User Prompt provided: {user_prompt is not None}")
    
    # Setup logging if not already configured
    if not logging.getLogger().handlers:
        # Set root logger to WARNING to suppress HTTP traffic
        logging.getLogger().setLevel(logging.WARNING)
        
        # Create a custom handler for our application logs
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add handler to root logger
        logging.getLogger().addHandler(handler)
        
        # Enable INFO level for our specific loggers
        logging.getLogger('processors.modular_parallel_processor').setLevel(logging.INFO)
        logging.getLogger('llm_metrics').setLevel(logging.INFO)
        logging.getLogger('common.file_analyzer').setLevel(logging.INFO)
        logging.getLogger('common.base_monitor').setLevel(logging.INFO)
    
    # Determine which files to process
    logging.info("🔍 Determining files to process...")
    if not pdf_file_paths:
        if not input_pdf_dir_path.exists():
            raise FileNotFoundError(f"The specified path '{input_pdf_dir_path}' does not exist.")
        # Get all PDFs in the directory and its subdirectories
        pdf_file_paths = list(input_pdf_dir_path.rglob("*.pdf"))
        logging.info(f"Found {len(pdf_file_paths)} PDF files in directory and subdirectories")
    
    # Convert to string paths for processing
    pdf_files = [str(path) for path in pdf_file_paths if str(path).endswith('.pdf')]
    logging.info(f"Processing {len(pdf_files)} PDF files using {strategy_type} strategy in {mode} mode")
    
    if not pdf_files:
        logging.error(f"No PDF files found to process")
        return {}
    
    try:
        # Get configuration
        logging.info(f"📋 Getting configuration for strategy: {strategy_type}")
        config = get_config_for_strategy(strategy_type, llm_provider=llm_provider, llm_model=llm_model, streaming=streaming)
        logging.info(f"✅ Configuration loaded successfully")
        
        # Determine actual LLM provider and model being used
        if strategy_type == STRATEGY_DIRECT_FILE:
            actual_llm_provider = config.get("llm_provider", "unknown")
            actual_llm_model = config.get("provider_configs", {}).get(actual_llm_provider, {}).get("model", "unknown")
        else:
            actual_llm_provider = config.get("llm_provider", "unknown")
            actual_llm_model = config.get("provider_configs", {}).get(actual_llm_provider, {}).get("model", "unknown")
        
        # Create run settings dictionary
        run_settings = {
            'strategy': strategy_type,
            'mode': mode,
            'llm_provider': actual_llm_provider,
            'llm_model': actual_llm_model
        }
        
        # Initialize benchmark comparator if test_match is enabled
        benchmark_comparator = None
        if test_match:
            try:
                from common.benchmark_comparator import BenchmarkComparator
                benchmark_comparator = BenchmarkComparator()
                logging.info(f"🔍 Benchmark comparison enabled")
            except Exception as e:
                logging.error(f"❌ Failed to initialize benchmark comparator: {e}")
                benchmark_comparator = None
        
        # Create processor
        logging.info(f"🔧 Creating ModularParallelProcessor...")
        processor = ModularParallelProcessor(
            config=config,
            strategy_type=strategy_type,
            mode=mode,
            max_workers=max_workers,
            checkpoint_file=checkpoint_file,
            output_file=output_file,
            real_time_save=True,
            run_settings=run_settings,
            csv_output_file=csv_output_file,
            benchmark_comparator=benchmark_comparator,
            streaming=streaming
        )
        logging.info(f"✅ Processor created successfully")
        
        # Process files
        logging.info(f"🚀 Starting processing with strategy: {strategy_type}, mode: {mode}")
        logging.info(f"📁 Files to process: {pdf_files}")
        results = processor.process_files(pdf_files=pdf_files, system_prompt=system_prompt, user_prompt=user_prompt)
        logging.info(f"✅ Processing completed, got structured output with keys: {list(results.keys())}")
        
        # Note: Benchmark comparison is already handled within the processor
        # The processor.process_files() method includes benchmark comparison for all processed files
        logging.info(f"📊 Benchmark comparison completed within processor")
        
        # Print summary
        # Summary is already printed by the processor
        
        logging.info(f"✅ Processing complete! Results saved to: {output_file}")
        return results
        
    except Exception as e:
        logging.error(f"❌ Processing failed: {e}", exc_info=True)
        raise


def run_file_processing(*, input_pdf_dir_path: Path, pdf_file_paths: List[Path] = [],
                      strategy_type: str = DEFAULT_STRATEGY_TYPE, mode: str = DEFAULT_MODE,
                      system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                      max_workers: int = DEFAULT_MAX_WORKERS, output_file: str = DEFAULT_OUTPUT_FILE,
                      checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
                      llm_provider: str = None, llm_model: str = None,
                      csv_output_file: str = None, test_match: bool = False,
                      streaming: bool = False) -> Dict[str, Any]:
    """
    Simplified wrapper function for file processing with default values from config_base.py.
    
    This function provides a simpler interface to run_file_processing_full() by using
    default values for unspecified parameters from config_base.py.
    
    Args:
        input_pdf_dir_path (Path): Directory containing PDF files to process (used only if pdf_file_paths is empty)
        pdf_file_paths (List[Path]): Optional list of specific PDF file paths to process
        strategy_type (str): Processing strategy (default: DEFAULT_STRATEGY_TYPE)
        mode (str): Processing mode (default: DEFAULT_MODE)
        system_prompt (Optional[str]): Optional system prompt for LLM configuration
        user_prompt (Optional[str]): User prompt for processing
        max_workers (int): Maximum number of concurrent workers
        output_file (str): Output file path
        checkpoint_file (str): Checkpoint file path
    
    Returns:
        Dict[str, Any]: Structured output containing processing results and statistics
    """
    return run_file_processing_full(
        input_pdf_dir_path=input_pdf_dir_path,
        pdf_file_paths=pdf_file_paths,
        strategy_type=strategy_type,
        mode=mode,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_workers=max_workers,
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        llm_provider=llm_provider,
        llm_model=llm_model,
        csv_output_file=csv_output_file,
        test_match=test_match,
        streaming=streaming
    )


def run_file_processing_simple(*, input_pdf_dir_path: Path, pdf_file_paths: List[Path] = []) -> Dict[str, Any]:
    """
    Ultra-simplified wrapper function for file processing with minimal parameters.
    
    This function provides the simplest possible interface to file processing by using
    all default values except for the essential parameters.
    
    Args:
        input_pdf_dir_path (Path): Directory containing PDF files to process (used only if pdf_file_paths is empty)
        pdf_file_paths (List[Path]): Optional list of specific PDF file paths to process
        system_prompt (Optional[str]): Optional system prompt for LLM configuration
        user_prompt (Optional[str]): User prompt for processing
    
    Returns:
        Dict[str, Any]: Filtered structured output containing only file_stats, overall_stats, and overall_cost
    """
    # Get full results from run_file_processing
    full_results = run_file_processing(
        input_pdf_dir_path=input_pdf_dir_path,
        pdf_file_paths=pdf_file_paths,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT
    )
    
    # Filter to only include the essential parts
    filtered_results = {
        "run_settings": full_results.get("run_settings", {}),
        "file_stats": {},
        "overall_stats": full_results.get("overall_stats", {}),
        "overall_cost": full_results.get("overall_cost", {})
    }
    
    # Filter file_stats to only include file_model_output for each file
    for file_path, file_data in full_results.get("file_stats", {}).items():
        if "file_model_output" in file_data:
            filtered_results["file_stats"][file_path] = {
                "file_model_output": file_data["file_model_output"]
            }
    
    return filtered_results


def run_combo_processing(combo_config_path: str, test_match: bool = False, combo_name: str = None, streaming: bool = False) -> int:
    """Run combination processing based on configuration file."""
    try:
        # Import the combo configuration
        import importlib.util
        spec = importlib.util.spec_from_file_location("combo_config", combo_config_path)
        combo_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(combo_config_module)
        
        combo_config = combo_config_module.combo_config
        
        logging.info(f"📋 Loaded combo configuration from: {combo_config_path}")
        logging.info(f"📊 Found {len(combo_config)} combo(s)")
        
        # Filter combos if specific combo_name is provided
        if combo_name:
            if combo_name not in combo_config:
                logging.error(f"❌ Combo '{combo_name}' not found in configuration. Available combos: {list(combo_config.keys())}")
                return 1
            combo_config = {combo_name: combo_config[combo_name]}
            logging.info(f"🎯 Running specific combo: {combo_name}")
        else:
            logging.info(f"🔄 Running all {len(combo_config)} combos")
        
        for combo_name, combo_settings in combo_config.items():
            logging.info(f"🚀 Starting combo: {combo_name}")
            
            # Create combo output directory using config-driven paths
            combo_output_dir = Path(f"{config_base.OUTPUT_COMBO_DIR}/{combo_name}")
            
            # Check if combo directory already exists and backup if it does
            if combo_output_dir.exists():
                # Generate backup timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
                
                # Create backup directory structure
                backup_base_dir = Path(config_base.OUTPUT_COMBO_BACKUP_DIR)
                backup_base_dir.mkdir(parents=True, exist_ok=True)
                backup_dir = backup_base_dir / f"{combo_name}_bk_{timestamp}"
                
                logging.info(f"📦 Existing combo directory found: {combo_output_dir}")
                logging.info(f"🔄 Moving to backup: {backup_dir}")
                
                # Move existing directory to backup
                combo_output_dir.rename(backup_dir)
                logging.info(f"✅ Backup created: {backup_dir}")
            
            # Create fresh combo directories
            combo_csv_dir = combo_output_dir / config_base.OUTPUT_COMBO_CSV_DIR
            combo_json_dir = combo_output_dir / config_base.OUTPUT_COMBO_JSON_DIR
            
            combo_csv_dir.mkdir(parents=True, exist_ok=True)
            combo_json_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"📁 Created fresh combo directories: {combo_output_dir}")
            
            # Get input files for this combo
            input_files_path = combo_settings["input_files"]
            logging.info(f"📂 Processing input files from: {input_files_path}")
            
            # Get PDF files
            pdf_files = get_pdf_files(input_files_path)
            logging.info(f"📊 Found {len(pdf_files)} PDF files")
            
            if not pdf_files:
                logging.warning(f"⚠️ No PDF files found in {input_files_path}")
                continue
            
            # Convert to Path objects
            pdf_file_paths = [Path(f) for f in pdf_files]
            
            # Process each parameter group
            parameter_group_names = combo_settings["parameter_groups"]
            logging.info(f"🔄 Processing {len(parameter_group_names)} parameter group(s)")
            
            # Import parameter groups definition
            from config.run_param_grps_def import param_grps
            
            for group_name in parameter_group_names:
                if group_name not in param_grps:
                    logging.error(f"❌ Parameter group '{group_name}' not found in param_grps definition")
                    continue
                    
                group_params = param_grps[group_name]
                logging.info(f"⚙️ Processing parameter group: {group_name}")
                logging.info(f"📋 Parameters: {group_params}")
                
                # Extract parameters
                strategy = group_params.get("strategy", STRATEGY_DIRECT_FILE)
                mode = group_params.get("mode", MODE_PARALLEL)
                provider = group_params.get("provider", "google")
                model = group_params.get("model", "gemini-2.5-flash")
                temperature = group_params.get("temperature", 0.1)
                
                # Generate timestamped filenames for this group
                json_filename = generate_timestamped_filename(strategy, mode, provider, model, "json")
                csv_filename = generate_timestamped_filename(strategy, mode, provider, model, "csv")
                
                output_file = str(combo_json_dir / json_filename)
                csv_output_file = str(combo_csv_dir / csv_filename)
                
                logging.info(f"📄 Output files: {output_file}, {csv_output_file}")
                
                try:
                    # Run the processing
                    results = run_file_processing(
                        input_pdf_dir_path=Path(input_files_path),
                        pdf_file_paths=pdf_file_paths,
                        strategy_type=strategy,
                        mode=mode,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=USER_PROMPT,
                        max_workers=5,
                        output_file=output_file,
                        checkpoint_file=f"modular_checkpoint_{combo_name}_{group_name}.pkl",
                        llm_provider=provider,
                        llm_model=model,
                        csv_output_file=csv_output_file,
                        test_match=test_match,
                        streaming=streaming
                    )
                    
                    logging.info(f"✅ Successfully processed combo {combo_name}, group {group_name}")
                    
                except Exception as e:
                    logging.error(f"❌ Error processing combo {combo_name}, group {group_name}: {str(e)}")
                    continue
            
            logging.info(f"✅ Completed combo: {combo_name}")
        
        return 0
        
    except Exception as e:
        logging.error(f"❌ Error in combo processing: {str(e)}")
        return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Modular PDF Processing System')
    
    # Combo configuration argument (first argument)
    parser.add_argument('--combo_config_path', type=str, 
                       help='Path to combo configuration file. If specified, other arguments will be ignored.')
    parser.add_argument('--combo_name', type=str, 
                       help='Specific combo name to run from the combo configuration file. If not specified, all combos will be run.')
    
    # Input/output arguments
    parser.add_argument('input', nargs='?', help='Input PDF file or directory containing PDF files (ignored if --combo_config_path is specified)')
    parser.add_argument('--output', '-o', default='modular_results.json', 
                       help='Output file path (default: modular_results.json)')
    
    # Strategy arguments
    parser.add_argument('--strategy', '-s', choices=[STRATEGY_DIRECT_FILE, STRATEGY_TEXT_FIRST, STRATEGY_IMAGE_FIRST, STRATEGY_HYBRID], 
                       default=STRATEGY_DIRECT_FILE, help=f'Processing strategy (default: {STRATEGY_DIRECT_FILE})')
    parser.add_argument('--mode', '-m', choices=[MODE_PARALLEL, MODE_BATCH], 
                       default=MODE_PARALLEL, help=f'Processing mode (default: {MODE_PARALLEL})')
    
    # LLM Provider and Model arguments
    parser.add_argument('--provider', choices=['google', 'openai', 'ollama', 'deepseek', 'claude', 'huggingface'], 
                       default=None, help='LLM provider (default: uses strategy default)')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM model name (default: uses strategy default)')
    
    # Processing arguments
    parser.add_argument('--max-workers', '-w', type=int, default=5,
                       help='Maximum number of concurrent workers (default: 5)')
    parser.add_argument('--checkpoint', '-c', default='modular_checkpoint.pkl',
                       help='Checkpoint file path (default: modular_checkpoint.pkl)')
    
    # Prompt arguments
    parser.add_argument('--system-prompt', type=str, default=None,
                       help='Optional system prompt for LLM configuration')
    parser.add_argument('--user-prompt', type=str, default=None,
                       help='User prompt for processing (default: uses strategy default)')
    
    # Logging arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-real-time-save', action='store_true',
                       help='Disable real-time saving of results')
    
    # Benchmark testing arguments
    parser.add_argument('--test-match', action='store_true',
                       help='Enable benchmark comparison testing against benchmark file')
    
    # Streaming arguments
    parser.add_argument('--streaming', action='store_true',
                       help='Enable streaming mode for LLM responses (currently only supported for Google GenAI)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Check if combo configuration is specified
        if args.combo_config_path:
            logging.info(f"🔄 Running in combo mode with config: {args.combo_config_path}")
            return run_combo_processing(args.combo_config_path, test_match=args.test_match, combo_name=args.combo_name, streaming=args.streaming)
        
        # Regular single processing mode
        if not args.input:
            logging.error("❌ Input path is required when not using combo configuration")
            return 1
            
        # Get PDF files
        logging.info(f"📁 Scanning for PDF files in: {args.input}")
        pdf_files = get_pdf_files(args.input)
        logging.info(f"📊 Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            logging.warning("⚠️ No PDF files found")
            return 1
        
        # Convert to Path objects for run_file_processing_full
        pdf_file_paths = [Path(f) for f in pdf_files]
        
        # Log which prompts are being used
        if args.system_prompt is None:
            logging.info("📝 Using example SYSTEM_PROMPT from config.py")
        if args.user_prompt is None:
            logging.info("📝 Using example USER_PROMPT from config.py")

        # Get configuration to determine actual LLM provider and model
        config = get_config_for_strategy(args.strategy, llm_provider=args.provider, llm_model=args.model, streaming=args.streaming)
        
        # Determine actual LLM provider and model being used
        if args.strategy == STRATEGY_DIRECT_FILE:
            actual_llm_provider = config.get("llm_provider", "unknown")
            actual_llm_model = config.get("provider_configs", {}).get(actual_llm_provider, {}).get("model", "unknown")
        else:
            actual_llm_provider = config.get("llm_provider", "unknown")
            actual_llm_model = config.get("provider_configs", {}).get(actual_llm_provider, {}).get("model", "unknown")
        
        # Generate timestamped filenames
        if args.output == 'modular_results.json':  # Only auto-generate if using default name
            json_filename = generate_timestamped_filename(args.strategy, args.mode, actual_llm_provider, actual_llm_model, "json")
            csv_filename = generate_timestamped_filename(args.strategy, args.mode, actual_llm_provider, actual_llm_model, "csv")
            output_file = f"{config_base.OUTPUT_NON_COMBO_JSON_DIR}/{json_filename}"
            csv_output_file = f"{config_base.OUTPUT_NON_COMBO_CSV_DIR}/{csv_filename}"
        else:
            output_file = args.output
            csv_output_file = args.output.replace('.json', '.csv') if args.output.endswith('.json') else f"{args.output}.csv"

        # Ensure output directories exist using config-driven paths
        os.makedirs(config_base.OUTPUT_NON_COMBO_CSV_DIR, exist_ok=True)
        os.makedirs(config_base.OUTPUT_NON_COMBO_JSON_DIR, exist_ok=True)

        # Use the simplified run_file_processing wrapper function
        # If no prompts provided, use the example prompts from config.py
        system_prompt = args.system_prompt if args.system_prompt is not None else SYSTEM_PROMPT
        user_prompt = args.user_prompt if args.user_prompt is not None else USER_PROMPT
                
        results = run_file_processing(
            input_pdf_dir_path=Path(args.input),
            pdf_file_paths=pdf_file_paths,
            strategy_type=args.strategy,
            mode=args.mode,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_workers=args.max_workers,
            output_file=output_file,
            checkpoint_file=args.checkpoint,
            llm_provider=args.provider,
            llm_model=args.model,
            csv_output_file=csv_output_file,
            test_match=args.test_match,
            streaming=args.streaming
        )
        
        logging.info(f"✅ Processing complete! Results saved to: {output_file}")
        return 0
        
    except Exception as e:
        logging.error(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 