"""
Benchmark Manager - Main orchestrator for benchmark comparison functionality.

This class provides a unified interface for all benchmark comparison operations
including validation, error tracking, and reporting.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .benchmark_validator import BenchmarkValidator
from .benchmark_reporter import BenchmarkReporter


class BenchmarkManager:
    """
    Main manager class for benchmark comparison operations.
    
    This class orchestrates all benchmark comparison functionality including:
    - Loading and managing benchmark data
    - Validating extracted results against benchmarks
    - Tracking errors and statistics
    - Generating reports and CSV files
    """
    
    def __init__(self, benchmark_file_path: str, mandatory_keys: List[str]):
        """
        Initialize the benchmark manager.
        
        Args:
            benchmark_file_path: Path to the benchmark Excel file
            mandatory_keys: List of mandatory keys to validate
        """
        self.benchmark_file_path = benchmark_file_path
        self.mandatory_keys = mandatory_keys
        
        # Initialize components
        self.validator = BenchmarkValidator(benchmark_file_path, mandatory_keys)
        self.reporter = BenchmarkReporter()
        
        # Statistics tracking
        self.total_unmatched_fields = 0
        self.total_unmatched_files = 0
        self.unmatched_fields_data = []
        
        logging.info(f"🔍 Benchmark Manager initialized with {len(mandatory_keys)} mandatory keys")
    
    def validate_file_results(self, file_path: str, extracted_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted results for a single file against benchmark data.
        
        Args:
            file_path: Path to the processed file
            extracted_result: Extracted data from the file
            
        Returns:
            Dictionary containing validation results and statistics
        """
        validation_result = self.validator.validate_single_file(file_path, extracted_result)
        
        # Track statistics
        if validation_result['has_errors']:
            self.total_unmatched_files += 1
            self.total_unmatched_fields += validation_result['unmatched_count']
            
            # Add to unmatched fields data for CSV generation
            for field_error in validation_result['field_errors']:
                self.unmatched_fields_data.append({
                    'file_path': file_path,
                    'field_name': field_error['field_name'],
                    'benchmark_value': field_error['benchmark_value'],
                    'extracted_value': field_error['extracted_value'],
                    'timestamp': datetime.now().isoformat()
                })
        
        return validation_result
    
    def validate_batch_results(self, file_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate results for multiple files against benchmark data.
        
        Args:
            file_results: Dictionary mapping file paths to extracted results
            
        Returns:
            Dictionary containing batch validation results and statistics
        """
        batch_results = {
            'total_files': len(file_results),
            'files_with_errors': 0,
            'total_unmatched_fields': 0,
            'file_results': {}
        }
        
        for file_path, result in file_results.items():
            file_validation = self.validate_file_results(file_path, result)
            batch_results['file_results'][file_path] = file_validation
            
            if file_validation['has_errors']:
                batch_results['files_with_errors'] += 1
                batch_results['total_unmatched_fields'] += file_validation['unmatched_count']
        
        return batch_results
    
    def get_benchmark_errors(self) -> Dict[str, Any]:
        """
        Get current benchmark error statistics.
        
        Returns:
            Dictionary containing error statistics
        """
        return {
            'total_unmatched_fields': self.total_unmatched_fields,
            'total_unmatched_files': self.total_unmatched_files
        }
    
    def generate_error_csv(self, output_path: str) -> str:
        """
        Generate CSV file with detailed error information.
        
        Args:
            output_path: Path where to save the error CSV file
            
        Returns:
            Path to the generated CSV file
        """
        if not self.unmatched_fields_data:
            logging.info("📊 No benchmark errors to report")
            return None
        
        csv_path = self.reporter.generate_error_csv(
            self.unmatched_fields_data, 
            output_path
        )
        
        logging.info(f"💾 Error CSV file saved: {csv_path}")
        return csv_path
    
    def generate_benchmark_report(self, output_dir: str) -> Dict[str, str]:
        """
        Generate comprehensive benchmark comparison report.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary containing paths to generated report files
        """
        report_files = self.reporter.generate_comprehensive_report(
            benchmark_errors=self.get_benchmark_errors(),
            unmatched_data=self.unmatched_fields_data,
            output_dir=output_dir
        )
        
        logging.info(f"📊 Benchmark report generated: {report_files}")
        return report_files
    
    def reset_statistics(self):
        """Reset all benchmark statistics."""
        self.total_unmatched_fields = 0
        self.total_unmatched_files = 0
        self.unmatched_fields_data = []
        logging.info("🔄 Benchmark statistics reset")
    
    def get_benchmark_value(self, file_path: str, field_name: str) -> Optional[str]:
        """
        Get benchmark value for a specific file and field.
        
        Args:
            file_path: Path to the file
            field_name: Name of the field
            
        Returns:
            Benchmark value or None if not found
        """
        return self.validator.get_benchmark_value(file_path, field_name) 