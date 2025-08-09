"""
Benchmark Adapter - Backward compatibility layer for existing code.

This module provides adapter classes to maintain compatibility with the existing
benchmark comparison code while using the new refactored benchmark classes.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .benchmark_manager import BenchmarkManager


class BenchmarkComparatorAdapter:
    """
    Adapter class to maintain backward compatibility with existing BenchmarkComparator.
    
    This class provides the same interface as the original BenchmarkComparator
    but uses the new refactored BenchmarkManager internally.
    """
    
    def __init__(self, benchmark_file_path: str, mandatory_keys: List[str]):
        """
        Initialize the adapter with benchmark file and mandatory keys.
        
        Args:
            benchmark_file_path: Path to the benchmark Excel file
            mandatory_keys: List of mandatory keys to validate
        """
        self.benchmark_manager = BenchmarkManager(benchmark_file_path, mandatory_keys)
        self.benchmark_file_path = benchmark_file_path
        self.mandatory_keys = mandatory_keys
        
        logging.info(f"🔗 Benchmark Adapter initialized for backward compatibility")
    
    def get_benchmark_errors(self) -> Dict[str, Any]:
        """
        Get benchmark error statistics (backward compatibility method).
        
        Returns:
            Dictionary containing error statistics
        """
        return self.benchmark_manager.get_benchmark_errors()
    
    def save_unmatched_to_csv(self, output_path: str) -> str:
        """
        Save unmatched data to CSV (backward compatibility method).
        
        Args:
            output_path: Path where to save the CSV file
            
        Returns:
            Path to the generated CSV file
        """
        return self.benchmark_manager.generate_error_csv(output_path)
    
    def compare_results(self, file_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare results against benchmark (backward compatibility method).
        
        Args:
            file_results: Dictionary mapping file paths to extracted results
            
        Returns:
            Dictionary containing comparison results
        """
        return self.benchmark_manager.validate_batch_results(file_results)
    
    def get_benchmark_value(self, file_path: str, field_name: str) -> Optional[str]:
        """
        Get benchmark value for a specific file and field (backward compatibility method).
        
        Args:
            file_path: Path to the file
            field_name: Name of the field
            
        Returns:
            Benchmark value or None if not found
        """
        return self.benchmark_manager.get_benchmark_value(file_path, field_name)
    
    def validate_single_file(self, file_path: str, extracted_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate single file results (backward compatibility method).
        
        Args:
            file_path: Path to the processed file
            extracted_result: Extracted data from the file
            
        Returns:
            Dictionary containing validation results
        """
        return self.benchmark_manager.validate_file_results(file_path, extracted_result)
    
    # Additional convenience methods for the new functionality
    def generate_comprehensive_report(self, output_dir: str) -> Dict[str, str]:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary containing paths to generated report files
        """
        return self.benchmark_manager.generate_benchmark_report(output_dir)
    
    def reset_statistics(self):
        """Reset benchmark statistics."""
        self.benchmark_manager.reset_statistics()


# Convenience function to create a benchmark manager with default settings
def create_benchmark_manager(benchmark_file_path: str, 
                           mandatory_keys: List[str] = None) -> BenchmarkManager:
    """
    Create a benchmark manager with default mandatory keys if not provided.
    
    Args:
        benchmark_file_path: Path to the benchmark Excel file
        mandatory_keys: List of mandatory keys (defaults to standard keys if None)
        
    Returns:
        Configured BenchmarkManager instance
    """
    if mandatory_keys is None:
        try:
            from config.config_base import MANDATORY_KEYS as _MANDATORY_KEYS
            mandatory_keys = _MANDATORY_KEYS or []
        except Exception:
            mandatory_keys = []
    
    return BenchmarkManager(benchmark_file_path, mandatory_keys)


# Convenience function to create a backward-compatible adapter
def create_benchmark_adapter(benchmark_file_path: str, 
                           mandatory_keys: List[str] = None) -> BenchmarkComparatorAdapter:
    """
    Create a backward-compatible benchmark adapter.
    
    Args:
        benchmark_file_path: Path to the benchmark Excel file
        mandatory_keys: List of mandatory keys (defaults to standard keys if None)
        
    Returns:
        Configured BenchmarkComparatorAdapter instance
    """
    if mandatory_keys is None:
        try:
            from config.config_base import MANDATORY_KEYS as _MANDATORY_KEYS
            mandatory_keys = _MANDATORY_KEYS or []
        except Exception:
            mandatory_keys = []
    
    return BenchmarkComparatorAdapter(benchmark_file_path, mandatory_keys) 