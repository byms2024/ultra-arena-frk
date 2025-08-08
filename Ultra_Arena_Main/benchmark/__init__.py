"""
Benchmark comparison module for PDF processing results.

This module provides comprehensive benchmark comparison functionality for
evaluating the accuracy of extracted data against reference benchmark files.
"""

from .benchmark_manager import BenchmarkManager
from .benchmark_validator import BenchmarkValidator
from .benchmark_reporter import BenchmarkReporter
from .benchmark_adapter import BenchmarkComparatorAdapter, create_benchmark_manager, create_benchmark_adapter

__all__ = [
    'BenchmarkManager',
    'BenchmarkValidator', 
    'BenchmarkReporter',
    'BenchmarkComparatorAdapter',
    'create_benchmark_manager',
    'create_benchmark_adapter'
] 