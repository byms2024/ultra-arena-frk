"""
Benchmark Reporter - Report generation and CSV export functionality.

This class handles the generation of benchmark comparison reports,
error CSV files, and statistical summaries.
"""

import logging
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class BenchmarkReporter:
    """
    Reporter class for benchmark comparison results.
    
    This class handles:
    - Generating error CSV files
    - Creating comprehensive benchmark reports
    - Statistical summaries and visualizations
    """
    
    def __init__(self):
        """Initialize the benchmark reporter."""
        pass
    
    def generate_error_csv(self, unmatched_data: List[Dict[str, Any]], output_path: str) -> str:
        """
        Generate CSV file with detailed error information.
        
        Args:
            unmatched_data: List of unmatched field data
            output_path: Base path for the CSV file
            
        Returns:
            Path to the generated CSV file
        """
        if not unmatched_data:
            return None
        
        # Create errors directory
        output_path = Path(output_path)
        errors_dir = output_path.parent / 'errors'
        errors_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate error CSV filename
        csv_filename = output_path.name
        error_csv_filename = f"errors_{csv_filename}"
        error_csv_path = errors_dir / error_csv_filename
        
        # Create DataFrame and save to CSV
        df_errors = pd.DataFrame(unmatched_data)
        df_errors.to_csv(error_csv_path, index=False)
        
        logging.info(f"💾 Error CSV file saved: {error_csv_path}")
        return str(error_csv_path)
    
    def generate_comprehensive_report(self, benchmark_errors: Dict[str, Any], 
                                   unmatched_data: List[Dict[str, Any]], 
                                   output_dir: str) -> Dict[str, str]:
        """
        Generate comprehensive benchmark comparison report.
        
        Args:
            benchmark_errors: Dictionary containing error statistics
            unmatched_data: List of unmatched field data
            output_dir: Directory to save report files
            
        Returns:
            Dictionary containing paths to generated report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        # Generate summary report
        summary_report = self._generate_summary_report(benchmark_errors, unmatched_data)
        summary_path = output_path / f"benchmark_summary_{timestamp}.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        report_files['summary'] = str(summary_path)
        
        # Generate detailed error report
        if unmatched_data:
            error_report = self._generate_detailed_error_report(unmatched_data)
            error_path = output_path / f"benchmark_errors_{timestamp}.json"
            
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=2, ensure_ascii=False)
            
            report_files['errors'] = str(error_path)
        
        # Generate CSV report
        if unmatched_data:
            csv_path = output_path / f"benchmark_errors_{timestamp}.csv"
            df_errors = pd.DataFrame(unmatched_data)
            df_errors.to_csv(csv_path, index=False)
            report_files['csv'] = str(csv_path)
        
        return report_files
    
    def _generate_summary_report(self, benchmark_errors: Dict[str, Any], 
                               unmatched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary report with key statistics.
        
        Args:
            benchmark_errors: Dictionary containing error statistics
            unmatched_data: List of unmatched field data
            
        Returns:
            Dictionary containing summary statistics
        """
        # Calculate additional statistics
        unique_files = set()
        field_error_counts = {}
        
        for error in unmatched_data:
            unique_files.add(error['file_path'])
            field_name = error['field_name']
            field_error_counts[field_name] = field_error_counts.get(field_name, 0) + 1
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_unmatched_fields': benchmark_errors.get('total_unmatched_fields', 0),
            'total_unmatched_files': benchmark_errors.get('total_unmatched_files', 0),
            'unique_files_with_errors': len(unique_files),
            'field_error_breakdown': field_error_counts,
            'error_rate_percentage': self._calculate_error_rate(benchmark_errors, unmatched_data)
        }
        
        return summary
    
    def _generate_detailed_error_report(self, unmatched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate detailed error report with file-level breakdown.
        
        Args:
            unmatched_data: List of unmatched field data
            
        Returns:
            Dictionary containing detailed error information
        """
        file_errors = {}
        
        for error in unmatched_data:
            file_path = error['file_path']
            if file_path not in file_errors:
                file_errors[file_path] = []
            
            file_errors[file_path].append({
                'field_name': error['field_name'],
                'benchmark_value': error['benchmark_value'],
                'extracted_value': error['extracted_value'],
                'timestamp': error.get('timestamp', '')
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_errors': len(unmatched_data),
            'files_with_errors': len(file_errors),
            'file_error_details': file_errors
        }
    
    def _calculate_error_rate(self, benchmark_errors: Dict[str, Any], 
                            unmatched_data: List[Dict[str, Any]]) -> float:
        """
        Calculate error rate percentage.
        
        Args:
            benchmark_errors: Dictionary containing error statistics
            unmatched_data: List of unmatched field data
            
        Returns:
            Error rate as a percentage
        """
        total_fields = benchmark_errors.get('total_unmatched_fields', 0)
        if total_fields == 0:
            return 0.0
        
        # Estimate total fields (assuming 5 mandatory keys per file)
        estimated_total_fields = len(set(error['file_path'] for error in unmatched_data)) * 5
        
        if estimated_total_fields == 0:
            return 0.0
        
        return (total_fields / estimated_total_fields) * 100
    
    def generate_benchmark_statistics(self, file_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate benchmark statistics from file results.
        
        Args:
            file_results: Dictionary containing file processing results
            
        Returns:
            Dictionary containing benchmark statistics
        """
        stats = {
            'total_files': len(file_results),
            'files_with_errors': 0,
            'total_errors': 0,
            'error_rate': 0.0,
            'field_error_breakdown': {},
            'file_error_breakdown': {}
        }
        
        for file_path, result in file_results.items():
            if result.get('has_errors', False):
                stats['files_with_errors'] += 1
                stats['total_errors'] += result.get('unmatched_count', 0)
                
                # Track field errors
                for error in result.get('field_errors', []):
                    field_name = error['field_name']
                    stats['field_error_breakdown'][field_name] = stats['field_error_breakdown'].get(field_name, 0) + 1
                
                # Track file errors
                stats['file_error_breakdown'][file_path] = result.get('unmatched_count', 0)
        
        # Calculate error rate
        if stats['total_files'] > 0:
            stats['error_rate'] = (stats['files_with_errors'] / stats['total_files']) * 100
        
        return stats 