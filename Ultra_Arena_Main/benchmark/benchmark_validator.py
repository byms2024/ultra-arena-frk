"""
Benchmark Validator - Core validation logic for benchmark comparison.

This class handles the validation of extracted results against benchmark data,
including mandatory key checking and value comparison.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class BenchmarkValidator:
    """
    Core validator class for benchmark comparison operations.
    
    This class handles:
    - Loading and managing benchmark data
    - Validating extracted results against benchmarks
    - Mandatory key validation with benchmark comparison
    """
    
    def __init__(self, benchmark_file_path: str, mandatory_keys: List[str]):
        """
        Initialize the benchmark validator.
        
        Args:
            benchmark_file_path: Path to the benchmark Excel file
            mandatory_keys: List of mandatory keys to validate
        """
        self.benchmark_file_path = benchmark_file_path
        self.mandatory_keys = mandatory_keys
        self.benchmark_data = None
        
        # Load benchmark data
        self._load_benchmark_data()
        
        logging.info(f"🔍 Benchmark Validator initialized with {len(mandatory_keys)} mandatory keys")
    
    def _load_benchmark_data(self):
        """Load benchmark data from Excel file."""
        try:
            self.benchmark_data = pd.read_excel(self.benchmark_file_path)
            logging.info(f"📊 Loaded benchmark data: {len(self.benchmark_data)} records")
        except Exception as e:
            logging.error(f"❌ Failed to load benchmark data: {e}")
            self.benchmark_data = pd.DataFrame()
    
    def _find_benchmark_record(self, file_path: str) -> Optional[pd.Series]:
        """
        Find benchmark record for a given file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Benchmark record as pandas Series or None if not found
        """
        if self.benchmark_data.empty:
            return None
        
        # Extract filename from path
        filename = Path(file_path).name
        
        # Try to find exact match
        match = self.benchmark_data[self.benchmark_data['filename'] == filename]
        
        if not match.empty:
            return match.iloc[0]
        
        # Try partial match if exact match fails
        for _, record in self.benchmark_data.iterrows():
            if filename in str(record.get('filename', '')):
                return record
        
        return None
    
    def get_benchmark_value(self, file_path: str, field_name: str) -> Optional[str]:
        """
        Get benchmark value for a specific file and field.
        
        Args:
            file_path: Path to the file
            field_name: Name of the field
            
        Returns:
            Benchmark value or None if not found
        """
        record = self._find_benchmark_record(file_path)
        if record is not None and field_name in record:
            value = record[field_name]
            return str(value) if pd.notna(value) else None
        return None
    
    def _values_match(self, value1: Any, value2: Any) -> bool:
        """
        Compare two values for equality, handling None/null cases.
        
        Args:
            value1: First value to compare
            value2: Second value to compare
            
        Returns:
            True if values match, False otherwise
        """
        # Handle None/null cases
        if pd.isna(value1) and pd.isna(value2):
            return True
        if pd.isna(value1) or pd.isna(value2):
            return False
        
        # Convert to strings for comparison (handles different data types)
        str1 = str(value1).strip()
        str2 = str(value2).strip()
        
        return str1 == str2
    
    def validate_single_file(self, file_path: str, extracted_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted results for a single file against benchmark data.
        
        Args:
            file_path: Path to the processed file
            extracted_result: Extracted data from the file
            
        Returns:
            Dictionary containing validation results
        """
        if not extracted_result or not isinstance(extracted_result, dict):
            return {
                'has_errors': True,
                'unmatched_count': len(self.mandatory_keys),
                'field_errors': [],
                'missing_keys': self.mandatory_keys,
                'present_keys': [],
                'matching_keys': [],
                'non_matching_keys': self.mandatory_keys
            }
        
        # Skip validation for 'Outros' documents
        if extracted_result.get('DOC_TYPE') == 'Outros':
            return {
                'has_errors': False,
                'unmatched_count': 0,
                'field_errors': [],
                'missing_keys': [],
                'present_keys': [],
                'matching_keys': [],
                'non_matching_keys': []
            }
        
        # Find benchmark record
        benchmark_record = self._find_benchmark_record(file_path)
        
        missing_keys = []
        present_keys = []
        matching_keys = []
        non_matching_keys = []
        field_errors = []
        
        for key in self.mandatory_keys:
            extracted_value = extracted_result.get(key)
            
            # Check if key is present
            if extracted_value is None or extracted_value == "" or extracted_value == "Not found":
                missing_keys.append(key)
                continue
            
            present_keys.append(key)
            
            # Compare with benchmark if available
            if benchmark_record is not None and key in benchmark_record:
                benchmark_value = benchmark_record[key]
                benchmark_str = str(benchmark_value) if pd.notna(benchmark_value) else None
                
                if self._values_match(extracted_value, benchmark_str):
                    matching_keys.append(key)
                else:
                    non_matching_keys.append(key)
                    field_errors.append({
                        'field_name': key,
                        'benchmark_value': benchmark_str,
                        'extracted_value': str(extracted_value)
                    })
            else:
                # No benchmark data available, consider as non-matching
                non_matching_keys.append(key)
                field_errors.append({
                    'field_name': key,
                    'benchmark_value': None,
                    'extracted_value': str(extracted_value)
                })
        
        # Log validation results
        if matching_keys:
            logging.info(f"🎯 Some present key values match benchmark: {matching_keys}")
        
        if non_matching_keys:
            logging.warning(f"⚠️ Some present key values don't match benchmark: {non_matching_keys}")
            
            # Log specific mismatches
            for error in field_errors:
                logging.info(f"🔍 Value mismatch for {error['field_name']}: benchmark='{error['benchmark_value']}' vs extracted='{error['extracted_value']}'")
        
        if missing_keys:
            logging.warning(f"⚠️ Missing mandatory keys: {missing_keys}. Present keys: {present_keys}")
        
        return {
            'has_errors': len(missing_keys) > 0 or len(non_matching_keys) > 0,
            'unmatched_count': len(missing_keys) + len(non_matching_keys),
            'field_errors': field_errors,
            'missing_keys': missing_keys,
            'present_keys': present_keys,
            'matching_keys': matching_keys,
            'non_matching_keys': non_matching_keys
        }
    
    def check_mandatory_keys_with_benchmark(self, result: Dict[str, Any], file_path: str) -> Tuple[bool, List[str]]:
        """
        Check mandatory keys with benchmark comparison (legacy interface).
        
        Args:
            result: Extracted result dictionary
            file_path: Path to the file
            
        Returns:
            Tuple of (has_all_keys, missing_keys)
        """
        validation_result = self.validate_single_file(file_path, result)
        
        # For backward compatibility, return the same format as the original method
        missing_keys = validation_result['missing_keys'] + validation_result['non_matching_keys']
        has_all_keys = len(missing_keys) == 0
        
        return has_all_keys, missing_keys 