"""
Benchmark comparison module for test-match functionality.

This module handles loading benchmark data and comparing processed results
against benchmark values to identify mismatches.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from config.config_base import BENCHMARK_FILE_PATH, MANDATORY_KEYS


class BenchmarkComparator:
    """Handles benchmark comparison functionality."""
    
    def __init__(self, benchmark_file_path: str = BENCHMARK_FILE_PATH):
        """Initialize the benchmark comparator.
        
        Args:
            benchmark_file_path: Path to the benchmark Excel file
        """
        self.benchmark_file_path = benchmark_file_path
        self.df_benchmark = None
        self.df_unmatched = []
        self.total_unmatched_fields = 0
        self.total_unmatched_files = 0
        self.processed_files = set()
        
        # Load benchmark data
        self._load_benchmark_data()
    
    def _load_benchmark_data(self) -> None:
        """Load benchmark data from Excel file."""
        try:
            logging.info(f"📊 Loading benchmark data from: {self.benchmark_file_path}")
            self.df_benchmark = pd.read_excel(self.benchmark_file_path)
            logging.info(f"✅ Loaded benchmark data with {len(self.df_benchmark)} records")
            logging.info(f"📋 Benchmark columns: {list(self.df_benchmark.columns)}")
        except Exception as e:
            logging.error(f"❌ Failed to load benchmark data: {e}")
            self.df_benchmark = pd.DataFrame()
    
    def compare_file_result(self, file_path: str, processed_result: Dict) -> None:
        """Compare a processed file result against benchmark data.
        
        Args:
            file_path: Path to the processed file
            processed_result: Dictionary containing the processed result
        """
        if self.df_benchmark.empty:
            logging.warning("⚠️ No benchmark data available for comparison")
            return
        
        # Extract filename from file path for matching
        filename = Path(file_path).name
        
        # Find matching benchmark record
        benchmark_record = self._find_benchmark_record(filename)
        if benchmark_record is None:
            logging.warning(f"⚠️ No benchmark record found for file: {filename}")
            return
        
        # Track if this file has any mismatches
        file_has_mismatches = False
        
        # Compare each mandatory key
        for key in MANDATORY_KEYS:
            benchmark_value = benchmark_record.get(key)
            processed_value = processed_result.get(key)
            
            # Compare values (handle None/null cases)
            if not self._values_match(benchmark_value, processed_value):
                # Record mismatch
                mismatch_record = {
                    'file_path': file_path,
                    'filename': filename,
                    'field_name': key,
                    'benchmark_value': benchmark_value,
                    'processed_value': processed_value
                }
                self.df_unmatched.append(mismatch_record)
                
                # Increment counters
                self.total_unmatched_fields += 1
                file_has_mismatches = True
                
                logging.info(f"🔍 Mismatch found in {filename} - {key}: "
                           f"benchmark='{benchmark_value}' vs processed='{processed_value}'")
        
        # Increment file counter if any mismatches found
        if file_has_mismatches and file_path not in self.processed_files:
            self.total_unmatched_files += 1
            self.processed_files.add(file_path)
    
    def _find_benchmark_record(self, filename: str) -> Optional[Dict]:
        """Find benchmark record for a given filename.
        
        Args:
            filename: Name of the file to find
            
        Returns:
            Dictionary containing benchmark data or None if not found
        """
        # Try to find by filename column (common column names)
        possible_filename_columns = ['filename', 'file_name', 'file', 'path', 'file_path']
        
        for col in possible_filename_columns:
            if col in self.df_benchmark.columns:
                matches = self.df_benchmark[self.df_benchmark[col] == filename]
                if not matches.empty:
                    return matches.iloc[0].to_dict()
        
        # If no exact match found, try partial matching
        for col in possible_filename_columns:
            if col in self.df_benchmark.columns:
                matches = self.df_benchmark[self.df_benchmark[col].str.contains(filename, na=False)]
                if not matches.empty:
                    return matches.iloc[0].to_dict()
        
        return None
    
    def _values_match(self, value1, value2) -> bool:
        """Compare two values for equality, handling None/null cases.
        
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
    
    def get_benchmark_value(self, file_path: str, field_name: str) -> Optional[str]:
        """Get benchmark value for a specific file and field.
        
        Args:
            file_path: Path to the file
            field_name: Name of the field to get benchmark value for
            
        Returns:
            Benchmark value for the field or None if not found
        """
        if self.df_benchmark.empty:
            return None
        
        # Extract filename from file path for matching
        filename = Path(file_path).name
        
        # Find matching benchmark record
        benchmark_record = self._find_benchmark_record(filename)
        if benchmark_record is None:
            return None
        
        # Get the benchmark value for the field
        benchmark_value = benchmark_record.get(field_name)
        
        # Convert to string and handle None/null cases
        if pd.isna(benchmark_value):
            return None
        
        return str(benchmark_value).strip()
    
    def get_benchmark_errors(self) -> Dict:
        """Get benchmark error statistics.
        
        Returns:
            Dictionary containing error statistics
        """
        return {
            "total_unmatched_fields": self.total_unmatched_fields,
            "total_unmatched_files": self.total_unmatched_files
        }
    
    def get_unmatched_dataframe(self) -> pd.DataFrame:
        """Get unmatched data as a pandas DataFrame.
        
        Returns:
            DataFrame containing unmatched records
        """
        if not self.df_unmatched:
            return pd.DataFrame()
        
        return pd.DataFrame(self.df_unmatched)
    
    def save_unmatched_to_csv(self, csv_file_path: str) -> None:
        """Save unmatched data to CSV file.
        
        Args:
            csv_file_path: Path where to save the CSV file
        """
        if not self.df_unmatched:
            logging.info("📊 No unmatched records to save")
            return
        
        try:
            # Create errors directory if it doesn't exist
            errors_dir = Path(csv_file_path).parent / "errors"
            errors_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate errors CSV filename
            csv_filename = Path(csv_file_path).name
            errors_csv_path = errors_dir / f"errors_{csv_filename}"
            
            # Save to CSV
            df_unmatched = self.get_unmatched_dataframe()
            df_unmatched.to_csv(errors_csv_path, index=False)
            
            logging.info(f"📊 Saved {len(df_unmatched)} unmatched records to: {errors_csv_path}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save unmatched data to CSV: {e}") 