"""
Base performance monitoring functionality shared by all processing modes.
"""

import time
import threading
import logging
from datetime import datetime
from typing import Dict, Any


class BasePerformanceMonitor:
    """Base class for performance monitoring with common functionality."""
    
    def __init__(self, mode: str):
        self.start_time = time.time()
        self.mode = mode
        self.metrics = {
            'files_processed': 0,
            'files_successful': 0,
            'files_failed': 0,
            'files_retried': 0,
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'upload_times': [],
            'processing_times': [],
            'api_calls': 0,
            'errors': [],
            'file_sizes': [],
            'current_workers': 0,
            'peak_workers': 0,
            'files_deleted': 0,
            'storage_saved_mb': 0.0
        }
        self.lock = threading.Lock()
        self.progress_callback = None
    
    def update(self, **kwargs):
        """Thread-safe metric update."""
        with self.lock:
            for key, value in kwargs.items():
                if key in self.metrics:
                    if isinstance(self.metrics[key], list):
                        self.metrics[key].append(value)
                    else:
                        self.metrics[key] += value
                        
                        # Track peak workers
                        if key == 'current_workers':
                            self.metrics['peak_workers'] = max(self.metrics['peak_workers'], value)
                        
                        # Update total tokens when input/output tokens are updated
                        if key in ['input_tokens', 'output_tokens']:
                            self.metrics['total_tokens'] = self.metrics['input_tokens'] + self.metrics['output_tokens']
    
    def get_stats(self):
        """Get current statistics."""
        with self.lock:
            elapsed = time.time() - self.start_time
            avg_upload = sum(self.metrics['upload_times']) / len(self.metrics['upload_times']) if self.metrics['upload_times'] else 0
            avg_processing = sum(self.metrics['processing_times']) / len(self.metrics['processing_times']) if self.metrics['processing_times'] else 0
            avg_file_size = sum(self.metrics['file_sizes']) / len(self.metrics['file_sizes']) if self.metrics['file_sizes'] else 0
            
            return {
                'mode': self.mode,
                'files_processed': self.metrics['files_processed'],
                'files_successful': self.metrics['files_successful'],
                'files_failed': self.metrics['files_failed'],
                'files_retried': self.metrics['files_retried'],
                'elapsed_time': elapsed,
                'files_per_second': self.metrics['files_processed'] / elapsed if elapsed > 0 else 0,
                'avg_upload_time': avg_upload,
                'avg_processing_time': avg_processing,
                'avg_file_size_mb': avg_file_size,
                'total_tokens': self.metrics['total_tokens'],
                'input_tokens': self.metrics['input_tokens'],
                'output_tokens': self.metrics['output_tokens'],
                'error_rate': self.metrics['files_failed'] / max(self.metrics['files_processed'], 1),
                'success_rate': self.metrics['files_successful'] / max(self.metrics['files_processed'], 1),
                'api_calls': self.metrics['api_calls'],
                'peak_workers': self.metrics['peak_workers'],
                'total_file_size_mb': sum(self.metrics['file_sizes']),
                'files_deleted': self.metrics['files_deleted'],
                'storage_saved_mb': self.metrics['storage_saved_mb']
            }
    
    def log_progress(self, message: str, level: str = "INFO"):
        """Log progress with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        if level.upper() == "INFO":
            logging.info(log_message)
        elif level.upper() == "WARNING":
            logging.warning(log_message)
        elif level.upper() == "ERROR":
            logging.error(log_message)
        elif level.upper() == "DEBUG":
            logging.debug(log_message)
        
        # Call progress callback if set
        if self.progress_callback:
            self.progress_callback(log_message) 