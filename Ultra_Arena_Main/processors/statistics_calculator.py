"""
Statistics calculator for processing results and metrics.
"""

import logging
import time
from typing import Dict, List, Any


class StatisticsCalculator:
    """Calculates various statistics for processing results."""
    
    def __init__(self, structured_output: Dict[str, Any]):
        self.structured_output = structured_output
    
    def calculate_final_statistics(self, start_time: float):
        """Calculate final processing statistics."""
        total_time = time.time() - start_time
        
        # Calculate overall statistics
        total_files = sum(stats.get('total_files', 0) for stats in self.structured_output['group_stats'].values())
        successful_files = sum(stats.get('successful_files', 0) for stats in self.structured_output['group_stats'].values())
        failed_files = sum(stats.get('failed_files', 0) for stats in self.structured_output['group_stats'].values())
        
        # Calculate success rate
        success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
        
        # Calculate total tokens
        total_tokens = sum(stats.get('total_tokens', 0) for stats in self.structured_output['group_stats'].values())
        estimated_tokens = sum(stats.get('estimated_tokens', 0) for stats in self.structured_output['group_stats'].values())
        
        # Calculate total processing time
        total_processing_time = sum(stats.get('processing_time', 0) for stats in self.structured_output['group_stats'].values())
        
        self.structured_output['overall_stats'] = {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': round(success_rate, 2),
            'total_tokens': total_tokens,
            'estimated_tokens': estimated_tokens,
            'total_processing_time': total_processing_time,
            'total_time': round(total_time, 2)
        }
        
        logging.info(f"📊 Final statistics calculated: {successful_files}/{total_files} files successful ({success_rate:.1f}%)")
    
    def calculate_retry_statistics(self):
        """Calculate retry-related statistics."""
        retry_stats = self.structured_output['retry_stats']
        
        # Calculate retry percentages
        if retry_stats['num_files_may_need_retry'] > 0:
            retry_stats['percentage_files_had_retry'] = round(
                (retry_stats['num_files_had_retry'] / retry_stats['num_files_may_need_retry']) * 100, 2
            )
        else:
            retry_stats['percentage_files_had_retry'] = 0.0
        
        # Calculate retry success rate
        if retry_stats['num_files_had_retry'] > 0:
            retry_success_rate = round(
                ((retry_stats['num_files_had_retry'] - retry_stats['num_file_failed_after_max_retries']) / retry_stats['num_files_had_retry']) * 100, 2
            )
        else:
            retry_success_rate = 0.0
        
        retry_stats['retry_success_rate'] = retry_success_rate
        
        logging.info(f"🔄 Retry statistics: {retry_stats['num_files_had_retry']}/{retry_stats['num_files_may_need_retry']} files retried ({retry_stats['percentage_files_had_retry']:.1f}%)")
        logging.info(f"📈 Retry success rate: {retry_success_rate:.1f}%")
    
    def calculate_token_statistics(self):
        """Calculate token-related statistics."""
        # Calculate total tokens from all groups
        total_tokens = sum(stats.get('total_tokens', 0) for stats in self.structured_output['group_stats'].values())
        estimated_tokens = sum(stats.get('estimated_tokens', 0) for stats in self.structured_output['group_stats'].values())
        
        # Add retry tokens
        retry_tokens = self.structured_output['retry_stats'].get('actual_tokens_for_retries', 0)
        total_tokens += retry_tokens
        
        # Calculate token efficiency
        token_efficiency = 0
        if estimated_tokens > 0:
            token_efficiency = round((total_tokens / estimated_tokens) * 100, 2)
        
        # Update overall stats with token information
        if 'overall_stats' in self.structured_output:
            self.structured_output['overall_stats'].update({
                'total_tokens': total_tokens,
                'estimated_tokens': estimated_tokens,
                'retry_tokens': retry_tokens,
                'token_efficiency_percentage': token_efficiency
            })
        
        logging.info(f"🔢 Token statistics: {total_tokens} actual, {estimated_tokens} estimated ({token_efficiency:.1f}% efficiency)")
    
    def estimate_prompt_tokens(self) -> int:
        """Estimate prompt tokens for the entire processing."""
        # Simple estimation based on number of files and average prompt length
        total_files = sum(stats.get('total_files', 0) for stats in self.structured_output['group_stats'].values())
        estimated_prompt_tokens = total_files * 500  # Rough estimate
        return estimated_prompt_tokens
    
    def estimate_file_tokens(self) -> int:
        """Estimate file tokens for the entire processing."""
        # Simple estimation based on number of files and average file size
        total_files = sum(stats.get('total_files', 0) for stats in self.structured_output['group_stats'].values())
        estimated_file_tokens = total_files * 2000  # Rough estimate
        return estimated_file_tokens
    
    def print_summary(self):
        """Print a summary of processing results."""
        overall_stats = self.structured_output.get('overall_stats', {})
        retry_stats = self.structured_output.get('retry_stats', {})
        
        print("=" * 60)
        print("📊 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"📁 Total files processed: {overall_stats.get('total_files', 0)}")
        print(f"⏱️  Total processing time: {overall_stats.get('total_time', 0):.2f}s")
        print(f"🔄 Files that needed retry: {retry_stats.get('num_files_may_need_retry', 0)}")
        print(f"❌ Files failed after max retries: {retry_stats.get('num_file_failed_after_max_retries', 0)}")
        print(f"📈 Retry success rate: {retry_stats.get('retry_success_rate', 0):.1f}%")
        print("=" * 60) 