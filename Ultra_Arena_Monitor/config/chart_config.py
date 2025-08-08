
CHART_TITLE = "chart_title"
DECIMAL = "decimal_point"

# Configuration for chart layout
CHARTS_PER_ROW = 4  # Number of charts to display per row on desktop

# Import monitoring configuration
try:
    from .monitoring_config import (
        REAL_TIME_MONITORING, 
        UPDATE_FREQUENCY_SECONDS, 
        FILE_WATCH_ENABLED
    )
except ImportError:
    # Fallback values if monitoring config is not available
    REAL_TIME_MONITORING = True
    UPDATE_FREQUENCY_SECONDS = 1
    FILE_WATCH_ENABLED = True

# Configuration for data sources
JSON_DATA_DIR = "../My_Ultra_Parallel/output/results/combo/.../json"  # Directory containing JSON data files

chart_config_all = {
    "chart_config_1": {
        "comparing_fields" : {
            "retry_stats": {
                "percentage_files_had_retry" : {CHART_TITLE : "Percent of Files Retried", DECIMAL: 2}
            },
            "overall_stats": {
                # "total_files": {CHART_TITLE : "Total Files"},
                "total_wall_time_in_sec": {CHART_TITLE : "Total Processing Time In Seconds"},
                "total_actual_tokens": {CHART_TITLE : "All Tokens Spent"}
            },
            "overall_cost": {
                # "total_prompt_token_cost":  {CHART_TITLE : "Total Cost for Prompt Tokens ($USD)"},
                # "total_candidate_token_cost":  {CHART_TITLE : "Total Cost for Candidate Tokens ($USD)"},
                # "total_other_token_cost":  {CHART_TITLE : "Total Cost for Other Token ($USD)"},
                "total_token_cost":  {CHART_TITLE : "Total Cost for All Tokens ($USD)", DECIMAL: 6}
            },
            "benchmark_comparison": {
                # "total_unmatched_fields": {CHART_TITLE : "Total Incorrect Data Fields Count"},
                # "total_unmatched_files": {CHART_TITLE : "Total Incorrect Extract File Count"},
                "invalid_fields_percent": {CHART_TITLE : "Percentage of Incorrectly Extracted Fields", DECIMAL: 2},
                "invalid_files_percent": {CHART_TITLE : "Percentage of Incorrectly Processed Files", DECIMAL: 2}
            }
        }
    }

}