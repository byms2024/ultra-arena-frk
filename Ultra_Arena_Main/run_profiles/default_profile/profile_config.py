# Example private profile overrides

# Where your input PDFs live for this profile
INPUT_DIR = "../input_files"

# Redirect all outputs for this profile
OUTPUT_BASE_DIR = "../output"

# Optional overrides
DEFAULT_LLM_PROVIDER = "google"
MAX_WORKERS = 5

# =============================================================================
# BENCHMARK
# =============================================================================

BENCHMARK_FILE_PATH = "benchmark_file/xxx.xlsx"

# =============================================================================
# PROMPTS
# =============================================================================

# System Prompt
SYSTEM_PROMPT = """
"""

# JSON Formatting Instructions (concatenated to main prompt)
JSON_FORMAT_INSTRUCTIONS = """
"""

# User Prompt
USER_PROMPT = """
""" + JSON_FORMAT_INSTRUCTIONS 