# Example private profile overrides

# Where your input PDFs live for this profile
INPUT_DIR = "./input_files"

# Combo input root for combo runs
INPUT_DIR_FOR_COMBO = INPUT_DIR

# Redirect all outputs for this profile
OUTPUT_BASE_DIR = "./output"

# Optional overrides
DEFAULT_LLM_PROVIDER = "google"
MAX_WORKERS = 5

# =============================================================================
# BENCHMARK
# =============================================================================

BENCHMARK_FILE_PATH = "benchmark_file/benchmark_1.xlsx"

# =============================================================================
# PROMPTS
# =============================================================================

# System Prompt
SYSTEM_PROMPT = """
extract key info and make sure the content is less than 100 characters
"""

# JSON Formatting Instructions (concatenated to main prompt)
JSON_FORMAT_INSTRUCTIONS = """
needs to be in json format
"""

# User Prompt
USER_PROMPT = """
""" + JSON_FORMAT_INSTRUCTIONS 

# Mandatory Keys Configuration (profile-scoped)
MANDATORY_KEYS = ['']
