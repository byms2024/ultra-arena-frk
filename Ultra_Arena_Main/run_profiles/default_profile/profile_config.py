# Example private profile overrides

# Where your input PDFs live for this profile
INPUT_DIR = "../input_files"

# Redirect all outputs for this profile
OUTPUT_BASE_DIR = "../output"

# Optional overrides
DEFAULT_LLM_PROVIDER = "google"
MAX_WORKERS = 5

# Mandatory Keys Configuration (profile-scoped)
MANDATORY_KEYS = ['']

# Combo input root for combo runs
INPUT_DIR_FOR_COMBO = "input_files"

# =============================================================================
# BENCHMARK
# =============================================================================

BENCHMARK_FILE_PATH = "benchmark_file/xxx.xlsx"

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

# Mandatory Keys
MANDATORY_KEYS = ['']