"""
Base configuration settings for the PDF processing system.

This file serves as the base config that other config files import from,
minimizing code duplication. It also contains the only SYSTEM_PROMPT and USER_PROMPT
for use as examples or tests with the top-level run_batch_processing() function.
"""

# =============================================================================
# BASE CONFIGURATION CONSTANTS (imported by other config files)
# =============================================================================

# Provider Configuration
PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_CLAUDE = "claude"
PROVIDER_OLLAMA = "ollama"
PROVIDER_HUGGINGFACE = "huggingface"

PROVIDER = PROVIDER_OPENAI  # "google" or "openai"

# Mandatory keys are now provided by profiles (see PROFILE_DIR/profile_config.py)
# NUM_RETRY_FOR_MANDATORY_KEYS remains here unless overridden by profile
NUM_RETRY_FOR_MANDATORY_KEYS = 2

# File Processing Configuration
MAX_NUM_FILE_PARTS_PER_REQUEST = 4  # Reduced from 6 to 4 to prevent Google GenAI truncation
MAX_NUM_FILE_PARTS_PER_BATCH = 100    # For Batch/Slow/High Throughput Mode

# Google GenAI Configuration
# IMPORTANT: Use gemini-2.5-flash for production and testing to avoid 400 INVALID_ARGUMENT errors
# Experimental models like gemini-2.0-flash-exp may not support all API parameters and can cause errors
# NOTE: For large file batches (>6 files), the client will automatically use streaming to avoid truncation
GCP_API_KEY = ""
GOOGLE_MODEL_ID_GEMINI_25_FLASH = "gemini-2.5-flash"
GOOGLE_MODEL_ID = GOOGLE_MODEL_ID_GEMINI_25_FLASH
GOOGLE_MODEL_TEMPERATURE = 0.1
GOOGLE_MODEL_MAX_TOKENS = 1000000  # Increased from 32000 to 1000000 to handle large responses

# OpenAI Configuration
# https://chatgpt.com/share/688aced8-1b0c-800e-a96a-b201969f21b4 
# - open API is almost impossible to upload pdfs, and more expensive to read images
# - after a lot of testing,cursor concluded: "So the image_first strategy is the only way to use GPT-4.1 with PDF documents - we convert them to images first, then let GPT-4.1 process the images. This is why our recent test was successful!"
# Performance Improvements with GPT-4.1 (over GPT-4o-mini) with 2 pdfs that were missing CLAIM_NUMBER:
# Token Usage: Much more efficient (~2,450 tokens vs ~38,000 tokens with GPT-4o-mini)
# Processing Time: Faster (29.33s vs previous longer times)
# Data Quality: Better extraction - found chassis numbers and second CNPJ that were missing before
# Success Rate: 100% for API calls, much better field extraction
OPENAI_API_KEY = ""
OPENAI_MODEL_GPT_41 = "gpt-4.1"
# OPENAI_MODEL_GPT_41 = "gpt-4o-mini"
# https://openai.com/api/pricing/
# OPENAI_MODEL = "gpt-4.1-nano"  #https://platform.openai.com/docs/models/gpt-4.1-nano -- not working 
OPENAI_IMAGE_CHAT_MODEL = "gpt-4o-mini"  #https://platform.openai.com/docs/models/gpt-4o-mini -- working with image first strategy
# OPENAI_IMAGE_CHAT_MODEL = OPENAI_MODEL_GPT_41 #https://platform.openai.com/docs/models/gpt-4.1 -- working with image first strategy
OPENAI_MODEL_TEMPERATURE = 0.1

# DeepSeek Configuration
DEEPSEEK_API_KEY = ""
DEEPSEEK_MODEL_DCHAT = "deepseek-chat"
DEEPSEEK_MODEL = DEEPSEEK_MODEL_DCHAT
DEEPSEEK_MODEL_TEMPERATURE = 0.1

# Claude Configuration
CLAUDE_API_KEY = ""
CLAUDE_MODEL_CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
CLAUDE_MODEL = CLAUDE_MODEL_CLAUDE_4_SONNET
CLAUDE_MODEL_TEMPERATURE = 1.0
CLAUDE_MODEL_MAX_TOKENS = 10000

# HuggingFace Configuration
HUGGINGFACE_TOKEN = "" # read token
# huggingface biggest and most popular vision models:
# huggingface host open source smaller models, but they are not as good as the models hosted by openai and google
HUGGINGFACE_MODEL_ID_QWEN2_VL_72B = "Qwen/Qwen2.5-VL-72B-Instruct"
HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B = "meta-llama/Llama-3.2-90B-Vision-Instruct"
# HUGGINGFACE_MODEL_INTERNLVL2_76B = "internvl2-76b" # https://huggingface.co/OpenGVLab/InternVL2_5-78B not yet integrated in hugging face api apparently
HUGGINGFACE_MODEL_ID = HUGGINGFACE_MODEL_ID_QWEN2_VL_72B
HUGGINGFACE_MODEL_TEMPERATURE = 1.0



# Ollama Configuration (for local LLMs)
LOCAL_OLLAMA_MODEL = "deepseek-r1:8b"
LOCAL_OLLAMA_TEMPERATURE = 0.1
LOCAL_OLLAMA_MAX_TOKENS = 4000
LOCAL_OLLAMA_TIMEOUT = 120

# Processing Modes
MODE_BATCH_OPTIMIZED = "batch_optimized"
MODE_BATCH_PARALLEL = "parallel"

# File Limits
MAX_TOKENS_PER_REQUEST = 50000
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB for direct upload

# Retry Configuration 
# - infrastructure: Retries for API/network failures (timeouts, connection errors, etc.)
API_INFRA_MAX_RETRIES = 3
API_INFRA_RETRY_DELAY_SECONDS = 1
API_INFRA_BACKOFF_MULTIPLIER = 2

# Hybrid approach retry limits
TEXT_FIRST_MAX_RETRY = 3  # Maximum retries for text-first processing in hybrid mode
FILE_DIRECT_MAX_RETRY = 2  # Maximum retries for direct file processing in hybrid mode

# Text processing limits
MAX_TEXT_LENGTH = 10000  # Maximum text length for processing
MAX_TEXT_CHUNK_SIZE = 10000  # Maximum text chunk size for processing

# Output directories
OUTPUT_TXT_DIRECTORY = "processed_txt_output"
OUTPUT_IMAGE_DIRECTORY = "processed_image_output"

# =============================================================================
# OUTPUT DIRECTORY STRUCTURE CONFIGURATION
# =============================================================================

# Base output directory structure
OUTPUT_BASE_DIR = "output"
OUTPUT_RESULTS_DIR = f"{OUTPUT_BASE_DIR}/results"

# Combo processing output structure
OUTPUT_COMBO_DIR = f"{OUTPUT_RESULTS_DIR}/combo"
OUTPUT_COMBO_BACKUP_DIR = f"{OUTPUT_COMBO_DIR}/backup"
OUTPUT_COMBO_CSV_DIR = "csv"
OUTPUT_COMBO_JSON_DIR = "json"

# Non-combo processing output structure  
OUTPUT_NON_COMBO_DIR = f"{OUTPUT_RESULTS_DIR}/non-combo"
OUTPUT_NON_COMBO_CSV_DIR = f"{OUTPUT_NON_COMBO_DIR}/csv"
OUTPUT_NON_COMBO_JSON_DIR = f"{OUTPUT_NON_COMBO_DIR}/json"

# Other output directories
OUTPUT_CHECKPOINTS_DIR = f"{OUTPUT_BASE_DIR}/checkpoints"
OUTPUT_LOGS_DIR = f"{OUTPUT_BASE_DIR}/logs"
OUTPUT_NOTE_GEN_DIR = f"{OUTPUT_BASE_DIR}/note_gen"

# =============================================================================
# STRATEGY AND MODE CONSTANTS
# =============================================================================

STRATEGY_DIRECT_FILE = "direct_file"
STRATEGY_TEXT_FIRST = "text_first"
STRATEGY_IMAGE_FIRST = "image_first"
STRATEGY_HYBRID = "hybrid"

MODE_PARALLEL = "parallel"
MODE_BATCH = "batch"

# =============================================================================
# DEFAULT VALUES FOR run_file_processing() WRAPPER
# =============================================================================

# Default processing settings
DEFAULT_STRATEGY_TYPE = STRATEGY_DIRECT_FILE
DEFAULT_MODE = MODE_PARALLEL
DEFAULT_MAX_WORKERS = 5
DEFAULT_OUTPUT_FILE = "modular_results.json"
DEFAULT_CHECKPOINT_FILE = "modular_checkpoint.pkl"
DEFAULT_LLM_PROVIDER = "google"  # Default provider for all strategies

PROFILE_ROOT_DIR = "run_profiles"
DEFAULT_PROFILE_DIR = f"{PROFILE_ROOT_DIR}/default_profile"
PROFILE_DIR = DEFAULT_PROFILE_DIR

"""
Profile-driven configuration
- PROFILE_DIR determines which profile to load.
- Values from the profile (e.g., OUTPUT_BASE_DIR, prompts, benchmark path) override defaults here.
"""

from pathlib import Path
import importlib.util

# Defaults (may be overridden by profile)
BENCHMARK_FILE_PATH = ""
SYSTEM_PROMPT = ""
JSON_FORMAT_INSTRUCTIONS = ""
USER_PROMPT = ""
PROFILE_INPUT_DIR = ""
INPUT_DIR_FOR_COMBO = ""

def _load_profile_overrides() -> None:
    try:
        root_dir = Path(__file__).resolve().parent.parent  # Ultra_Arena_Main/
        profile_cfg = root_dir / PROFILE_DIR / "profile_config.py"
        if not profile_cfg.exists():
            return
        spec = importlib.util.spec_from_file_location("run_profile_config", str(profile_cfg))
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Optional: input dir for convenience in runners
        if hasattr(module, "INPUT_DIR"):
            globals()["PROFILE_INPUT_DIR"] = getattr(module, "INPUT_DIR")

        # Output base dir overrides and recompute structure
        if hasattr(module, "OUTPUT_BASE_DIR"):
            globals()["OUTPUT_BASE_DIR"] = getattr(module, "OUTPUT_BASE_DIR")
            globals()["OUTPUT_RESULTS_DIR"] = f"{OUTPUT_BASE_DIR}/results"
            globals()["OUTPUT_COMBO_DIR"] = f"{OUTPUT_RESULTS_DIR}/combo"
            globals()["OUTPUT_COMBO_BACKUP_DIR"] = f"{OUTPUT_COMBO_DIR}/backup"
            globals()["OUTPUT_COMBO_CSV_DIR"] = "csv"
            globals()["OUTPUT_COMBO_JSON_DIR"] = "json"
            globals()["OUTPUT_NON_COMBO_DIR"] = f"{OUTPUT_RESULTS_DIR}/non-combo"
            globals()["OUTPUT_NON_COMBO_CSV_DIR"] = f"{OUTPUT_NON_COMBO_DIR}/csv"
            globals()["OUTPUT_NON_COMBO_JSON_DIR"] = f"{OUTPUT_NON_COMBO_DIR}/json"
            globals()["OUTPUT_CHECKPOINTS_DIR"] = f"{OUTPUT_BASE_DIR}/checkpoints"
            globals()["OUTPUT_LOGS_DIR"] = f"{OUTPUT_BASE_DIR}/logs"
            globals()["OUTPUT_NOTE_GEN_DIR"] = f"{OUTPUT_BASE_DIR}/note_gen"

        # Provider default
        if hasattr(module, "DEFAULT_LLM_PROVIDER"):
            globals()["DEFAULT_LLM_PROVIDER"] = getattr(module, "DEFAULT_LLM_PROVIDER")

        # Mandatory keys (profile)
        if hasattr(module, "MANDATORY_KEYS"):
            globals()["MANDATORY_KEYS"] = getattr(module, "MANDATORY_KEYS")

        # Benchmark and prompts
        if hasattr(module, "BENCHMARK_FILE_PATH"):
            globals()["BENCHMARK_FILE_PATH"] = getattr(module, "BENCHMARK_FILE_PATH")
        if hasattr(module, "SYSTEM_PROMPT"):
            globals()["SYSTEM_PROMPT"] = getattr(module, "SYSTEM_PROMPT")
        if hasattr(module, "JSON_FORMAT_INSTRUCTIONS"):
            globals()["JSON_FORMAT_INSTRUCTIONS"] = getattr(module, "JSON_FORMAT_INSTRUCTIONS")
        if hasattr(module, "USER_PROMPT"):
            globals()["USER_PROMPT"] = getattr(module, "USER_PROMPT")

        # Combo input dir override
        if hasattr(module, "INPUT_DIR_FOR_COMBO"):
            globals()["INPUT_DIR_FOR_COMBO"] = getattr(module, "INPUT_DIR_FOR_COMBO")
    except Exception:
        # Fail silent; caller can proceed with defaults
        pass

_load_profile_overrides()