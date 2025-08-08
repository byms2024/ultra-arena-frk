"""
Configuration for text-first processing strategy.
Imports base configuration from config.py to minimize code duplication.
"""

# Import base configuration
from .config_base import (
    # Provider Configuration
    PROVIDER,
    
    # Mandatory Keys Configuration
    MANDATORY_KEYS,
    NUM_RETRY_FOR_MANDATORY_KEYS,
    
    # File Processing Configuration
    MAX_NUM_FILE_PARTS_PER_REQUEST,
    MAX_NUM_FILE_PARTS_PER_BATCH,
    
    # API Keys and Model Settings
    GCP_API_KEY,
    GOOGLE_MODEL_ID,
    GOOGLE_MODEL_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_IMAGE_CHAT_MODEL,
    OPENAI_MODEL_TEMPERATURE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_MODEL_TEMPERATURE,
    CLAUDE_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MODEL_TEMPERATURE,
    
    # Ollama Configuration
    LOCAL_OLLAMA_MODEL,
    LOCAL_OLLAMA_TEMPERATURE,
    LOCAL_OLLAMA_MAX_TOKENS,
    LOCAL_OLLAMA_TIMEOUT,
    
    # File Limits
    MAX_TOKENS_PER_REQUEST,
    MAX_TEXT_LENGTH,
    MAX_TEXT_CHUNK_SIZE,
    
    # Retry Configuration
    API_INFRA_MAX_RETRIES,
    API_INFRA_RETRY_DELAY_SECONDS,
    API_INFRA_BACKOFF_MULTIPLIER,
    
    # Hybrid approach retry limits
    TEXT_FIRST_MAX_RETRY,
    FILE_DIRECT_MAX_RETRY,
    
    # Output directories
    OUTPUT_TXT_DIRECTORY,
    OUTPUT_IMAGE_DIRECTORY
)

# LLM Provider Configuration
LOCAL_LLM_PROVIDER = "ollama"  # "ollama", "google", "openai", "deepseek"

# PDF Text Extraction Configuration
PDF_EXTRACTOR_LIB = "pymupdf"  # "pymupdf" or "pytesseract"

# Provider-specific settings for text processing
TEXT_PROVIDER_CONFIGS = {
    "ollama": {
        "model": LOCAL_OLLAMA_MODEL,
        "temperature": LOCAL_OLLAMA_TEMPERATURE,
        "max_tokens": LOCAL_OLLAMA_MAX_TOKENS,
        "timeout": LOCAL_OLLAMA_TIMEOUT
    },
    "google": {
        "api_key": GCP_API_KEY,
        "model": GOOGLE_MODEL_ID,
        "temperature": GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_IMAGE_CHAT_MODEL,
        "temperature": OPENAI_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "model": DEEPSEEK_MODEL,
        "temperature": DEEPSEEK_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
           "claude": {
           "api_key": CLAUDE_API_KEY,
           "model": CLAUDE_MODEL,
           "temperature": CLAUDE_MODEL_TEMPERATURE,
           "max_tokens": 10000,
           "timeout": 60
       }
} 