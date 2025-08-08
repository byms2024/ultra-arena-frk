"""
Configuration for direct file processing strategy.
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
    
    # File Limits
    MAX_TOKENS_PER_REQUEST,
    MAX_FILE_SIZE_MB,
    
    # Retry Configuration
    API_INFRA_MAX_RETRIES,
    API_INFRA_RETRY_DELAY_SECONDS,
    API_INFRA_BACKOFF_MULTIPLIER,
    
    # Hybrid approach retry limits
    TEXT_FIRST_MAX_RETRY,
    FILE_DIRECT_MAX_RETRY
)

# Retry Configuration
MAX_RETRIES = 2  # Maximum retry rounds for failed files
RETRY_DELAY_SECONDS = 1  # Delay between retry rounds

# Provider-specific settings
# NOTE: For Google GenAI, use gemini-2.5-flash to avoid 400 INVALID_ARGUMENT errors
# Experimental models may not support all API parameters like response_mime_type
PROVIDER_CONFIGS = {
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