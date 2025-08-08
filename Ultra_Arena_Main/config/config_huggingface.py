"""
Configuration for HuggingFace provider with image-first processing strategy.
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
    
    # HuggingFace Configuration
    HUGGINGFACE_TOKEN,
    HUGGINGFACE_MODEL_ID_QWEN2_VL_72B,
    HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B,
    HUGGINGFACE_MODEL_ID,
    HUGGINGFACE_MODEL_TEMPERATURE,
    
    # File Limits
    MAX_TOKENS_PER_REQUEST,
    MAX_FILE_SIZE_MB,
    
    # Retry Configuration
    API_INFRA_MAX_RETRIES,
    API_INFRA_RETRY_DELAY_SECONDS,
    API_INFRA_BACKOFF_MULTIPLIER,
    
    # Output directories
    OUTPUT_TXT_DIRECTORY,
    OUTPUT_IMAGE_DIRECTORY
)

# PDF to Image Conversion Configuration
PDF_TO_IMAGE_DPI = 300  # DPI for image conversion
PDF_TO_IMAGE_FORMAT = "PNG"  # Output image format
PDF_TO_IMAGE_QUALITY = 95  # Image quality (for JPEG)

# HuggingFace Provider Configuration
HUGGINGFACE_PROVIDER_CONFIG = {
    "api_key": HUGGINGFACE_TOKEN,
    "model": HUGGINGFACE_MODEL_ID,
    "temperature": HUGGINGFACE_MODEL_TEMPERATURE,
    "max_tokens": 4000,
    "timeout": 60
}

# Model-specific configurations
HUGGINGFACE_MODEL_CONFIGS = {
    HUGGINGFACE_MODEL_ID_QWEN2_VL_72B: {
        "api_key": HUGGINGFACE_TOKEN,
        "model": HUGGINGFACE_MODEL_ID_QWEN2_VL_72B,
        "temperature": HUGGINGFACE_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    },
    HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B: {
        "api_key": HUGGINGFACE_TOKEN,
        "model": HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B,
        "temperature": HUGGINGFACE_MODEL_TEMPERATURE,
        "max_tokens": 4000,
        "timeout": 60
    }
} 