from config import config_base

param_grps = {
    "grp_directF_dSeek_dChat_para" : {
        "strategy": config_base.STRATEGY_DIRECT_FILE,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_DEEPSEEK,
        "model": config_base.DEEPSEEK_MODEL_DCHAT,
        "temperature": config_base.DEEPSEEK_MODEL_TEMPERATURE
    }, 
    "grp_imageF_dSeek_dChat_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_DEEPSEEK,
        "model": config_base.DEEPSEEK_MODEL_DCHAT,
        "temperature": config_base.DEEPSEEK_MODEL_TEMPERATURE
    },
    "grp_textF_dSeek_dChat_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_DEEPSEEK,
        "model": config_base.DEEPSEEK_MODEL_DCHAT,
        "temperature": config_base.DEEPSEEK_MODEL_TEMPERATURE
    },
    "grp_directF_google_gemini25_para" : {
        "strategy": config_base.STRATEGY_DIRECT_FILE,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_GOOGLE,
        "model": config_base.GOOGLE_MODEL_ID_GEMINI_25_FLASH,
        "temperature": config_base.GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": config_base.GOOGLE_MODEL_MAX_TOKENS
    },
    "grp_imageF_google_gemini25_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_GOOGLE,
        "model": config_base.GOOGLE_MODEL_ID_GEMINI_25_FLASH,
        "temperature": config_base.GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": config_base.GOOGLE_MODEL_MAX_TOKENS
    },
    "grp_textF_google_gemini25_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_GOOGLE,
        "model": config_base.GOOGLE_MODEL_ID_GEMINI_25_FLASH,
        "temperature": config_base.GOOGLE_MODEL_TEMPERATURE,
        "max_tokens": config_base.GOOGLE_MODEL_MAX_TOKENS
    },
    "grp_test_imageF_openai_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_OPENAI,
        "model": config_base.OPENAI_MODEL_GPT_41,
        "temperature": config_base.OPENAI_MODEL_TEMPERATURE
    },
    # "grp_test_textF_openai_para" : {
    #     "strategy": config_base.STRATEGY_TEXT_FIRST,
    #     "mode": config_base.MODE_BATCH_PARALLEL,
    #     "provider": config_base.PROVIDER_OPENAI,
    #     "model": config_base.OPENAI_MODEL_GPT_41,
    #     "temperature": config_base.OPENAI_MODEL_TEMPERATURE
    # },
    "grp_test_imageF_claude_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_CLAUDE,
        "model": config_base.CLAUDE_MODEL_CLAUDE_4_SONNET,
        "temperature": config_base.CLAUDE_MODEL_TEMPERATURE
    },
    "grp_test_textF_claude_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_CLAUDE,
        "model": config_base.CLAUDE_MODEL_CLAUDE_4_SONNET,
        "temperature": config_base.CLAUDE_MODEL_TEMPERATURE
    },
    "grp_textFirst_openai_gpt4_para" : {
        "strategy": config_base.STRATEGY_TEXT_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_OPENAI,
        "model": config_base.OPENAI_IMAGE_CHAT_MODEL,
        "temperature": config_base.OPENAI_MODEL_TEMPERATURE
    },
    "grp_imageF_huggingface_qwen_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_HUGGINGFACE,
        "model": config_base.HUGGINGFACE_MODEL_ID_QWEN2_VL_72B,
        "temperature": config_base.HUGGINGFACE_MODEL_TEMPERATURE
    },
    "grp_imageF_huggingface_llama_para" : {
        "strategy": config_base.STRATEGY_IMAGE_FIRST,
        "mode": config_base.MODE_BATCH_PARALLEL,
        "provider": config_base.PROVIDER_HUGGINGFACE,
        "model": config_base.HUGGINGFACE_MODEL_ID_LLAMA_VISION_90B,
        "temperature": config_base.HUGGINGFACE_MODEL_TEMPERATURE
    }
}