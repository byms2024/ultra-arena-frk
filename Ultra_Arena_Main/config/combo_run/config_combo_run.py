from config import config_base
from config.run_param_grps_def import param_grps

DIR_MAIN_INPUT = "input_files"
# Single file test directory
DIR_1_FILES = f"{DIR_MAIN_INPUT}/1_file"
DIR_2_FILES = f"{DIR_MAIN_INPUT}/2_files"
DIR_4_FILES = f"{DIR_MAIN_INPUT}/4_files"
DIR_13_FILES = f"{DIR_MAIN_INPUT}/13_files"
DIR_30_FILES = f"{DIR_MAIN_INPUT}/30_files"
DIR_252_FILES = f"{DIR_MAIN_INPUT}/252_files"

combo_config = {
    "combo1" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_directF_dSeek_dChat_para",
            "grp_directF_google_gemini25_para"
        ]
    },
    "combo2" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_directF_dSeek_dChat_para",
            "grp_textFirst_openai_gpt4_para"
        ]
    },
    "combo_test_4_strategies" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_directF_dSeek_dChat_para",
            "grp_test_imageF_openai_para",
            "grp_test_imageF_claude_para"
        ]
    },
    "combo_test_8_strategies_1f" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            "grp_directF_dSeek_dChat_para",
            # "grp_imageF_dSeek_dChat_para", #    WARNING - Unsupported file type: temp_images/image_5eb95081.png
            "grp_textF_dSeek_dChat_para",
            "grp_test_imageF_openai_para",
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para"  
        ]
    },    
    "combo_test_8_strategies_4f" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            "grp_directF_dSeek_dChat_para",
            # "grp_imageF_dSeek_dChat_para", #    WARNING - Unsupported file type: temp_images/image_5eb95081.png
            "grp_textF_dSeek_dChat_para",
            "grp_test_imageF_openai_para",
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para"  
        ]
    },
    "combo_test_8_strategies_13f" : {
        "input_files" : DIR_13_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            "grp_directF_dSeek_dChat_para",
            # "grp_imageF_dSeek_dChat_para", #    WARNING - Unsupported file type: temp_images/image_5eb95081.png
            "grp_textF_dSeek_dChat_para",
            "grp_test_imageF_openai_para",
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para"  
        ]
    },
    "combo_test_8_strategies_252f" : {
        "input_files" : DIR_252_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para",
            "grp_textF_google_gemini25_para",
            "grp_directF_dSeek_dChat_para",
            # "grp_imageF_dSeek_dChat_para", #    WARNING - Unsupported file type: temp_images/image_5eb95081.png
            "grp_textF_dSeek_dChat_para",
            "grp_test_imageF_openai_para",
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para"  
        ]
    },    
    "combo_test_imageF_strategies" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_test_imageF_claude_para",
            "grp_imageF_google_gemini25_para",
            "grp_test_imageF_openai_para",
            # "grp_textF_dSeek_dChat_para", # no longer bad, after fixing ""list indices must be integers or slices, not str""
        ]
    },

    "combo_test_bad_strategies" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_test_imageF_claude_para",
            "grp_imageF_google_gemini25_para",
            # "grp_textF_dSeek_dChat_para", # no longer bad, after fixing ""list indices must be integers or slices, not str""
        ]
    },
    "combo_test_google_strategies" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_textF_google_gemini25_para",
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para", 
        ]
    },  
    "combo_test_google_file_strategies_4f" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para", 
        ]
    }, 
    "combo_test_google_directF_strategies_13f" : {
        "input_files" : DIR_13_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para"
        ]
    },
    "combo_test_google_directF_strategies_30f" : {
        "input_files" : DIR_30_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para"
        ]
    },    
    "combo_test_google_directF_strategies_4f" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para",
            "grp_directF_dSeek_dChat_para"
        ]
    },
    "combo_test_google_directF_strategies_1f" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_directF_google_gemini25_para"
        ]
    },    
    "combo_test_google_imageF_strategies_1f" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_imageF_google_gemini25_para"
        ]
    },
    "combo_test_google_imageF_strategies_4f" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_imageF_google_gemini25_para"
        ]
    },
    "combo_test_imageF_google_strategies" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            # "grp_textF_google_gemini25_para",
            # "grp_directF_google_gemini25_para",
            "grp_imageF_google_gemini25_para", 
        ]
    }, 
    "combo_test_claude_strategies" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para"
        ]
    },  
    "combo_test_deepseek_strategies" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            # "grp_imageF_dSeek_dChat_para", #  TODO: investigateno more this error but still failed:  WARNING - Unsupported file type: temp_images/image_5eb95081.png
            "grp_directF_dSeek_dChat_para",
            "grp_textF_dSeek_dChat_para", 
        ]
    },    
    "combo_test_3_textF_strategies" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_textF_dSeek_dChat_para",
            "grp_textF_google_gemini25_para",
            "grp_test_textF_claude_para"
        ]
    },
    "test_directF_dSeek_dChat_only" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_directF_dSeek_dChat_para"
        ]
    },
    "test_textF_dSeek_dChat_only" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_textF_dSeek_dChat_para"
        ]
    },  
    # "test_imageF_dSeek_dChat_only" : {
    #     "input_files" : DIR_1_FILES,
    #     "parameter_groups" : [
    #         "grp_imageF_dSeek_dChat_para"
    #     ]
    # },        
    "test_imageF_openai_only" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_test_imageF_openai_para"
        ]
    },
    "test_textF_openai_only" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_test_textF_openai_para"
        ]
    },
    "test_imageF_claude_only" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_test_imageF_claude_para"
        ]
    },
    "test_textF_claude_only" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_test_textF_claude_para"
        ]
    },
    "test_both_strategies_claude" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_test_imageF_claude_para",
            "grp_test_textF_claude_para"
        ]
    },
    "test_both_strategies_openai" : {
        "input_files" : DIR_4_FILES,
        "parameter_groups" : [
            "grp_test_imageF_openai_para",
            "grp_test_textF_openai_para"
        ]
    },
    "test_huggingface_models" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_imageF_huggingface_qwen_para",
            "grp_imageF_huggingface_llama_para"
        ]
    },
    "test_huggingface_qwen_only" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_imageF_huggingface_qwen_para"
        ]
    },
    "test_huggingface_llama_only" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_imageF_huggingface_llama_para"
        ]
    },
    "test_textF_ollama_single_file" : {
        "input_files" : DIR_1_FILES,
        "parameter_groups" : [
            "grp_textF_ollama_deepseek_r1_8b_para"
        ]
    }
}