from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Dict, Any

from config import config_base


def _load_profile_module(profile_name: str) -> ModuleType:
    base_dir = Path(__file__).parent
    profile_py = base_dir / profile_name / "profile_config.py"
    if not profile_py.exists():
        raise FileNotFoundError(f"Profile not found: {profile_py}")

    spec = importlib.util.spec_from_file_location(f"run_profile_{profile_name}", str(profile_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load profile module: {profile_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_profile(profile_name: str) -> Dict[str, Any]:
    """Load and apply a run profile by name.

    The profile can override:
      - INPUT_DIR (path to input files)
      - OUTPUT_BASE_DIR (base dir for all outputs)
      - DEFAULT_LLM_PROVIDER, MAX_WORKERS, prompts, etc. (optional)

    Returns a dict with applied settings, including `input_dir` if set.
    """
    module = _load_profile_module(profile_name)

    applied: Dict[str, Any] = {"profile": profile_name}

    # Resolve output base dir
    default_output_base = f"Ultra_Arena_Main/run_profiles/{profile_name}/output"
    output_base_dir: str = getattr(module, "OUTPUT_BASE_DIR", default_output_base)

    # Apply output directories into config_base
    config_base.OUTPUT_BASE_DIR = output_base_dir
    config_base.OUTPUT_RESULTS_DIR = f"{output_base_dir}/results"
    config_base.OUTPUT_COMBO_DIR = f"{config_base.OUTPUT_RESULTS_DIR}/combo"
    config_base.OUTPUT_COMBO_BACKUP_DIR = f"{config_base.OUTPUT_COMBO_DIR}/backup"
    config_base.OUTPUT_COMBO_CSV_DIR = "csv"
    config_base.OUTPUT_COMBO_JSON_DIR = "json"
    config_base.OUTPUT_NON_COMBO_DIR = f"{config_base.OUTPUT_RESULTS_DIR}/non-combo"
    config_base.OUTPUT_NON_COMBO_CSV_DIR = f"{config_base.OUTPUT_NON_COMBO_DIR}/csv"
    config_base.OUTPUT_NON_COMBO_JSON_DIR = f"{config_base.OUTPUT_NON_COMBO_DIR}/json"
    config_base.OUTPUT_CHECKPOINTS_DIR = f"{output_base_dir}/checkpoints"
    config_base.OUTPUT_LOGS_DIR = f"{output_base_dir}/logs"
    config_base.OUTPUT_NOTE_GEN_DIR = f"{output_base_dir}/note_gen"

    # Ensure directories exist
    for d in [
        config_base.OUTPUT_NON_COMBO_CSV_DIR,
        config_base.OUTPUT_NON_COMBO_JSON_DIR,
        config_base.OUTPUT_CHECKPOINTS_DIR,
        config_base.OUTPUT_LOGS_DIR,
        config_base.OUTPUT_NOTE_GEN_DIR,
        config_base.OUTPUT_COMBO_DIR,
        f"{config_base.OUTPUT_COMBO_DIR}/{config_base.OUTPUT_COMBO_CSV_DIR}",
        f"{config_base.OUTPUT_COMBO_DIR}/{config_base.OUTPUT_COMBO_JSON_DIR}",
    ]:
        os.makedirs(d, exist_ok=True)

    applied.update({
        "output_base_dir": output_base_dir,
        "output_csv_dir": config_base.OUTPUT_NON_COMBO_CSV_DIR,
        "output_json_dir": config_base.OUTPUT_NON_COMBO_JSON_DIR,
    })

    # Optional overrides
    if hasattr(module, "DEFAULT_LLM_PROVIDER"):
        config_base.DEFAULT_LLM_PROVIDER = getattr(module, "DEFAULT_LLM_PROVIDER")
        applied["default_llm_provider"] = config_base.DEFAULT_LLM_PROVIDER

    if hasattr(module, "MAX_WORKERS"):
        # Not directly in config_base; kept for the caller to use if needed
        applied["max_workers"] = getattr(module, "MAX_WORKERS")

    # Prompts (optional)
    if hasattr(module, "SYSTEM_PROMPT"):
        config_base.SYSTEM_PROMPT = getattr(module, "SYSTEM_PROMPT")
        applied["system_prompt_override"] = True
    if hasattr(module, "USER_PROMPT"):
        config_base.USER_PROMPT = getattr(module, "USER_PROMPT")
        applied["user_prompt_override"] = True

    # Input directory (optional)
    if hasattr(module, "INPUT_DIR"):
        input_dir = getattr(module, "INPUT_DIR")
        applied["input_dir"] = input_dir

    return applied

