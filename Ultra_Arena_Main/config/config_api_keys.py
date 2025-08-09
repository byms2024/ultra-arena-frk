"""
Centralized API keys for providers. Override via environment or profiles as needed.
Do NOT commit real secrets.
"""

import os

GCP_API_KEY = os.getenv("GCP_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", os.getenv("CLAUDE_API_KEY", ""))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", os.getenv("HUGGINGFACE_TOKEN", ""))

