"""
This file loads the Hugging Face token from .env and seeds it in environment variables.
"""

import os
from pathlib import Path


def load_hf_token(src: str = ".env"):
    token = os.environ.get("HF_TOKEN")

    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        return token

    env_path = Path(src)
    if not env_path.exists():
        return None

    for line in env_path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() != "HF_TOKEN":
            continue

        token = value.strip().strip('"').strip("'")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        return token

    return None
