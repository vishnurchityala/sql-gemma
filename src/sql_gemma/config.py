"""Configuration for the multi-model SQL runtime."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv
import torch

# Load environment variables from .env for local development.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings."""

    sql_model_id: str = "vishnurchityala/sql-gemma3"
    gemini_model_id: str = "gemini-3-flash-preview"
    sql_max_new_tokens: int = 160
    sql_do_sample: bool = False
    gemini_temperature: float = 0.0

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def google_api_key(self) -> str:
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Missing GOOGLE_API_KEY. Add it to .env or your shell environment.")
        return key
