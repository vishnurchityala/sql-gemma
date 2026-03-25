"""Local Transformers runtime for the fine-tuned SQL model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Settings


class SQLGemmaRuntime:
    """Thin runtime wrapper around the fine-tuned Hugging Face model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(settings.sql_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(settings.sql_model_id)
        self.model.to(settings.device)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate raw model output for the SQL task."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.settings.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.sql_max_new_tokens,
                do_sample=self.settings.sql_do_sample,
            )

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_tokens:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
