"""
This file loads the Gemma model with LoRA.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..utils import load_hf_token


def get_gemma_model(model_name: str = "google/gemma-3-270m-it", return_tokenizer: bool = False):
    hf_token = load_hf_token()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Gemma tokenizer from Hugging Face. "
            "Make sure your account has access to google/gemma-3-270m-it and, "
            "if you are using a fine-grained token, enable access to public gated repositories."
        ) from exc
    tokenizer.pad_token = tokenizer.eos_token

    try:
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                token=hf_token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
            )
            if torch.backends.mps.is_available():
                model = model.to("mps")
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Gemma model from Hugging Face. "
            "Confirm that you accepted the gated model access request on the model page and "
            "that your HF token can read public gated repositories."
        ) from exc

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if return_tokenizer:
        return tokenizer, model

    return model
