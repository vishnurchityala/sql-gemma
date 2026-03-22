"""
Load the Unsloth Gemma model and tokenizer for SQL fine-tuning.
"""

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

from ..utils import load_hf_token

DEFAULT_MODEL_NAME = "unsloth/gemma-3-270m-it"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_RANDOM_STATE = 3407
DEFAULT_LORA_RANK = 128
DEFAULT_LORA_ALPHA = 128
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_gemma_model(
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    full_finetuning: bool = False,
    return_tokenizer: bool = False,
):
    load_hf_token()

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=full_finetuning,
    )

    model = FastModel.get_peft_model(
        model,
        r=DEFAULT_LORA_RANK,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=DEFAULT_RANDOM_STATE,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma3",
    )

    if return_tokenizer:
        return tokenizer, model

    return model
