"""
Utilities for loading and formatting the SQL fine-tuning dataset.
"""

from datasets import Dataset, load_dataset

DEFAULT_DATASET_NAME = "gretelai/synthetic_text_to_sql"
DEFAULT_SYSTEM_PROMPT = "You are an expert SQL query generator."
DEFAULT_SEED = 3407
DEFAULT_TRAIN_LIMIT = 10000
DEFAULT_TEST_LIMIT = 2000


def load_sql_dataset_split(
    split: str,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> Dataset:
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return dataset


def convert_to_chatml(
    example: dict,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict:
    return {
        "conversations": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"{example['sql_context']}\n\n{example['sql_prompt']}",
            },
            {
                "role": "assistant",
                "content": f"{example['sql']}\n\n{example['sql_explanation']}",
            },
        ]
    }


def formatting_prompts_func(examples: dict, tokenizer) -> dict:
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}


def prepare_sql_dataset(
    dataset: Dataset,
    tokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dataset:
    dataset = dataset.map(
        convert_to_chatml,
        fn_kwargs={"system_prompt": system_prompt},
    )
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )
    return dataset
