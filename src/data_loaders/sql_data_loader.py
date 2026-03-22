"""
Load the SQL fine-tuning dataset in the chat format expected by Unsloth SFT.
"""

from datasets import DatasetDict

from ..utils.load_dataset_sql import (
    DEFAULT_SEED,
    DEFAULT_TEST_LIMIT,
    DEFAULT_TRAIN_LIMIT,
    load_sql_dataset_split,
    prepare_sql_dataset,
)


def get_data_loader(
    tokenizer,
    split: str | None = None,
    train_limit: int = DEFAULT_TRAIN_LIMIT,
    test_limit: int = DEFAULT_TEST_LIMIT,
    seed: int = DEFAULT_SEED,
):
    dataset = DatasetDict(
        {
            "train": prepare_sql_dataset(
                load_sql_dataset_split("train", limit=train_limit, seed=seed),
                tokenizer=tokenizer,
            ),
            "test": prepare_sql_dataset(
                load_sql_dataset_split("test", limit=test_limit, seed=seed),
                tokenizer=tokenizer,
            ),
        }
    )

    if split is not None:
        return dataset[split]

    return dataset
