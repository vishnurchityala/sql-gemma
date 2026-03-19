"""
This file loads the SQL fine-tuning dataset.
"""

from pathlib import Path

from datasets import load_dataset


def format_example(example):
    return {
        "text": f"""Context:
{example['context']}

Question:
{example['prompt']}

Write a SQL query:
{example['response']}"""
    }


def _get_data_file(split):
    if split == "train":
        candidates = ["train.tsv", "data/train_data.tsv"]
    else:
        candidates = ["test.tsv", "data/test_data.tsv"]

    for file_name in candidates:
        if Path(file_name).exists():
            return file_name

    raise FileNotFoundError(f"Could not find a local {split} TSV file.")


def get_data_loader(split=None, batch_size=2):
    del batch_size

    dataset = load_dataset(
        "csv",
        data_files={
            "train": _get_data_file("train"),
            "test": _get_data_file("test"),
        },
        delimiter="\t",
    )
    dataset = dataset.map(format_example)

    if split is not None:
        return dataset[split]

    return dataset
