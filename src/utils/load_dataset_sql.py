"""
This file is created to load and save the Traning prompts with Responses in csv file.

Step 1: Load dataset from hugging face
Step 2: Parse them to desired input for LLM Finetuning
Step 3: Save the dataset as csv file
"""

from .load_hf_token import load_hf_token

def load_save_dataset(train_path:str="train_data.tsv",test_path:str="test_data.tsv") -> None:

    # Loading the dataset
    import csv
    from datasets import load_dataset

    hf_token = load_hf_token()

    train_ds = load_dataset("gretelai/synthetic_text_to_sql",split="train", token=hf_token)
    test_ds = load_dataset("gretelai/synthetic_text_to_sql",split="test", token=hf_token)

    # Iterating over whole dataset and collecting training and testing data rows
    it = iter(train_ds)
    train_data = [['id','context','prompt','response','explanation']]
    try:
        while True:
            record = next(it)
            record_row = [record['id'],record["sql_context"],record["sql_prompt"],record["sql"],record["sql_explanation"]]
            train_data.append(record_row)
    except StopIteration:
        pass

    it = iter(test_ds)
    test_data = [['id','context','prompt','response','explanation']]
    try:
        while True:
            record = next(it)
            record_row = [record['id'],record["sql_context"],record["sql_prompt"],record["sql"],record["sql_explanation"]]
            test_data.append(record_row)
    except StopIteration:
        pass


    # Saving data rows in tsv files
    with open(train_path,"w+") as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(train_data)
        print(f"Saved {len(train_data)} Training Rows")

    with open(test_path,"w+") as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(test_data)
        print(f"Saved {len(test_data)} Testing Rows")
