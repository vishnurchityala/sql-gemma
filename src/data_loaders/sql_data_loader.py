"""
This file is to create and return tensorflow dataloader for the SQL finetuning dataset.

Step 1: Load TSV files from /data folder
Step 2: Parse all rows in pairs of prompts & responses
Step 3: Create and Return tensorflow data loader
"""
import csv
import tensorflow as tf

def get_data_loader(split:str="train",batch_size:int=2) -> tf.data.Dataset:
    rows = None
    size = None
    if split == "train":
        size = 10000
        with open("./data/train_data.tsv","r") as f:
            reader = csv.reader(f,delimiter='\t')
            rows = list(reader)[1:]
    else:
        size = 2500
        with open("./data/test_data.tsv","r") as f:
            reader = csv.reader(f,delimiter='\t')
            rows = list(reader)[1:]
    
    prompts = []
    responses = []
    for row in rows:
        prompt = f""" Context: {row[1]} Input: {row[2]}"""
        response = f""" Explanation: {row[3]} Response: {row[4]}"""
        prompts.append(prompt)
        responses.append(response)
    ds = tf.data.Dataset.from_tensor_slices({
        "prompts": prompts[:size],
        "responses": responses[:size],
    })

    return ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)        