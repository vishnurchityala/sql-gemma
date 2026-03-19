"""
Loading and Saving Dataset in TSV format
"""
# from src.utils import load_save_dataset

# load_save_dataset("./data/train_data.tsv","./data/test_data.tsv")

# import pandas as pd

# data = pd.read_csv("./data/train_data.tsv",sep="\t")

""" Test Script Saving Kaggle Credentials in Env"""
# import os
# import json
# with open('kaggle.json') as f:
#     creds = json.load(f)
# os.environ['KAGGLE_USERNAME'] = creds['username']
# os.environ['KAGGLE_KEY'] = creds['key']

""" Loading and Testing Tensorflow Dataset object"""
# from src.data_loaders import sql_data_loader

# sql_dataset = sql_data_loader.get_data_loader()

# print(next(iter(sql_dataset)))

""" Loading Gemma Model """
from src.models.gemma_model_loader import get_gemma_model

gemma_lm = get_gemma_model()