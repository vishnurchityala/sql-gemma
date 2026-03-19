"""
This file is to fetch kaggle credentials and seed them in enviornment for Gemma model use.
"""
import os
import json

def load_kaggle_credentials(src:str='kaggle.json'):
    with open(src) as f:
        creds = json.load(f)
    os.environ['KAGGLE_USERNAME'] = creds['username']
    os.environ['KAGGLE_KEY'] = creds['key']