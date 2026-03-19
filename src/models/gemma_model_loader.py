import keras_hub
from ..utils import load_kaggle_credentials

def get_gemma_model(model_artifact:str="gemma3_instruct_270m"):
    load_kaggle_credentials()
    gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(model_artifact)
    gemma_lm.summary()
    return gemma_lm
