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
# from src.models.gemma_model_loader import get_gemma_model

# gemma_lm = get_gemma_model()

""" Using Unsloth & Fine-Tuning Gemma-3 on dataset """

# from unsloth import FastModel
# from datasets import load_dataset
# from unsloth.chat_templates import get_chat_template, train_on_responses_only
# from trl import SFTTrainer, SFTConfig
# from transformers import TextStreamer
# import matplotlib.pyplot as plt
# import pandas as pd

# max_seq_length = 2048

# model, tokenizer = FastModel.from_pretrained(
#     model_name = "unsloth/gemma-3-270m-it",
#     max_seq_length = max_seq_length,
#     load_in_4bit = True,
#     full_finetuning = False,
# )

# model = FastModel.get_peft_model(
#     model,
#     r = 128,
#     target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha = 128,
#     lora_dropout = 0,
#     bias = "none",
#     use_gradient_checkpointing = "unsloth",
#     random_state = 3407,
# )

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "gemma3",
# )

# train_ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
# test_ds  = load_dataset("gretelai/synthetic_text_to_sql", split="test")
# train_ds = train_ds.shuffle(seed=3407).select(range(10000))
# test_ds = test_ds.shuffle(seed=3407).select(range(2000))

# def convert_to_chatml(example):
#     return {
#         "conversations": [
#             {
#                 "role": "system",
#                 "content": "You are an expert SQL query generator."
#             },
#             {
#                 "role": "user",
#                 "content": example["sql_context"] + "\n\n" + example["sql_prompt"]
#             },
#             {
#                 "role": "assistant",
#                 "content": example["sql"] + "\n\n" + example["sql_explanation"]
#             }
#         ]
#     }

# train_ds = train_ds.map(convert_to_chatml)
# test_ds  = test_ds.map(convert_to_chatml)

# def formatting_prompts_func(examples):
#     convos = examples["conversations"]
#     texts = [
#         tokenizer.apply_chat_template(
#             convo,
#             tokenize=False,
#             add_generation_prompt=False
#         ).removeprefix("<bos>")
#         for convo in convos
#     ]
#     return {"text": texts}

# train_ds = train_ds.map(formatting_prompts_func, batched=True)
# test_ds  = test_ds.map(formatting_prompts_func, batched=True)

# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = train_ds,
#     eval_dataset = test_ds,
#     args = SFTConfig(
#         dataset_text_field = "text",
#         per_device_train_batch_size = 4,
#         gradient_accumulation_steps = 1,
#         warmup_steps = 100,

#         num_train_epochs = 2,   # FIXED

#         learning_rate = 5e-5,
#         logging_steps = 50,
#         eval_steps = 500,
#         save_steps = 500,

#         optim = "adamw_8bit",
#         weight_decay = 0.001,
#         lr_scheduler_type = "linear",

#         seed = 3407,
#         output_dir = "outputs",
#         report_to = "none",
#     ),
# )

# trainer = train_on_responses_only(
#     trainer,
#     instruction_part = "<start_of_turn>user\n",
#     response_part = "<start_of_turn>model\n",
# )

# trainer.train()

# messages = [
#     {"role": "system", "content": train_ds["conversations"][10][0]["content"]},
#     {"role": "user", "content": train_ds["conversations"][10][1]["content"]},
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# ).removeprefix("<bos>")

# _ = model.generate(
#     **tokenizer(text, return_tensors="pt").to("cuda"),
#     max_new_tokens=150,
#     temperature=0.7,
#     top_p=0.9,
#     top_k=50,
#     streamer=TextStreamer(tokenizer, skip_prompt=True),
# )

# model.save_pretrained("gemma_3_270m_lora_sql")
# tokenizer.save_pretrained("gemma_3_270m_lora_sql")

# merged_model = model.merge_and_unload()

# merged_model.save_pretrained("gemma_3_270m_full_sql")
# tokenizer.save_pretrained("gemma_3_270m_full_sql")

# logs = trainer.state.log_history

# df = pd.DataFrame(logs)

# df = df[["step", "loss", "eval_loss"]].dropna(how="all")

# csv_path = "training_curve.csv"
# df.to_csv(csv_path, index=False)

# csv_path

# plt.figure()

# if "loss" in df:
#     plt.plot(df["step"], df["loss"], label="Train Loss")

# if "eval_loss" in df:
#     plt.plot(df["step"], df["eval_loss"], label="Eval Loss")

# plt.xlabel("Steps")
# plt.ylabel("Loss")
# plt.title("Training Curve")
# plt.legend()

# plot_path = "training_curve.png"
# plt.savefig(plot_path)

# plot_path

""" Loading Finetuned Model """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./weights/gemma3_1b_sql_full"

tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
).to(device)

model.eval()

prompt = (
    "<start_of_turn>user\n"
    "CREATE TABLE customers(id INT, name TEXT);\n"
    "CREATE TABLE orders(id INT, customer_id INT, order_date DATE);\n"
    "CREATE TABLE order_items(id INT, order_id INT, product_id INT, quantity INT);\n"
    "CREATE TABLE products(id INT, name TEXT, price FLOAT);\n\n"
    
    "Find the names of customers who have spent more than 500 in total across all their orders.\n"
    "Return customer name and total spending.\n"
    
    "<start_of_turn>model\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

result = response.split("<start_of_turn>model")[-1].strip()

print("\n===== MODEL OUTPUT =====\n")
print(result)