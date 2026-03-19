"""
Train Gemma on Training Dataset with Transformers and PEFT
"""

import torch
from transformers import TrainingArguments, Trainer

from src.data_loaders.sql_data_loader import get_data_loader
from src.models.gemma_model_loader import get_gemma_model

# Load Dataset
dataset = get_data_loader()

# Load Model + Tokenizer
tokenizer, model = get_gemma_model(return_tokenizer=True)

# Tokenization
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    tokenized["labels"] = [
        [token if token != tokenizer.pad_token_id else -100 for token in row]
        for row in tokenized["input_ids"]
    ]
    return tokenized


tokenized_dataset = dataset.map(tokenize, batched=True)

# Training Config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100,
    fp16=False,
    dataloader_pin_memory=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train
trainer.train()

# Save Fine-Tuned Model
save_path = "./fine_tuned_gemma_sql"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Inference
def generate_sql(prompt):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=5,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Test Prompt
prompt = """Context:
Table: employees(id, name, salary)

Question:
Find employees with salary greater than 50000

Write a SQL query:
"""

print(generate_sql(prompt))
