"""
Train Gemma 3 270M on the SQL dataset with Unsloth SFT.
"""
import os
import torch
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from peft import PeftModel

max_seq_length = 1024
output_lora = "./weights/gemma3_270m_sql_lora"
output_full = "./weights/gemma3_270m_sql_full"

os.makedirs(output_lora, exist_ok=True)
os.makedirs(output_full, exist_ok=True)

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","v_proj","o_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma3",
)

train_ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
test_ds  = load_dataset("gretelai/synthetic_text_to_sql", split="test")

train_ds = train_ds.shuffle(seed=42).select(range(10000))
test_ds  = test_ds.shuffle(seed=42).select(range(2000))

def convert_to_chatml(example):
    return {
        "conversations": [
            {
                "role": "user",
                "content": example["sql_context"] + "\n\n" + example["sql_prompt"]
            },
            {
                "role": "assistant",
                "content": example["sql"]
            }
        ]
    }

train_ds = train_ds.map(convert_to_chatml, num_proc=2)
test_ds  = test_ds.map(convert_to_chatml, num_proc=2)

def formatting_prompts_func(examples):
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        ).removeprefix("<bos>")
        for convo in examples["conversations"]
    ]
    return {"text": texts}

train_ds = train_ds.map(formatting_prompts_func, batched=True, num_proc=2)
test_ds  = test_ds.map(formatting_prompts_func, batched=True, num_proc=2)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        packing=False,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=20,
        eval_steps=200,
        optim="adamw_8bit",
        output_dir="./logs/",
        report_to="none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

trainer.train()

trainer.model.save_pretrained(output_lora)
tokenizer.save_pretrained(output_lora)

print("LoRA adapter saved.")

base_model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=max_seq_length,
    load_in_4bit=False,
)

merged = PeftModel.from_pretrained(base_model, output_lora)
merged = merged.merge_and_unload()

merged.save_pretrained(output_full)
tokenizer.save_pretrained(output_full)

print("Full model saved.")

prompt = (
    "<start_of_turn>user\n"
    "CREATE TABLE employees(id, name, salary);\n\n"
    "Find average salary.\n"
    "<start_of_turn>model\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = merged.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))