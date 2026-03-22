"""
Train Gemma 3 270M on the SQL dataset with Unsloth SFT.
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from src.data_loaders.sql_data_loader import get_data_loader
from src.models.gemma_model_loader import get_gemma_model

OUTPUT_DIR = "outputs"
LORA_OUTPUT_DIR = "gemma_3_270m_lora_sql"
MERGED_OUTPUT_DIR = "gemma_3_270m_full_sql"
TRAINING_CURVE_CSV = "training_curve.csv"
TRAINING_CURVE_PNG = "training_curve.png"
SAMPLE_INDEX = 10


def export_training_curve(log_history):
    df = pd.DataFrame(log_history)
    metric_columns = [column for column in ("step", "loss", "eval_loss") if column in df.columns]

    if not metric_columns:
        raise RuntimeError("Trainer logs did not contain any of the expected loss columns.")

    df = df[metric_columns].dropna(how="all")
    df.to_csv(TRAINING_CURVE_CSV, index=False)

    plt.figure()

    if "loss" in df.columns:
        plt.plot(df["step"], df["loss"], label="Train Loss")

    if "eval_loss" in df.columns:
        plt.plot(df["step"], df["eval_loss"], label="Eval Loss")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.savefig(TRAINING_CURVE_PNG)
    plt.close()


def run_sample_generation(model, tokenizer, dataset, sample_index: int = SAMPLE_INDEX):
    sample_index = min(sample_index, len(dataset) - 1)
    conversations = dataset["conversations"][sample_index]
    messages = [
        {"role": "system", "content": conversations[0]["content"]},
        {"role": "user", "content": conversations[1]["content"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix("<bos>")

    device = next(model.parameters()).device

    with torch.inference_mode():
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to(device),
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )


def main():
    tokenizer, model = get_gemma_model(return_tokenizer=True)
    dataset = get_data_loader(tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            num_train_epochs=2,
            learning_rate=5e-5,
            logging_steps=50,
            eval_steps=500,
            save_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    trainer.train()

    run_sample_generation(model, tokenizer, dataset["train"])

    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

    export_training_curve(trainer.state.log_history)

    print(f"LoRA adapter saved to {LORA_OUTPUT_DIR}")
    print(f"Merged model saved to {MERGED_OUTPUT_DIR}")
    print(f"Training curve CSV saved to {TRAINING_CURVE_CSV}")
    print(f"Training curve plot saved to {TRAINING_CURVE_PNG}")


if __name__ == "__main__":
    main()
