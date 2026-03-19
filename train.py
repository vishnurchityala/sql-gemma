"""
Train Gemma-270M on Training Dataset with AdamW
"""
import keras
import tensorflow as tf
from src.data_loaders.sql_data_loader import get_data_loader
from src.models.gemma_model_loader import get_gemma_model

# Load Dataset and Gemma Model
train_dataset = get_data_loader(split="train")
test_dataset = get_data_loader(split="test")
gemma_lm = get_gemma_model()

# Generating Pre-Finetuning Response
dummy = tf.constant(["Write a SQL Query to fetch all records from Products table"])
dummy_response = gemma_lm.generate(dummy)
print("Dummy Response Before Fine-Tuning:")
print(dummy_response)

# Finetuning Configuration
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.preprocessor.sequence_length = 256

optimizer = keras.optimizers.AdamW(
    learning_rate=2e-5,
    weight_decay=0.01,
)

optimizer.exclude_from_weight_decay(
    var_names=["bias", "scale"]
)

gemma_lm.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Training Loop
history = gemma_lm.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=1
)

# Post-Finetuning Model Testing
template = """Instruction:
{instruction}

Response:
"""

prompt = template.format(
    instruction="""Context:
Table: employees(id, name, salary)

Question:
Find employees with salary greater than 50000

Write a SQL query."""
)

output = gemma_lm.generate(prompt, max_length=200)
print(output)

# Saving LoRA Weights
gemma_lm.backbone.save_lora_weights("weights/gemma_sql_lora.lora.h5")