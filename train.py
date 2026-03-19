# !pip install -U keras keras-hub datasets kaggle -q



# # Upload kaggle.json (Colab only)
# from google.colab import files
# files.upload()

# # Move credentials
# !mkdir -p /root/.kaggle
# !mv kaggle.json /root/.kaggle/
# !chmod 600 /root/.kaggle/kaggle.json

# import os
# import json
# with open('kaggle.json') as f:
#     creds = json.load(f)
# os.environ['KAGGLE_USERNAME'] = creds['username']
# os.environ['KAGGLE_KEY'] = creds['key']

# import keras
# import keras_hub
# import tensorflow as tf
# from datasets import load_dataset

# def load_sql_dataset():
#     dataset = load_dataset("gretelai/synthetic_text_to_sql")

#     train_prompts, train_responses = [], []
#     val_prompts, val_responses = [], []

#     def build_prompt(example):
#         return f"""Context:
# {example['sql_context']}

# Question:
# {example['sql_prompt']}

# Write a SQL query."""

#     for ex in dataset["train"]:
#         train_prompts.append(build_prompt(ex))
#         train_responses.append(ex["sql"])

#     for ex in dataset["test"]:
#         val_prompts.append(build_prompt(ex))
#         val_responses.append(ex["sql"])

#     return (
#         {"prompts": train_prompts[:20000], "responses": train_responses[:20000]},
#         {"prompts": val_prompts[:2000], "responses": val_responses[:2000]},
#     )

# train_raw, val_raw = load_sql_dataset()

# BATCH_SIZE = 2

# def to_tf_dataset(data):
#     ds = tf.data.Dataset.from_tensor_slices({
#         "prompts": data["prompts"],
#         "responses": data["responses"],
#     })
#     return ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# train_data = to_tf_dataset(train_raw)
# val_data = to_tf_dataset(val_raw)

# gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(
#     "gemma3_instruct_270m"
# )

# gemma_lm.summary()

# gemma_lm.backbone.enable_lora(rank=4)
# gemma_lm.preprocessor.sequence_length = 256

# dummy = tf.constant(["Hello world"])
# _ = gemma_lm.generate(dummy)

# optimizer = keras.optimizers.AdamW(
#     learning_rate=2e-5,
#     weight_decay=0.01,
# )

# optimizer.exclude_from_weight_decay(
#     var_names=["bias", "scale"]
# )

# gemma_lm.compile(
#     optimizer=optimizer,
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )

# gemma_lm.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=2
# )

# sampler = keras_hub.samplers.TopKSampler(k=5, seed=42)
# gemma_lm.compile(sampler=sampler)

# template = """Instruction:
# {instruction}

# Response:
# """

# prompt = template.format(
#     instruction="""Context:
# Table: employees(id, name, salary)

# Question:
# Find employees with salary greater than 50000

# Write a SQL query."""
# )

# output = gemma_lm.generate(prompt, max_length=200)
# print(output)

# gemma_lm.backbone.save_lora_weights("gemma_sql_lora.lora.h5")