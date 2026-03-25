# SQL - Gemma-270M

## Overview

This project implements a fine-tuning pipeline for large language models (LLMs) specialized in SQL query generation. The implementation leverages the Gemma-270M parameter model as the base LLM and utilizes a comprehensive synthetic dataset from Hugging Face containing 100,000 rows of SQL query examples.

## Project Details

### Objective
Develop and train a domain-specific LLM capable of generating accurate and efficient SQL queries through supervised fine-tuning techniques.

### Model Architecture
- **Base Model**: Gemma-3-270M
  - URL: https://huggingface.co/google/gemma-3-270m-it
  - Parameter Count: 270M (selected due to computational constraints)

### Dataset
- **Source**: Gretel AI Synthetic Text-to-SQL Dataset
  - URL: https://huggingface.co/datasets/gretelai/synthetic_text_to_sql
  - Size: 100,000 rows
  - Type: Synthetic text-to-SQL query pairs

## Implementation Notes

This is a focused implementation designed to demonstrate fine-tuning capabilities within computational resource constraints. The use of a smaller model variant allows for efficient training and inference while maintaining practical utility for SQL query generation tasks.

### Difference between Base-LLM and Instruction LLM

Base LLM is pre-trained text generation model, where the goal of the model is to complete the sentence provided using training weights learned on huge corpus of data.

Whereas Instruction LLM model is a Base LLM finetuned on instructions data where the goal of the LLM model is to follow human task and complete the given task.

| Feature         | Base LLM           | Instruction-Tuned LLM                 |
| --------------- | ------------------ | ------------------------------------- |
| Training        | Pretraining only   | Pretraining + instruction fine-tuning |
| Objective       | Predict next token | Follow instructions                   |
| Prompt behavior | Continues text     | Executes tasks                        |
| Reliability     | Lower for tasks    | Higher                                |
| Alignment       | Minimal            | Human-aligned                         |



## Some Usefull Links

- https://github.com/adithya-s-k/AI-Engineering.academy/tree/main/archives
- https://huggingface.co/google/gemma-3-270m-it
- https://huggingface.co/datasets/gretelai/synthetic_text_to_sql
- https://huggingface.co/blog/gemma-peft
- https://github.com/google-gemini/gemma-cookbook/tree/main/Gemma

## Multi-Model Inference App

This repo now includes a local Streamlit app that uses two models:

- `gemini-3-flash-preview` for parsing and summarizing uploaded `context.md`
- `vishnurchityala/sql-gemma3` for SQL query generation only

### Architecture

1. Upload `context.md` in the Streamlit UI.
2. Gemini summarizes schema, relationships, and rules into a structured object.
3. The app validates that structure.
4. SQL-Gemma receives the normalized context and user question.
5. A sanitizer enforces a single read-only SQL statement.
6. The app returns SQL only.

### Run the App

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Google API key (required for Gemini summarization) in `.env`:

```bash
echo "GOOGLE_API_KEY=your_api_key_here" >> .env
```

Launch Streamlit:

```bash
streamlit run streamlit_app.py
```

### `context.md` Template

Use a markdown file with clear schema details, for example:

```md
# Overview
HR analytics schema.

# Dialect
PostgreSQL

# Tables
## employees
Description: Employee records
Columns:
- id
- name
- department
- salary

## departments
Description: Department budgets
Columns:
- department
- budget

# Relationships
- employees.department = departments.department

# Rules
- Return one read-only SQL query.
- Do not use markdown fences in SQL output.
```

You can also start with the included sample file: `context.sample.md`.

### Notes

- First run can be slow due to model downloads.
- SQL is generated but not executed in this version.
- The app is designed for local development and single-turn generation.
