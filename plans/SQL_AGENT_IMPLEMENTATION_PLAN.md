# Gemma SQL Agent Implementation Plan

## Summary

Build the project in two major tracks, in order: first fine-tune a small SQL-specialized model using Gemma 270M, then build an agent runtime that uses a user-provided Markdown file as the database context source at inference time. The first working milestone is not "full autonomous analytics," but a reliable read-only SQL generator that takes a user question plus a database-context `.md` file and returns valid SQL with clear constraints.

## Key Changes

### Phase 1: Define the SQL generation contract

- Lock the exact task the model is being trained for: natural-language question in, SQL query out, PostgreSQL dialect, read-only only.
- Define the output rules the model must follow: single query only, no markdown fences in raw generation output, no explanations in the model output, no DDL/DML.
- Write a strict prompt format for training and inference so the same task shape is preserved across both stages.
- Define success criteria for the fine-tuned model before any agent work begins: valid SQL rate, read-only compliance, and schema-faithful generation on held-out examples.

### Phase 2: Build the fine-tuning data pipeline

- Create a reproducible dataset preparation pipeline around the Gretel text-to-SQL dataset.
- Filter or transform examples so they match the target runtime behavior: PostgreSQL-first, read-only, single-query generation.
- Normalize fields into a training schema such as `instruction`, `input_question`, `expected_sql`, and optional `schema_context`.
- Create train/validation/test splits and freeze them so later model comparisons are meaningful.
- Add a small hand-curated evaluation set with realistic business-style questions, because synthetic data alone will not reflect actual client asks.

### Phase 3: Fine-tune Gemma 270M

- Use Gemma 270M as the base model and set up parameter-efficient fine-tuning first, keeping memory and training cost manageable.
- Implement a training script with configurable hyperparameters, checkpointing, logging, and model export.
- Save artifacts in a clean structure: base config, adapter weights, tokenizer settings, training logs, and eval metrics.
- Treat this phase as complete only when the fine-tuned model beats the base model on the frozen validation/test set.

### Phase 4: Evaluate and harden the model before agent work

- Build an offline evaluation harness that tests syntax validity, execution validity where possible, and rule compliance.
- Categorize failure modes: wrong table, wrong join, missing filter, invalid SQL, non-read-only SQL, hallucinated columns.
- Add a lightweight post-processing or validation layer if needed, but avoid hiding fundamental model weaknesses with excessive patching.
- Use the results to decide whether one more fine-tuning round is needed before moving to the runtime agent.

### Phase 5: Define the Markdown context format

- Design a strict, user-authored `.md` template that explains the database for the runtime agent.
- Require sections for business overview, table descriptions, column meanings, relationships, allowed joins, important filters, naming quirks, and SQL-generation rules.
- Include a section for explicit constraints such as "always filter active users," "never join these tables directly," or "date column to use for revenue."
- Keep the format concise and structured enough that it can be parsed into prompt-ready sections without building a full document parser.

### Phase 6: Build the context loader for runtime

- Implement a module that ingests the Markdown file, validates required sections, and converts it into a clean runtime context object.
- Add safeguards for missing sections, contradictory rules, and oversized files that would overflow prompt limits.
- Support versioning so the agent response can reference which context document version was used.
- This context loader is the replacement for live schema introspection in the first version.

### Phase 7: Build the SQL agent runtime

- Create the end-to-end inference pipeline:
  1. accept user question
  2. load the client's Markdown context file
  3. assemble the grounded prompt
  4. call the fine-tuned Gemma model
  5. validate the SQL
  6. return SQL plus explanation and validation results
- Keep the generation model focused on producing SQL only; add explanation generation as a wrapper concern after SQL is produced.
- Make the runtime read-only by policy, with hard checks for write statements, multi-statement outputs, and unsafe constructs.
- Start with a local CLI harness and backend API rather than a UI.

### Phase 8: Validation and safety layer

- Add SQL parsing and dialect-aware checks before any query is accepted as usable.
- Reject outputs that violate project rules even if the model generated them confidently.
- Validate that the SQL aligns with the Markdown-provided context and does not use clearly forbidden tables, joins, or patterns.
- Return actionable validation messages so the user can understand whether the failure came from the model or from rule enforcement.

### Phase 9: Usable MVP and iteration loop

- Expose a minimal API and CLI to test common user flows with different Markdown context files.
- Add logging for input question, context version, raw model output, final accepted SQL, validation result, and latency.
- Collect real failure cases from manual trials and convert them into eval examples and future fine-tuning improvements.
- Declare MVP complete only when a new client can provide one Markdown database guide, ask questions through the API/CLI, and consistently receive safe, grounded SQL.

## Public APIs / Interfaces

- Training dataset record format for supervised fine-tuning of Gemma 270M.
- Markdown context template used by clients at runtime.
- `generate_sql(question, context_file)` -> returns `sql`, `explanation`, `validation`, and `context_version`.
- `validate_sql(sql, context_file)` -> returns rule-check and safety-check results.
- `run_eval(model_artifact, eval_set)` -> returns generation and safety metrics for model comparison.

## Test Plan

- Dataset pipeline tests for filtering, split stability, and prompt-format correctness.
- Training smoke tests to confirm the fine-tuning pipeline runs end-to-end on a small sample.
- Evaluation tests comparing base Gemma vs fine-tuned Gemma on the same frozen set.
- Markdown context parser tests for valid files, missing required sections, contradictory rules, and prompt-size edge cases.
- Runtime integration tests covering question -> context load -> SQL generation -> validation.
- Safety tests that ensure writes, multi-statement SQL, schema changes, and disallowed constructs are rejected.

## Assumptions and defaults

- Base model is Gemma 270M.
- Fine-tuning is done before agent/runtime development.
- Runtime context comes from a user-authored Markdown file, not live DB introspection.
- Markdown context is runtime-only, not the primary source for initial fine-tuning data.
- PostgreSQL is the first supported dialect.
- V1 is read-only and should generate SQL, not autonomously mutate data or schema.
- The first product surface is backend API plus CLI.
