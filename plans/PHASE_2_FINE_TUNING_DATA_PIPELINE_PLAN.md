# Phase 2 Plan: Build the Fine-Tuning Data Pipeline

## Summary

Phase 2 turns the Phase 1 SQL contract into a clean, reproducible training dataset for Gemma 270M. The purpose of this phase is not model training yet, but transforming the raw Gretel `synthetic_text_to_sql` dataset into a PostgreSQL-first, read-only, contract-compliant corpus that can be used safely for supervised fine-tuning.

By the end of this phase, the project should have a frozen training pipeline, a cleaned dataset split into train/validation/test sets, a small hand-curated evaluation set, and clear reports showing what was kept, dropped, rewritten, and why.

## Key Changes

### 1. Define the raw-to-clean dataset workflow

- Use `gretelai/synthetic_text_to_sql` as the primary raw dataset source for Phase 2.
- Build a deterministic ingestion flow that reads the dataset fields needed for training and analysis:
  - `sql_prompt`
  - `sql_context`
  - `sql`
  - `sql_explanation`
  - metadata such as complexity and task type where available
- Freeze the ingestion path so future runs produce the same raw intermediate representation.

### 2. Filter the dataset against the Phase 1 contract

- Remove all examples that violate the v1 read-only scope:
  - `INSERT`
  - `UPDATE`
  - `DELETE`
  - DDL and schema-management queries
  - multi-statement outputs
- Remove or flag examples that produce prose, mixed formatting, or non-executable target SQL.
- Filter out malformed records such as missing prompt, missing SQL, empty target, or broken context.
- Create explicit drop reasons so every excluded sample is categorized rather than silently discarded.

### 3. Align the dataset to PostgreSQL

- Inspect the retained SQL for dialect-specific functions and syntax that do not fit PostgreSQL.
- Remove or rewrite examples that depend on non-PostgreSQL semantics where safe transformation is possible.
- Normalize target SQL into a consistent PostgreSQL-oriented style:
  - stable casing strategy
  - consistent alias usage
  - normalized quoting policy where practical
- Treat dialect correction as a documented transformation step, not an ad hoc cleanup pass.

### 4. Normalize records into the project training schema

- Convert the retained dataset into a project-owned training format such as:
  - `instruction`
  - `question`
  - `schema_context`
  - `target_sql`
  - `task_type`
  - `complexity`
  - `source_dataset`
  - `sample_id`
- Map `sql_prompt` to the user-question field.
- Preserve `sql_context` as the schema/context field for fine-tuning, while keeping in mind it is not the same as the future client Markdown runtime guide.
- Add enough metadata to support debugging, re-filtering, and eval slicing later.

### 5. Create reproducible dataset splits

- Generate train/validation/test splits from the cleaned dataset using a fixed seed and stable logic.
- Keep the split procedure deterministic so model comparisons later are fair.
- Prevent accidental leakage of near-duplicate examples across splits where possible.
- Save the cleaned split manifests so later training runs use identical data boundaries.

### 6. Add a hand-curated evaluation set

- Create a small PostgreSQL-first eval set outside the raw Gretel data to reflect the exact v1 task better.
- Include representative analytics questions covering:
  - filtering
  - joins
  - aggregation
  - grouping
  - date logic
  - top-N queries
- Include anti-examples and difficult edge cases that test contract compliance:
  - write requests
  - ambiguous prompts
  - prompts that tempt the model to hallucinate schema
- Keep this eval set frozen and versioned so it becomes part of the long-term benchmark.

### 7. Add dataset quality reporting

- Produce a report for each pipeline run showing:
  - total raw samples
  - total kept samples
  - total dropped samples
  - counts by drop reason
  - counts by complexity level
  - counts by task type
- Report how many records needed PostgreSQL rewriting versus direct acceptance.
- Surface examples of common bad rows so future cleanup work is informed by evidence.

### 8. Freeze the Phase 2 outputs for training

- Store the final cleaned dataset in a stable project-owned format for Phase 3.
- Freeze:
  - split files
  - cleaning rules
  - transformation rules
  - eval-set version
- Treat this frozen output as the official training input for the first Gemma 270M fine-tuning run.

## Public APIs / Interfaces

- Raw ingestion record:
  - dataset-native fields from Gretel for traceability
- Cleaned training record:
  - `instruction`
  - `question`
  - `schema_context`
  - `target_sql`
  - `task_type`
  - `complexity`
  - `sample_id`
  - `source_dataset`
- Data pipeline outputs:
  - cleaned dataset files
  - split manifests
  - drop-reason report
  - transformation report
  - hand-curated eval set

## Implementation Tasks

1. Define the raw ingestion schema for the Gretel dataset.
2. Implement dataset loading and raw record extraction.
3. Implement contract-based filters for read-only and single-statement compliance.
4. Implement malformed-record checks and drop-reason labeling.
5. Implement PostgreSQL compatibility filtering and rewrite rules.
6. Normalize retained examples into the project training schema.
7. Build deterministic train/validation/test split generation.
8. Create a small hand-curated PostgreSQL eval set.
9. Add dataset statistics and quality reports.
10. Freeze and version the cleaned dataset outputs for Phase 3.

## Expected Outputs

- One reproducible raw ingestion pipeline for the Gretel dataset.
- One cleaned and normalized project training dataset.
- One deterministic train/validation/test split definition.
- One PostgreSQL-focused hand-curated eval set.
- One quality report showing kept, dropped, rewritten, and malformed records.
- One documented data contract for what Phase 3 will train on.

## Expectations From This Phase

- Phase 3 should be able to start fine-tuning without needing to inspect raw Hugging Face rows manually.
- The cleaned data should match the Phase 1 SQL contract closely enough that training does not teach the model conflicting behavior.
- The team should understand exactly what kinds of examples were lost due to read-only restrictions or dialect mismatch.
- The project should have a baseline dataset version that can be rerun and compared over time.
- Dataset quality decisions should be reproducible and auditable, not based on one-off notebook cleanup.

## Brief Roadmap

1. Ingest the raw Gretel dataset into a stable intermediate format.
2. Filter out contract-violating and malformed examples.
3. Remove or rewrite non-PostgreSQL examples.
4. Normalize the retained examples into the project training schema.
5. Generate deterministic train/validation/test splits.
6. Add a small project-owned eval set that reflects the true product task.
7. Produce dataset quality reports and freeze the output for training.

## Test Plan

- Verify raw ingestion loads all required Gretel fields consistently.
- Verify write-operation and multi-statement examples are excluded.
- Verify malformed rows are caught and assigned explicit drop reasons.
- Verify PostgreSQL incompatibilities are either transformed correctly or removed.
- Verify the cleaned training schema is complete for every retained sample.
- Verify dataset splits are deterministic across repeated runs.
- Verify no obvious near-duplicate leakage appears across train and eval splits.
- Verify the hand-curated eval set conforms exactly to the Phase 1 SQL contract.
- Verify quality reports accurately match the final dataset contents.

## Assumptions and defaults

- The Gretel dataset is the initial raw corpus, not the final training dataset as-is.
- PostgreSQL is the only supported dialect in this phase.
- V1 remains read-only only.
- `sql_prompt` is the main natural-language input field and `sql` is the target generation field.
- `sql_context` is preserved for training compatibility even though runtime grounding will later come from client-authored Markdown.
- A small project-owned eval set is necessary because the synthetic dataset alone is not sufficient to measure real product behavior.
