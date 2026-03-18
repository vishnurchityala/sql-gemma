# Phase 1 Plan: Define the SQL Generation Contract

## Summary

Phase 1 is the specification and alignment phase for the model. Its purpose is to define exactly what Gemma 270M will be fine-tuned to do before any dataset curation or training begins.

The goal of this phase is to freeze the SQL generation contract for v1: what input the model receives, what output it must produce, which SQL behaviors are in scope, which are forbidden, and how success will be measured. The output of this phase should be strong enough that Phase 2 can prepare the Gretel dataset without guessing product intent.

## Key Changes

### 1. Define the exact model task

- Freeze the core task statement:
  - given a natural-language analytics question, generate one PostgreSQL read-only SQL query
- Define the model boundary clearly:
  - the model produces SQL only
  - explanation, repair, execution, and user messaging belong to later runtime layers
- Define the target user intent for v1:
  - analytics and reporting questions
  - not operational write actions
  - not database administration tasks

### 2. Freeze the raw output contract

- Define the only acceptable model output as a single SQL query string.
- Disallow:
  - markdown code fences
  - explanatory prose
  - comments
  - multiple statements
  - placeholders like "your_table"
- Define formatting expectations:
  - output must be directly executable SQL
  - avoid `SELECT *` unless explicitly justified
  - aliases should be readable and deterministic
- Define the failure rule:
  - any output that is not one clean PostgreSQL query is non-compliant

### 3. Define the v1 SQL scope

- Allow only read-only query generation using:
  - `SELECT`
  - `WHERE`
  - `ORDER BY`
  - `GROUP BY`
  - `HAVING`
  - `LIMIT`
  - joins
  - aggregate functions
  - simple subqueries if needed for analytics use cases
- Disallow:
  - `INSERT`
  - `UPDATE`
  - `DELETE`
  - `CREATE`
  - `ALTER`
  - `DROP`
  - transactions
  - stored procedures
  - temp tables
  - recursive or highly advanced SQL unless it becomes a requirement later
- Freeze PostgreSQL as the only supported dialect in this phase.

### 4. Define the input contract and prompt format

- Define the fixed input structure for training and inference:
  - instruction
  - user question
  - optional schema or context placeholder
- Decide that `schema_context` remains part of the data shape even if Phase 1 does not fully use runtime Markdown context yet.
- Write the canonical prompt template that later phases must reuse.
- Keep the training and inference prompt behavior aligned so the model is not trained on one task and deployed on another.
- Include a few representative examples to demonstrate exactly what compliant input/output pairs look like.

### 5. Define evaluation rules before training starts

- Freeze the success metrics that later phases must report:
  - SQL syntax validity
  - read-only compliance
  - normalized SQL match
  - execution success where executable evaluation is possible
- Define how SQL should be normalized for fair comparison:
  - ignore whitespace and non-semantic formatting differences
  - do not ignore wrong joins, wrong filters, wrong tables, or wrong aggregation logic
- Define baseline comparison rules:
  - later training must compare base Gemma 270M against fine-tuned Gemma 270M on the same frozen eval set

### 6. Define the failure taxonomy

- Create a stable list of failure categories for all later evaluation:
  - invalid SQL
  - non-read-only SQL
  - hallucinated table
  - hallucinated column
  - wrong join
  - wrong filter
  - wrong aggregation
  - wrong ordering/top-N logic
  - incomplete query
  - extra prose in output
- Require later eval reports to classify failures using this taxonomy so model improvement is measurable.

### 7. Create a small gold reference set

- Build a hand-written reference set of representative v1 examples covering:
  - simple retrieval
  - filtering
  - grouping
  - aggregation
  - joins
  - date range queries
  - top-N reporting
- Add a small anti-example set for forbidden requests such as:
  - delete a record
  - update a table
  - create a table
  - run multiple queries
- Use this set as the first contract-check artifact before data preparation starts.

## Implementation Tasks

1. Write the one-sentence model task definition and freeze it as the source of truth.
2. Write the v1 in-scope and out-of-scope SQL rules.
3. Define the exact raw output compliance rules.
4. Draft the canonical prompt template for training and inference.
5. Define the future-facing training record shape including `schema_context`.
6. Define the evaluation metrics and SQL normalization policy.
7. Define the failure taxonomy for later model analysis.
8. Write a small gold example set and anti-example set to validate the contract.
9. Review all Phase 1 artifacts together and remove contradictions before Phase 2 begins.

## Expected Outputs

- One SQL generation contract document.
- One prompt template specification.
- One v1 SQL rules document with allowed and forbidden behaviors.
- One evaluation definition sheet with metrics and failure labels.
- One small gold reference set of valid examples.
- One anti-example set of forbidden requests and outputs.

## Expectations From This Phase

- Phase 2 should be able to ingest and filter the Gretel dataset without guessing what the product wants.
- Phase 3 should know exactly what Gemma 270M is expected to learn.
- The project should have one stable definition of "good SQL generation" before any training work begins.
- There should be no ambiguity around SQL dialect, read-only scope, output shape, or evaluation criteria.
- Later runtime safety rules should be enforcing a contract already defined here, not inventing new behavior.

## Brief Roadmap

1. Lock task definition and scope.
2. Lock output behavior and prompt structure.
3. Lock SQL capability boundaries for v1.
4. Lock evaluation metrics and failure categories.
5. Validate the contract with a gold example set.
6. Freeze Phase 1 artifacts for use in dataset preparation.

## Test Plan

- Review representative analytics questions and verify each has one clear expected SQL behavior.
- Check that every positive example conforms to the output contract exactly.
- Check that every forbidden example is clearly rejected by the written scope rules.
- Check that the evaluation rules can distinguish syntax failure, safety failure, and semantic failure.
- Confirm that another engineer could start Phase 2 from these artifacts without needing product clarification.

## Assumptions and defaults

- Base model is Gemma 270M.
- PostgreSQL is the only target dialect in v1.
- V1 is read-only only.
- The model returns SQL only.
- Client Markdown context is a later runtime concern, but `schema_context` remains in the data contract for forward compatibility.
- The Gretel dataset will be curated against the rules defined in this phase rather than used as-is.
