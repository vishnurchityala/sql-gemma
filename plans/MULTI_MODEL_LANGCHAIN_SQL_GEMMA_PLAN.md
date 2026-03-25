# Multi-Model LangChain Plan: `gemini-3-flash-preview` + `sql-gemma3`

## Summary

This plan defines a two-model SQL generation application:

1. `gemini-3-flash-preview` is used for context parsing and summarization.
2. `vishnurchityala/sql-gemma3` is used only for SQL generation.
3. Python code validates and normalizes context before generation.
4. A deterministic sanitizer enforces the one-query output contract.
5. A Streamlit app provides the local user interface.

This split keeps each model focused on one job:

- Gemini handles noisy, user-authored markdown context.
- SQL-Gemma handles SQL generation from a compact structured schema summary.

The result is simpler prompting, clearer debugging, and a more reliable path to a small agent later.

## Architecture

### Model responsibilities

- `gemini-3-flash-preview`
  - parse uploaded `context.md`
  - summarize schema, relationships, rules, and dialect
  - return structured output through LangChain
- `vishnurchityala/sql-gemma3`
  - receive normalized SQL context plus the user question
  - generate exactly one SQL query
- Python guardrails
  - validate Gemini structured output
  - convert context into internal typed objects
  - sanitize SQL-Gemma output
  - reject empty, multi-statement, or non-read-only output

### High-level request flow

1. User uploads `context.md` in Streamlit.
2. Gemini summarizes the file into structured schema data.
3. The app validates and stores that summary.
4. User asks a natural-language analytics question.
5. SQL-Gemma receives the normalized context and question.
6. SQL output is sanitized.
7. Streamlit displays SQL only.

### Why typed objects still matter

Even though Gemini parses the markdown, the app still needs a stable internal shape so downstream code does not depend on raw markdown or raw model responses.

Example:

```python
from dataclasses import dataclass, field

@dataclass
class TableSchema:
    name: str
    description: str
    columns: list[str]

@dataclass
class ContextSummary:
    overview: str
    dialect: str
    tables: list[TableSchema]
    relationships: list[str]
    rules: list[str] = field(default_factory=list)
    glossary: list[str] = field(default_factory=list)
```

`TableSchema` represents one table after Gemini has summarized the context file. It gives the rest of the system a clean object like:

```python
TableSchema(
    name="employees",
    description="Employee master records",
    columns=["id", "name", "department", "salary"],
)
```

## Target Project Shape

```text
sql-gemma/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── src/sql_gemma/
    ├── __init__.py
    ├── config.py
    ├── types.py
    ├── schemas.py
    ├── gemini_context_chain.py
    ├── sql_prompt_builder.py
    ├── sql_gemma_runtime.py
    ├── sanitizer.py
    ├── pipeline.py
    └── tools.py
```

## Implementation Details

### 1. Configuration

Create `src/sql_gemma/config.py` to centralize all runtime settings.

```python
from dataclasses import dataclass
import os
import torch

@dataclass(frozen=True)
class Settings:
    sql_model_id: str = "vishnurchityala/sql-gemma3"
    gemini_model_id: str = "gemini-3-flash-preview"
    sql_max_new_tokens: int = 160
    gemini_temperature: float = 0.0
    gemini_thinking_level: str = "minimal"

    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def google_api_key(self) -> str:
        return os.environ["GOOGLE_API_KEY"]
```

Implementation notes:

- Keep all model IDs out of UI code.
- Use deterministic settings for Gemini structured parsing.
- Auto-select device for SQL-Gemma.

### 2. Internal types

Create `src/sql_gemma/types.py`.

```python
from dataclasses import dataclass, field

@dataclass
class TableSchema:
    name: str
    description: str
    columns: list[str]

@dataclass
class ContextSummary:
    overview: str
    dialect: str
    tables: list[TableSchema]
    relationships: list[str]
    rules: list[str] = field(default_factory=list)
    glossary: list[str] = field(default_factory=list)

@dataclass
class GenerationResult:
    sql: str | None
    raw_output: str
    error: str | None = None
    sanitized: bool = False
```

Implementation notes:

- `ContextSummary` is the normalized bridge between Gemini and SQL-Gemma.
- `GenerationResult` keeps both the final SQL and the raw model output for debugging.

### 3. Structured output schema for Gemini

Create `src/sql_gemma/schemas.py`.

```python
from pydantic import BaseModel, Field

class TableSchemaModel(BaseModel):
    name: str = Field(description="Table name")
    description: str = Field(description="One-line business meaning")
    columns: list[str] = Field(description="Allowed columns")

class ContextSummaryModel(BaseModel):
    overview: str
    dialect: str
    tables: list[TableSchemaModel]
    relationships: list[str]
    rules: list[str] = []
    glossary: list[str] = []
```

Convert the structured result into internal dataclasses:

```python
from .types import TableSchema, ContextSummary

def to_context_summary(model: ContextSummaryModel) -> ContextSummary:
    return ContextSummary(
        overview=model.overview,
        dialect=model.dialect,
        tables=[
            TableSchema(
                name=t.name,
                description=t.description,
                columns=t.columns,
            )
            for t in model.tables
        ],
        relationships=model.relationships,
        rules=model.rules,
        glossary=model.glossary,
    )
```

Implementation notes:

- Gemini output must be structured and validated before use.
- Invalid or incomplete structured output should fail visibly.

### 4. Gemini context summarization chain

Create `src/sql_gemma/gemini_context_chain.py`.

Use LangChain with `langchain-google-genai` to summarize uploaded markdown into a structured schema object.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from .config import Settings
from .schemas import ContextSummaryModel

CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You convert uploaded database briefing markdown into a compact structured schema object.

Rules:
- Extract only facts supported by the markdown
- Do not invent tables or columns
- Keep descriptions concise
- Preserve relationship rules and important SQL constraints
- Dialect must be explicit
""".strip(),
        ),
        (
            "human",
            """
Analyze this database context markdown and convert it into the required structured format.

<context_markdown>
{context_markdown}
</context_markdown>
""".strip(),
        ),
    ]
)

def build_context_chain():
    settings = Settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model_id,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
        thinking_level=settings.gemini_thinking_level,
    )
    return CONTEXT_PROMPT | llm.with_structured_output(ContextSummaryModel)
```

Implementation notes:

- Gemini is called once per uploaded file, not once per question.
- Cache the summary in Streamlit using file content hash.
- Keep Gemini responsible only for summarization, not SQL generation.

### 5. SQL prompt builder

Create `src/sql_gemma/sql_prompt_builder.py`.

```python
from .types import ContextSummary

SQL_SYSTEM_PROMPT = """
You are an expert SQL generator.

Return exactly one SQL query and nothing else.
Do not include markdown.
Do not include comments.
Do not include explanations.
Do not return multiple statements.
Use only the provided schema context.
""".strip()

def build_sql_user_prompt(question: str, context: ContextSummary) -> str:
    table_blocks = []
    for table in context.tables:
        table_blocks.append(
            f"Table: {table.name}\n"
            f"Description: {table.description}\n"
            f"Columns: {', '.join(table.columns)}"
        )

    relationships = "\n".join(f"- {r}" for r in context.relationships)
    rules = "\n".join(f"- {r}" for r in context.rules) if context.rules else "- Return one read-only SQL query"

    return f"""
Dialect:
{context.dialect}

Overview:
{context.overview}

Tables:
{chr(10).join(table_blocks)}

Relationships:
{relationships}

Rules:
{rules}

Question:
{question}

Return only the SQL query.
""".strip()
```

Implementation notes:

- SQL-Gemma receives only normalized context.
- Prompt builder should not accept raw markdown.
- Keep SQL generation single-turn.

### 6. Local SQL-Gemma runtime

Create `src/sql_gemma/sql_gemma_runtime.py`.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import Settings

class SQLGemmaRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(settings.sql_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(settings.sql_model_id)
        self.model.to(self.settings.device)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.settings.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.sql_max_new_tokens,
                do_sample=False,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Implementation notes:

- Keep SQL-Gemma on raw `transformers`.
- Do not force it into LangChain in v1.
- Cache the runtime in Streamlit with `st.cache_resource`.

### 7. SQL sanitizer

Create `src/sql_gemma/sanitizer.py`.

```python
import re

def sanitize_sql_output(raw_output: str) -> tuple[str | None, bool, str | None]:
    text = raw_output.strip()

    upper = text.upper()
    select_idx = upper.find("SELECT")
    with_idx = upper.find("WITH")
    starts = [i for i in [select_idx, with_idx] if i >= 0]
    if starts:
        text = text[min(starts):]

    text = text.replace("```sql", "").replace("```", "").strip()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return None, False, "Model returned empty output."

    if text.count(";") > 1:
        return None, False, "Model returned multiple statements."

    first_token = text.split()[0].upper()
    if first_token not in {"SELECT", "WITH"}:
        return None, False, "Model returned a non-read-only query."

    if not text.endswith(";"):
        text += ";"

    return text, text != raw_output.strip(), None
```

Implementation notes:

- This is a deterministic guardrail, not an LLM repair pass.
- Reject bad output instead of guessing semantic fixes.

### 8. Orchestration pipeline

Create `src/sql_gemma/pipeline.py`.

```python
from .gemini_context_chain import build_context_chain
from .schemas import to_context_summary
from .sql_prompt_builder import SQL_SYSTEM_PROMPT, build_sql_user_prompt
from .sql_gemma_runtime import SQLGemmaRuntime
from .sanitizer import sanitize_sql_output
from .types import GenerationResult

def summarize_context(context_markdown: str):
    chain = build_context_chain()
    structured = chain.invoke({"context_markdown": context_markdown})
    return to_context_summary(structured)

def generate_sql(question: str, context_summary, runtime: SQLGemmaRuntime) -> GenerationResult:
    user_prompt = build_sql_user_prompt(question, context_summary)
    raw_output = runtime.generate(SQL_SYSTEM_PROMPT, user_prompt)
    sql, sanitized, error = sanitize_sql_output(raw_output)
    return GenerationResult(
        sql=sql,
        raw_output=raw_output,
        error=error,
        sanitized=sanitized,
    )
```

Implementation notes:

- `summarize_context()` runs per uploaded file.
- `generate_sql()` runs per user question.
- The two functions should stay separate and testable.

### 9. Streamlit application flow

Create `streamlit_app.py`.

```python
import hashlib
import streamlit as st
from src.sql_gemma.config import Settings
from src.sql_gemma.pipeline import summarize_context, generate_sql
from src.sql_gemma.sql_gemma_runtime import SQLGemmaRuntime

@st.cache_resource
def get_sql_runtime():
    return SQLGemmaRuntime(Settings())

def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

st.set_page_config(page_title="SQL Gemma", layout="wide")
st.title("SQL Gemma")
st.caption("Gemini summarizes schema context. SQL-Gemma generates the query.")

uploaded = st.file_uploader("Upload context.md", type=["md"])
question = st.text_area("Ask a SQL question", height=140)

context_text = None
context_summary = None

if uploaded:
    context_text = uploaded.read().decode("utf-8")
    current_hash = file_hash(context_text)

    if st.session_state.get("context_hash") != current_hash:
        with st.spinner("Summarizing schema with Gemini..."):
            context_summary = summarize_context(context_text)
        st.session_state["context_hash"] = current_hash
        st.session_state["context_summary"] = context_summary
    else:
        context_summary = st.session_state["context_summary"]

    st.success(f"Loaded {len(context_summary.tables)} tables | Dialect: {context_summary.dialect}")
    st.write("Tables:", ", ".join(t.name for t in context_summary.tables))

if st.button("Generate SQL", use_container_width=True):
    if not uploaded:
        st.error("Upload a context.md file first.")
    elif not question.strip():
        st.error("Enter a question first.")
    else:
        runtime = get_sql_runtime()
        with st.spinner("Generating SQL..."):
            result = generate_sql(question, st.session_state["context_summary"], runtime)

        if result.error:
            st.error(result.error)
            with st.expander("Raw model output"):
                st.code(result.raw_output)
        else:
            st.code(result.sql, language="sql")
```

Implementation notes:

- Context summary is cached per uploaded file content.
- SQL-Gemma runtime is cached per app session.
- SQL only is shown to the user.

### 10. Optional small agent extension

If a mini-agent is added later, keep it behind a dev flag and do not make it the default path.

Suggested tools:

- `summarize_context_tool`
- `generate_sql_tool`
- `sanitize_sql_tool`

Example tool shells:

```python
from langchain_core.tools import tool

@tool
def summarize_context_tool(context_markdown: str) -> dict:
    """Summarize uploaded schema markdown into structured context."""
    ...

@tool
def generate_sql_tool(question: str, context_json: str) -> str:
    """Generate raw SQL using sql-gemma3."""
    ...

@tool
def sanitize_sql_tool(raw_sql: str) -> dict:
    """Sanitize raw model output into one SQL query."""
    ...
```

Recommended policy:

- summarize context
- generate SQL
- sanitize output
- retry at most once if sanitization fails

Gemini should not directly repair SQL in v1.

## Dependencies

Add these packages to `requirements.txt`:

```text
streamlit
langchain
langchain-core
langchain-google-genai
pydantic
google-genai
```

Keep:

```text
torch
transformers
accelerate
```

## Environment and setup

Required environment variable:

```bash
export GOOGLE_API_KEY=your_key_here
```

Local app run command:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Public interfaces

These are the main interfaces the implementation should preserve:

```python
class TableSchema: ...
class ContextSummary: ...
class GenerationResult: ...

def summarize_context(context_markdown: str): ...
def generate_sql(question: str, context_summary, runtime): ...
def sanitize_sql_output(raw_output: str): ...
```

## Test Plan

### Unit tests

- Gemini structured output converts cleanly into `ContextSummary`
- invalid table data fails validation
- SQL prompt builder includes dialect, tables, relationships, rules, and question
- sanitizer removes markdown fences and prompt echo
- sanitizer rejects multi-statement output
- sanitizer rejects non-read-only SQL

### Integration tests

- `context.md -> Gemini summary -> typed context` works
- repeated questions reuse the cached context summary
- repeated app requests reuse the cached SQL-Gemma runtime
- `generate_sql()` returns a valid `GenerationResult`
- malformed Gemini output surfaces as a readable app error

### Manual acceptance scenarios

- upload a valid two-table schema and ask an aggregation question
- ask multiple questions against the same uploaded context
- upload malformed markdown
- SQL-Gemma returns prose before SQL and sanitizer trims it
- SQL-Gemma returns a write statement and sanitizer rejects it

## Assumptions and defaults

- `gemini-3-flash-preview` is used through LangChain's Google integration
- `vishnurchityala/sql-gemma3` is run locally with `transformers`
- Gemini is used only for context summarization in v1
- SQL-Gemma is used only for SQL generation in v1
- no live database execution is included in v1
- the app is local-first and single-turn
- typed objects like `TableSchema` remain part of the architecture because internal validation is still required

## References

- LangChain Google GenAI integration: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
- Gemini 3 docs: https://ai.google.dev/gemini-api/docs/gemini-3
- Vertex AI Gemini 3 Flash docs: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-flash
