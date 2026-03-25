"""Optional LangChain tools for future agent-style orchestration."""

import json

from langchain_core.tools import tool

from .config import Settings
from .pipeline import summarize_context
from .sanitizer import sanitize_sql_output
from .schemas import ContextSummaryModel
from .sql_gemma_runtime import SQLGemmaRuntime
from .sql_prompt_builder import SQL_SYSTEM_PROMPT, build_sql_user_prompt
from .types import ContextSummary, TableSchema


def _parse_context_json(context_json: str) -> ContextSummary:
    parsed = ContextSummaryModel.model_validate_json(context_json)
    return ContextSummary(
        overview=parsed.overview,
        dialect=parsed.dialect,
        tables=[
            TableSchema(
                name=table.name,
                description=table.description,
                columns=table.columns,
            )
            for table in parsed.tables
        ],
        relationships=parsed.relationships,
        rules=parsed.rules,
        glossary=parsed.glossary,
    )


@tool
def summarize_context_tool(context_markdown: str) -> str:
    """Summarize uploaded schema markdown into structured JSON."""
    context = summarize_context(context_markdown, Settings())
    payload = {
        "overview": context.overview,
        "dialect": context.dialect,
        "tables": [
            {"name": table.name, "description": table.description, "columns": table.columns}
            for table in context.tables
        ],
        "relationships": context.relationships,
        "rules": context.rules,
        "glossary": context.glossary,
    }
    return json.dumps(payload)


@tool
def generate_sql_tool(question: str, context_json: str) -> str:
    """Generate raw SQL text with sql-gemma3 from question and context JSON."""
    context = _parse_context_json(context_json)
    runtime = SQLGemmaRuntime(Settings())
    prompt = build_sql_user_prompt(question, context)
    return runtime.generate(SQL_SYSTEM_PROMPT, prompt)


@tool
def sanitize_sql_tool(raw_sql: str) -> str:
    """Sanitize raw model output into one SQL query or error payload."""
    sql, sanitized, error = sanitize_sql_output(raw_sql)
    return json.dumps({"sql": sql, "sanitized": sanitized, "error": error})
