"""Pipeline orchestration for context summarization and SQL generation."""

from .config import Settings
from .gemini_context_chain import build_context_chain
from .sanitizer import sanitize_sql_output
from .schemas import to_context_summary
from .sql_gemma_runtime import SQLGemmaRuntime
from .sql_prompt_builder import SQL_SYSTEM_PROMPT, build_sql_user_prompt
from .types import ContextSummary, GenerationResult


def summarize_context(context_markdown: str, settings: Settings | None = None) -> ContextSummary:
    """Summarize raw markdown context into a normalized context object."""
    cleaned_markdown = context_markdown.strip()
    if not cleaned_markdown:
        raise ValueError("Uploaded context.md is empty.")

    runtime_settings = settings or Settings()
    chain = build_context_chain(runtime_settings)
    structured = chain.invoke({"context_markdown": cleaned_markdown})
    context = to_context_summary(structured)

    if not context.tables:
        raise ValueError("No tables were extracted from context.md.")
    if not context.dialect.strip():
        raise ValueError("Dialect could not be extracted from context.md.")

    return context


def generate_sql(
    question: str,
    context_summary: ContextSummary,
    runtime: SQLGemmaRuntime,
) -> GenerationResult:
    """Generate and sanitize SQL for a natural-language question."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return GenerationResult(
            sql=None,
            raw_output="",
            error="Question is empty.",
            sanitized=False,
        )

    user_prompt = build_sql_user_prompt(cleaned_question, context_summary)
    raw_output = runtime.generate(SQL_SYSTEM_PROMPT, user_prompt)
    sql, sanitized, error = sanitize_sql_output(raw_output)

    return GenerationResult(
        sql=sql,
        raw_output=raw_output,
        error=error,
        sanitized=sanitized,
    )
