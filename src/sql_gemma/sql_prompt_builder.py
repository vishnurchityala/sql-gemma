"""Prompt builder for SQL generation using the fine-tuned SQL model."""

from .types import ContextSummary


SQL_SYSTEM_PROMPT = (
    "You are an expert SQL generator.\n\n"
    "Return exactly one SQL query and nothing else.\n"
    "Do not include markdown.\n"
    "Do not include comments.\n"
    "Do not include explanations.\n"
    "Do not return multiple statements.\n"
    "Use only the provided schema context."
)


def build_sql_user_prompt(question: str, context: ContextSummary) -> str:
    """Create a deterministic user prompt from normalized context + question."""
    table_blocks = []
    for table in context.tables:
        table_blocks.append(
            f"Table: {table.name}\n"
            f"Description: {table.description}\n"
            f"Columns: {', '.join(table.columns)}"
        )

    relationships = "\n".join(f"- {rel}" for rel in context.relationships) or "- None provided"
    rules = "\n".join(f"- {rule}" for rule in context.rules) or "- Return one read-only SQL query"

    return (
        f"Dialect:\n{context.dialect}\n\n"
        f"Overview:\n{context.overview}\n\n"
        f"Tables:\n{chr(10).join(table_blocks)}\n\n"
        f"Relationships:\n{relationships}\n\n"
        f"Rules:\n{rules}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Return only the SQL query."
    )
