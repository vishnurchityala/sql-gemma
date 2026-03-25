"""Pydantic schemas used for Gemini structured output."""

from pydantic import BaseModel, Field

from .types import ContextSummary, TableSchema


class TableSchemaModel(BaseModel):
    name: str = Field(description="Table name")
    description: str = Field(description="One-line business meaning")
    columns: list[str] = Field(description="List of allowed columns")


class ContextSummaryModel(BaseModel):
    overview: str
    dialect: str
    tables: list[TableSchemaModel]
    relationships: list[str]
    rules: list[str] = []
    glossary: list[str] = []


def to_context_summary(model: ContextSummaryModel) -> ContextSummary:
    """Convert validated Pydantic output into internal dataclasses."""
    return ContextSummary(
        overview=model.overview,
        dialect=model.dialect,
        tables=[
            TableSchema(
                name=table.name,
                description=table.description,
                columns=table.columns,
            )
            for table in model.tables
        ],
        relationships=model.relationships,
        rules=model.rules,
        glossary=model.glossary,
    )
