"""Shared dataclasses for context and generation outputs."""

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
