"""Streamlit app for the multi-model SQL generation workflow."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.sql_gemma.config import Settings
from src.sql_gemma.pipeline import generate_sql, summarize_context
from src.sql_gemma.sql_gemma_runtime import SQLGemmaRuntime
from src.sql_gemma.sql_prompt_builder import SQL_SYSTEM_PROMPT, build_sql_user_prompt

DIAGRAM_PATH = Path("img/multi-model-sql-agent.png")
SAMPLE_CONTEXT_PATH = Path("context.sample.md")
SQL_MODEL_URL = "https://huggingface.co/vishnurchityala/sql-gemma3"
GEMINI_MODEL_URL = "https://deepmind.google/models/gemini/flash/"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _context_to_dict(summary: Any) -> dict:
    return {
        "overview": summary.overview,
        "dialect": summary.dialect,
        "tables": [
            {
                "name": table.name,
                "description": table.description,
                "columns": table.columns,
            }
            for table in summary.tables
        ],
        "relationships": summary.relationships,
        "rules": summary.rules,
        "glossary": summary.glossary,
    }


def _init_session_state() -> None:
    defaults = {
        "context_hash": None,
        "context_summary": None,
        "context_error": None,
        "context_text": "",
        "summary_seconds": None,
        "last_question": "",
        "last_prompt": "",
        "last_result": None,
        "generation_seconds": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


@st.cache_resource
def get_sql_runtime() -> SQLGemmaRuntime:
    return SQLGemmaRuntime(get_settings())


def main() -> None:
    st.set_page_config(page_title="SQL Gemma Workbench", layout="wide", page_icon=":material/terminal:")
    _init_session_state()
    settings = get_settings()

    st.title("SQL Gemma Workbench")
    with st.container(border=True):
        header_left, header_right = st.columns([2, 1], gap="large")
        with header_left:
            st.markdown("#### Project Summary")
            st.write(
                "SQL Gemma Workbench is a Text-to-SQL system that converts natural-language "
                "questions into executable SQL using a LoRA-adapted fine-tuned model."
            )
            st.markdown("- Context is structured before generation to improve schema grounding.")
            st.markdown("- SQL generation is constrained by read-only, single-query guardrails.")
            st.markdown("- Full transparency is preserved across prompt, raw output, and final SQL.")
            st.caption(
                "Keywords: Text-to-SQL, Large Language Models, Parameter-Efficient Fine-Tuning, LoRA."
            )
        with header_right:
            st.markdown("#### Models Used")
            st.markdown(f"**Context Model**  \n`{settings.gemini_model_id}`")
            st.markdown(f"[Gemini Flash]({GEMINI_MODEL_URL})")
            st.markdown("---")
            st.markdown(f"**SQL Model**  \n`{settings.sql_model_id}`")
            st.markdown(f"[SQL Gemma 3 on Hugging Face]({SQL_MODEL_URL})")

    st.divider()
    st.subheader("Workspace")

    left_col, right_col = st.columns([1.7, 1], gap="large")

    with left_col:
        context_source = st.radio(
            "Context Source",
            options=["Upload context.md", "Use sample context"],
            horizontal=True,
        )
        uploaded = None
        if context_source == "Upload context.md":
            uploaded = st.file_uploader("Upload context.md", type=["md"])
        else:
            st.info(f"Using sample context from `{SAMPLE_CONTEXT_PATH}`")
            if SAMPLE_CONTEXT_PATH.exists():
                with st.expander("Preview sample context"):
                    st.code(SAMPLE_CONTEXT_PATH.read_text(encoding="utf-8"), language="markdown")
        question = st.text_area(
            "Ask a SQL question",
            height=140,
            placeholder="Example: Show top 5 departments by average salary where total payroll stays within budget.",
        )
        generate_clicked = st.button("Generate SQL", type="primary", use_container_width=True)
        inline_output_container = st.container()

    with right_col:
        st.markdown("#### Runtime Snapshot")
        context_loaded = "yes" if st.session_state["context_summary"] is not None else "no"
        summary_latency = (
            f"{st.session_state['summary_seconds']:.2f}s"
            if st.session_state["summary_seconds"] is not None
            else "-"
        )
        st.write(f"- Device: `{settings.device}`")
        st.write(f"- Context loaded: `{context_loaded}`")
        st.write(f"- Last summary latency: `{summary_latency}`")

        st.markdown("#### Model Structure")
        st.write("1. Gemini parses uploaded markdown into structured context.")
        st.write("2. SQL-Gemma generates SQL from normalized context + question.")
        st.write("3. Sanitizer enforces one read-only query and preserves traceability.")

        if DIAGRAM_PATH.exists():
            st.image(str(DIAGRAM_PATH), caption="Model architecture", use_container_width=True)

    context_text: str | None = None
    if context_source == "Upload context.md":
        if uploaded is not None:
            context_text = uploaded.read().decode("utf-8")
    else:
        if SAMPLE_CONTEXT_PATH.exists():
            context_text = SAMPLE_CONTEXT_PATH.read_text(encoding="utf-8")
        else:
            st.error(f"Sample context file not found: `{SAMPLE_CONTEXT_PATH}`")

    if context_text:
        current_hash = _hash_text(context_text)

        if st.session_state["context_hash"] != current_hash:
            with st.spinner("Summarizing context with Gemini..."):
                try:
                    start = time.perf_counter()
                    summary = summarize_context(context_text, settings)
                    elapsed = time.perf_counter() - start
                    st.session_state["context_summary"] = summary
                    st.session_state["context_error"] = None
                    st.session_state["context_hash"] = current_hash
                    st.session_state["context_text"] = context_text
                    st.session_state["summary_seconds"] = elapsed
                except Exception as exc:  # noqa: BLE001
                    st.session_state["context_summary"] = None
                    st.session_state["context_error"] = str(exc)
                    st.session_state["context_hash"] = current_hash
                    st.session_state["context_text"] = context_text
                    st.session_state["summary_seconds"] = None

        if st.session_state["context_error"]:
            st.error(st.session_state["context_error"])
        elif st.session_state["context_summary"]:
            summary = st.session_state["context_summary"]

            st.divider()
            st.subheader("Context Transparency")

            summary_metrics = st.columns(3)
            summary_metrics[0].metric("Tables extracted", len(summary.tables))
            summary_metrics[1].metric("Dialect", summary.dialect)
            summary_metrics[2].metric(
                "Summary latency",
                f"{st.session_state['summary_seconds']:.2f}s"
                if st.session_state["summary_seconds"] is not None
                else "-",
            )

            tab_context, tab_summary, tab_prompt = st.tabs(
                ["Uploaded Context", "Extracted Summary", "Prompt Preview"]
            )
            with tab_context:
                st.code(st.session_state["context_text"], language="markdown")

            with tab_summary:
                summary_dict = _context_to_dict(summary)
                st.json(summary_dict)
                table_rows = [
                    {
                        "table": table["name"],
                        "description": table["description"],
                        "columns": ", ".join(table["columns"]),
                    }
                    for table in summary_dict["tables"]
                ]
                st.dataframe(table_rows, use_container_width=True, hide_index=True)

            with tab_prompt:
                if question.strip():
                    prompt_preview = build_sql_user_prompt(question, summary)
                    st.code(
                        f"[SYSTEM]\n{SQL_SYSTEM_PROMPT}\n\n[USER]\n{prompt_preview}",
                        language="text",
                    )
                else:
                    st.info("Enter a question to preview the final SQL generation prompt.")

    if generate_clicked:
        summary = st.session_state["context_summary"]
        if not context_text:
            st.error("Provide a context file or choose sample context first.")
            return
        if st.session_state["context_error"]:
            st.error("Fix context.md parsing issues before generating SQL.")
            return
        if not summary:
            st.error("Context summary is unavailable.")
            return
        if not question.strip():
            st.error("Enter a question first.")
            return

        st.session_state["last_question"] = question.strip()
        st.session_state["last_prompt"] = build_sql_user_prompt(st.session_state["last_question"], summary)

        with st.spinner("Generating SQL with sql-gemma3..."):
            start = time.perf_counter()
            result = generate_sql(st.session_state["last_question"], summary, get_sql_runtime())
            elapsed = time.perf_counter() - start

        st.session_state["last_result"] = result
        st.session_state["generation_seconds"] = elapsed

    if st.session_state["last_result"] is not None:
        result = st.session_state["last_result"]
        with inline_output_container:
            st.markdown("#### Model Output")
            if result.error:
                st.error(result.error)
            else:
                st.code(result.sql or "", language="sql")

        st.divider()
        st.subheader("Generation Trace")

        trace_metrics = st.columns(2)
        trace_metrics[0].metric(
            "Generation latency",
            f"{st.session_state['generation_seconds']:.2f}s"
            if st.session_state["generation_seconds"] is not None
            else "-",
        )
        trace_metrics[1].metric("Sanitized", "Yes" if result.sanitized else "No")

        with st.expander("Prompt used (system + user)", expanded=False):
            st.code(
                f"[SYSTEM]\n{SQL_SYSTEM_PROMPT}\n\n[USER]\n{st.session_state['last_prompt']}",
                language="text",
            )

        with st.expander("Raw model output", expanded=True):
            st.code(result.raw_output, language="text")

        with st.expander("Run metadata", expanded=False):
            st.code(
                json.dumps(
                    {
                        "context_model": settings.gemini_model_id,
                        "sql_model": settings.sql_model_id,
                        "device": settings.device,
                        "summary_latency_seconds": st.session_state["summary_seconds"],
                        "generation_latency_seconds": st.session_state["generation_seconds"],
                        "sanitized": result.sanitized,
                        "error": result.error,
                    },
                    indent=2,
                ),
                language="json",
            )


if __name__ == "__main__":
    main()
