"""Deterministic post-processing and safety checks for generated SQL."""

import re


def sanitize_sql_output(raw_output: str) -> tuple[str | None, bool, str | None]:
    """Sanitize model output into a single read-only SQL statement."""
    original = raw_output.strip()
    text = original

    upper = text.upper()
    select_idx = upper.find("SELECT")
    with_idx = upper.find("WITH")
    starts = [idx for idx in (select_idx, with_idx) if idx >= 0]
    if starts:
        text = text[min(starts) :]

    text = text.replace("```sql", "").replace("```", "").strip()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return None, False, "Model returned empty output."

    if ";" in text:
        first_stmt, remainder = text.split(";", 1)
        if remainder.strip():
            return None, False, "Model returned multiple statements."
        text = first_stmt.strip() + ";"
    else:
        text = text.rstrip() + ";"

    match = re.match(r"^\s*([A-Za-z]+)", text)
    first_token = match.group(1).upper() if match else ""
    if first_token not in {"SELECT", "WITH"}:
        return None, False, "Model returned a non-read-only query."

    return text, text != original, None
