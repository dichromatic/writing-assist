# Diagram omitted - utility module with no significant information flow.

"""
Text filtering helpers for LLM-facing corpus artifacts.

These helpers operate at the semantic handoff boundary rather than inside core
parsing or promotion. That keeps source anchors and raw offsets intact while
still ensuring later LLM-facing artifacts do not inherit decorative emoji
noise from the corpus.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import regex

_EMOJI_RE = regex.compile(
    r"[\p{Extended_Pictographic}\p{Emoji_Presentation}\p{Emoji_Modifier}\u200d\ufe0f]+"
)
_SINGLE_LINE_LEADING_SPACE_RE = regex.compile(r"(?m)^ (?=\S)")
_POST_COLON_SPACE_RE = regex.compile(r"(:) {2,}(?=\S)")


def strip_emoji(text: str) -> str:
    """Remove emoji and tidy spacing for LLM-facing text.

    Args:
        text: Arbitrary text that may contain decorative emoji.

    Returns:
        Text with emoji removed and obvious leftover spacing cleaned up while
        preserving newlines and ordinary punctuation.
    """
    filtered = _EMOJI_RE.sub("", text)
    filtered = _SINGLE_LINE_LEADING_SPACE_RE.sub("", filtered)
    filtered = _POST_COLON_SPACE_RE.sub(r"\1 ", filtered)
    return filtered


def to_llm_safe_jsonable(value: Any) -> Any:
    """Convert nested artifact values into JSON-safe, emoji-filtered data.

    Args:
        value: Arbitrary nested value from an LLM-facing artifact.

    Returns:
        JSON-serializable deep structure with strings sanitized.
    """
    if isinstance(value, str):
        return strip_emoji(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: to_llm_safe_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        def _jsonable_key(raw_key: Any) -> Any:
            """Convert mapping keys into JSON-safe primitive values.

            Enum keys appear in a few artifact-facing traces, and JSON
            serialization requires primitive key types. Converting them here
            keeps the record types ergonomic in Python while preserving a
            stable artifact boundary.
            """
            if isinstance(raw_key, Enum):
                raw_key = raw_key.value
            if isinstance(raw_key, str):
                return strip_emoji(raw_key)
            return raw_key

        return {
            _jsonable_key(key): to_llm_safe_jsonable(inner_value)
            for key, inner_value in value.items()
        }
    if isinstance(value, list):
        return [to_llm_safe_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_llm_safe_jsonable(item) for item in value]
    return value
