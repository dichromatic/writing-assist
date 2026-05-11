# Diagram omitted - utility module with no significant information flow.

"""
Text filtering helpers for LLM-facing corpus artifacts.

These helpers operate at the semantic handoff boundary rather than inside core
parsing or promotion. That keeps source anchors and raw offsets intact while
still ensuring later LLM-facing artifacts do not inherit decorative emoji
noise from the corpus.
"""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
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


def sanitize_for_llm(value: Any) -> Any:
    """Recursively sanitize nested dataclasses and containers for LLM use.

    Args:
        value: Arbitrary nested value from a handoff artifact.

    Returns:
        Deep copy with all string leaves filtered through `strip_emoji`.
    """
    if isinstance(value, str):
        return strip_emoji(value)
    if isinstance(value, Enum):
        return value
    if is_dataclass(value):
        sanitized_fields = {
            field.name: sanitize_for_llm(getattr(value, field.name))
            for field in fields(value)
        }
        return type(value)(**sanitized_fields)
    if isinstance(value, list):
        return [sanitize_for_llm(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_llm(item) for item in value)
    if isinstance(value, dict):
        return {
            sanitize_for_llm(key) if isinstance(key, str) else key: sanitize_for_llm(inner_value)
            for key, inner_value in value.items()
        }
    return value


def to_llm_safe_jsonable(value: Any) -> Any:
    """Convert nested artifact values into JSON-safe, emoji-filtered data.

    Args:
        value: Arbitrary nested value from an LLM-facing artifact.

    Returns:
        JSON-serializable deep structure with strings sanitized.
    """
    sanitized = sanitize_for_llm(value)

    def _to_jsonable(inner_value: Any) -> Any:
        if isinstance(inner_value, Enum):
            return inner_value.value
        if is_dataclass(inner_value):
            return _to_jsonable(asdict(inner_value))
        if isinstance(inner_value, dict):
            return {key: _to_jsonable(value) for key, value in inner_value.items()}
        if isinstance(inner_value, list):
            return [_to_jsonable(item) for item in inner_value]
        if isinstance(inner_value, tuple):
            return [_to_jsonable(item) for item in inner_value]
        return inner_value

    return _to_jsonable(sanitized)
