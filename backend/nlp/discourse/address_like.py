"""Dialogue address-shape helpers shared by extraction and semantic review.

# Diagram omitted - utility module with no significant information flow.
"""

from __future__ import annotations


def is_word_like_token(token_text: str) -> bool:
    """Return True when token text behaves like lexical content."""
    return any(character.isalpha() for character in token_text)


def find_enclosing_quote_range(
    token,
    quote_ranges: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Return the quote range that fully contains a token span, if any."""
    for start, end in quote_ranges:
        if start <= token.start_char and token.end_char <= end:
            return (start, end)
    return None


def is_address_like_reference(
    sentence,
    token_index: int,
    quote_ranges: list[tuple[int, int]],
) -> bool:
    """Return True when a token behaves like direct address in dialogue.

    This keeps the check intentionally conservative by only marking common
    vocative patterns such as ``Captain, wait.`` and ``Yes, captain.``
    """
    token = sentence.tokens[token_index]
    quote_range = find_enclosing_quote_range(token, quote_ranges)
    if quote_range is None:
        return False

    quote_token_indexes = [
        index
        for index, quote_token in enumerate(sentence.tokens)
        if quote_range[0] <= quote_token.start_char and quote_token.end_char <= quote_range[1]
    ]
    if token_index not in quote_token_indexes:
        return False

    relative_index = quote_token_indexes.index(token_index)
    previous_indexes = quote_token_indexes[:relative_index]
    following_indexes = quote_token_indexes[relative_index + 1:]
    previous_word_indexes = [
        index for index in previous_indexes
        if is_word_like_token(sentence.tokens[index].text)
    ]
    following_word_indexes = [
        index for index in following_indexes
        if is_word_like_token(sentence.tokens[index].text)
    ]
    previous_token_text = sentence.tokens[previous_indexes[-1]].text if previous_indexes else ""
    following_token_text = sentence.tokens[following_indexes[0]].text if following_indexes else ""

    return (
        (not previous_word_indexes and following_token_text in {",", "!", "?"})
        or (previous_token_text == "," and not following_word_indexes)
    )
