"""
spaCy inspection tool - runs a spaCy pipeline on a single document and prints
a human-readable report with document, sentence, and entity summaries.

Usage:
    python inspect_spacy.py <path/to/doc.md>
    python inspect_spacy.py <path/to/doc.md> --model en_core_web_sm

# Diagram omitted - this is a CLI entry point with sequential processing only.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


def _hr(title: str = "") -> None:
    width = 72
    if title:
        pad = width - len(title) - 2
        print(f"\n-- {title} " + "-" * pad)
    else:
        print("-" * width)


def _truncate(text: str, limit: int = 80) -> str:
    text = text.replace("\n", " ")
    return text[:limit] + "..." if len(text) > limit else text


def _load_nlp(model_name: str):
    """Load a spaCy pipeline or raise a clear runtime error.

    The import is intentionally local so this script can still print argument
    errors or be byte-compiled in environments where spaCy is not installed.
    """
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "spaCy is not installed. Install it in the target environment first, "
            "for example: `uv add spacy`."
        ) from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Install it first, "
            f"for example: `uv run python -m spacy download {model_name}`."
        ) from exc


def _iter_non_space_tokens(doc) -> Iterable:
    for token in doc:
        if not token.is_space:
            yield token


def main(path: str, model_name: str) -> int:
    raw = Path(path).read_text(encoding="utf-8")
    nlp = _load_nlp(model_name)
    doc = nlp(raw)

    tokens = list(_iter_non_space_tokens(doc))
    sentences = list(doc.sents)
    ents = list(doc.ents)
    label_counts = Counter(ent.label_ for ent in ents)
    unique_entities = Counter((ent.text, ent.label_) for ent in ents)

    _hr("FILE")
    print(f"  Path     : {path}")
    print(f"  Length   : {len(raw)} chars")

    _hr("SPACY")
    print(f"  Model    : {model_name}")
    print(f"  Pipeline : {', '.join(nlp.pipe_names) if nlp.pipe_names else '(none)'}")
    print(f"  Tokens   : {len(tokens)}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Entities : {len(ents)}")

    if sentences:
        print()
        for sentence in sentences[:5]:
            print(
                f"    [{sentence.start_char}-{sentence.end_char}] "
                f"{_truncate(sentence.text.strip(), 70)!r}"
            )
        if len(sentences) > 5:
            print(f"    ... and {len(sentences) - 5} more")

    _hr("ENTITY LABELS")
    if label_counts:
        for label, count in label_counts.most_common():
            print(f"  {label:8s}: {count}")
    else:
        print("  No entities found.")

    _hr("ENTITY OCCURRENCES")
    if unique_entities:
        for (text, label), count in unique_entities.most_common(40):
            print(f"  [{label}] {text!r}  n={count}")
        if len(unique_entities) > 40:
            print(f"  ... and {len(unique_entities) - 40} more")
    else:
        print("  No entities found.")

    _hr("ENTITY SPANS")
    if ents:
        for ent in ents[:80]:
            print(
                f"  [{ent.start_char}-{ent.end_char}] {ent.label_:8s} "
                f"{_truncate(ent.text, 60)!r}"
            )
        if len(ents) > 80:
            print(f"  ... and {len(ents) - 80} more")
    else:
        print("  No entities found.")

    _hr()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a spaCy model over a manuscript and print inspection output."
    )
    parser.add_argument("path", help="Path to the input Markdown or text file.")
    parser.add_argument(
        "--model",
        default="en_core_web_sm",
        help="spaCy model name to load. Default: en_core_web_sm",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    try:
        raise SystemExit(main(args.path, args.model))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
