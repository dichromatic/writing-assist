"""
CLI entrypoint for the writing-assist extraction pipeline.

Provides subcommands for running the NLP pipeline across a corpus of
Markdown documents, persisting results to a SQLite store, and inspecting
or comparing runs.

Usage:
    uv run --project backend python backend/main.py run-corpus path/to/corpus/
    uv run --project backend python backend/main.py run-corpus path/to/corpus/ --db data/extraction.db --label "baseline run"

.. code-block:: mermaid

    flowchart TD
        A[CLI args] --> B{subcommand}
        B -->|run-corpus| C[Glob .md files from corpus_dir]
        C --> D[run_corpus_pipeline]
        D --> E[reconcile_document_entities]
        E --> F[initialize_db]
        F --> G[create_run]
        G --> H[persist_documents]
        H --> I[persist_document_entity_records]
        I --> J[persist_corpus_entities]
        J --> K[Print summary to stdout]
"""

from __future__ import annotations

import os as _os
import sys as _sys

# When Python runs this file as `python backend/main.py`, it inserts the
# script's directory (backend/) at sys.path[0]. This shadows stdlib modules
# like `stat` and `inspect`. Remove the script directory and ensure the
# workspace root is present so `import backend.*` resolves correctly.
_script_dir = _os.path.dirname(_os.path.realpath(__file__))
_workspace = _os.path.dirname(_script_dir)
_sys.path = [p for p in _sys.path if p != _script_dir]
if _workspace not in _sys.path:
    _sys.path.insert(0, _workspace)

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from backend.nlp.pipeline import run_corpus_pipeline
from backend.nlp.reconciliation.corpus_entities import reconcile_document_entities
from backend.store import (
    create_run,
    initialize_db,
    persist_corpus_entities,
    persist_document_entity_records,
    persist_documents,
)


def _get_git_commit() -> str:
    """Return the short git commit hash for the current working tree.

    Falls back to 'unknown' if git is unavailable or the working directory
    is not inside a git repository.

    Returns:
        Short hex commit string, or 'unknown'.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def cmd_run_corpus(args: argparse.Namespace) -> None:
    """Execute the run-corpus subcommand.

    Globs the corpus directory for Markdown files, runs the NLP pipeline
    and corpus reconciliation, then persists everything to the SQLite store.

    Args:
        args: Parsed CLI arguments containing corpus_dir, db, and label.
    """
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_dir():
        print(f"Error: '{corpus_dir}' is not a directory.", file=_sys.stderr)
        _sys.exit(1)

    md_paths = sorted(corpus_dir.glob("*.md"))
    if not md_paths:
        print(f"Error: no .md files found in '{corpus_dir}'.", file=_sys.stderr)
        _sys.exit(1)

    path_strings = [str(p) for p in md_paths]
    print(f"Found {len(path_strings)} document(s) in {corpus_dir}")

    # -- NLP pipeline --
    print("Running corpus pipeline...")
    corpus_result = run_corpus_pipeline(path_strings)
    print(f"  {len(corpus_result.entity_records)} document entity record(s)")

    # -- Corpus reconciliation --
    print("Reconciling corpus entities...")
    reconciliation_result = reconcile_document_entities(corpus_result.entity_records)
    print(f"  {len(reconciliation_result.canonical_entities)} corpus entity/entities")

    # -- Persistence --
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    git_commit = _get_git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = initialize_db(db_path)
    try:
        run_id = create_run(
            conn, timestamp, git_commit, len(path_strings), label=args.label
        )

        # Build a path -> raw_text mapping for document persistence.
        documents_dict = {
            str(p): p.read_text(encoding="utf-8") for p in md_paths
        }
        persist_documents(conn, documents_dict)
        persist_document_entity_records(conn, run_id, corpus_result.entity_records)
        persist_corpus_entities(conn, run_id, reconciliation_result.canonical_entities)

        print(f"\nRun {run_id} saved to {db_path}")
        print(f"  Documents: {len(path_strings)}")
        print(f"  Document entity records: {len(corpus_result.entity_records)}")
        print(f"  Corpus entities: {len(reconciliation_result.canonical_entities)}")
        print(f"  Git commit: {git_commit}")
        print(f"  Timestamp: {timestamp}")
    finally:
        conn.close()


def _stub_command(name: str) -> None:
    """Print a not-yet-implemented message for a subcommand stub.

    Args:
        name: The subcommand name to display in the message.
    """
    print(f"'{name}' is not yet implemented.")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands.

    Returns:
        Configured argparse.ArgumentParser with subcommand parsers attached.
    """
    parser = argparse.ArgumentParser(
        prog="writing-assist",
        description="CLI for the writing-assist NLP extraction pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run-corpus --
    run_corpus_parser = subparsers.add_parser(
        "run-corpus",
        help="Run the NLP pipeline on a corpus directory and persist results.",
    )
    run_corpus_parser.add_argument(
        "corpus_dir",
        help="Directory containing .md files to process.",
    )
    run_corpus_parser.add_argument(
        "--db",
        default="data/extraction.db",
        help="Path to the SQLite database (default: data/extraction.db).",
    )
    run_corpus_parser.add_argument(
        "--label",
        default=None,
        help="Optional human-readable label for this run.",
    )

    # -- Stubs for future subcommands --
    subparsers.add_parser("delete-run", help="Delete a pipeline run.")
    subparsers.add_parser("inspect", help="Inspect a pipeline run.")
    subparsers.add_parser("compare-runs", help="Compare two pipeline runs.")
    subparsers.add_parser("run-rescue", help="Run LLM rescue on suppressed entities.")
    subparsers.add_parser("report", help="Generate a report for a run.")

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "run-corpus": cmd_run_corpus,
        "delete-run": lambda _: _stub_command("delete-run"),
        "inspect": lambda _: _stub_command("inspect"),
        "compare-runs": lambda _: _stub_command("compare-runs"),
        "run-rescue": lambda _: _stub_command("run-rescue"),
        "report": lambda _: _stub_command("report"),
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
