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
        E --> F[initialize_db + create_run]
        F --> G[persist documents + records + entities]
        G --> H[Print summary to stdout]
        B -->|run-rescue| R1[Load records + corpus entities from store]
        R1 --> R2[Build rescue task packets]
        R2 --> R3[Execute LLM packets]
        R3 --> R4[Persist rescue verdicts]
        R4 --> H
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

from backend.nlp.llm_tasks.provider import make_chat_responder, run_task_packets
from backend.nlp.llm_tasks.rescue import build_rescue_task_packets_from_records
from backend.nlp.pipeline import run_corpus_pipeline
from backend.nlp.reconciliation.corpus_entities import reconcile_document_entities
from backend.store import (
    create_run,
    delete_run,
    get_corpus_entities_for_run,
    get_corpus_entity,
    get_document_text,
    get_next_rescue_run_id,
    get_records_for_key,
    get_records_for_run,
    get_rescue_verdicts,
    get_run,
    initialize_db,
    list_runs,
    persist_corpus_entities,
    persist_document_entity_records,
    persist_documents,
    persist_rescue_verdict,
    reconstruct_evidence_context,
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


def _hr(title: str = "") -> None:
    """Print a horizontal rule with an optional section title."""
    width = 72
    if title:
        pad = width - len(title) - 2
        print(f"\n-- {title} " + "-" * max(pad, 2))
    else:
        print("-" * width)


def _resolve_run_id(conn, args) -> int | None:
    """Return the run_id from args, falling back to the latest run.

    Args:
        conn: Active database connection.
        args: Parsed CLI arguments (may have run_id attribute).

    Returns:
        The resolved run_id, or None if no runs exist.
    """
    if getattr(args, "run_id", None) is not None:
        return args.run_id
    runs = list_runs(conn)
    if not runs:
        return None
    return runs[-1]["run_id"]


def _find_absorbing_entity(conn, run_id: int, key: str):
    """Search corpus entities for one whose source_keys contains the given key.

    When a normalized key has been absorbed into a compound canonical entity,
    get_corpus_entity returns None because the canonical_key differs. This
    reverse lookup scans source_keys to find the absorbing entity.

    Args:
        conn: Active database connection.
        run_id: The run to search.
        key: The normalized key to look for in source_keys lists.

    Returns:
        The absorbing CorpusEntity, or None if no match is found.
    """
    for entity in get_corpus_entities_for_run(conn, run_id):
        if key in entity.source_keys and key != entity.canonical_key:
            return entity
    return None


def cmd_delete_run(args: argparse.Namespace) -> None:
    """Execute the delete-run subcommand.

    Verifies the run exists, then cascade-deletes it and all dependent
    rows (document entity records, corpus entities, rescue verdicts).

    Args:
        args: Parsed CLI arguments containing run_id and db.
    """
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database '{db_path}' does not exist.", file=_sys.stderr)
        _sys.exit(1)

    conn = initialize_db(db_path)
    try:
        run = get_run(conn, args.run_id)
        if run is None:
            print(
                f"Error: run {args.run_id} does not exist.", file=_sys.stderr
            )
            _sys.exit(1)

        delete_run(conn, args.run_id)
        print(
            f"Deleted run {args.run_id} "
            f"(label={run['label']!r}, created={run['created_at']})"
        )
    finally:
        conn.close()


def cmd_inspect(args: argparse.Namespace) -> None:
    """Execute the inspect subcommand.

    Displays all document entity records for a normalized key, reconstructs
    evidence context from stored document text, and shows the corpus entity
    if one exists. If the key was absorbed into a compound, the absorbing
    corpus entity is shown instead.

    Args:
        args: Parsed CLI arguments containing key, optional run_id, and db.
    """
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database '{db_path}' does not exist.", file=_sys.stderr)
        _sys.exit(1)

    conn = initialize_db(db_path)
    try:
        run_id = _resolve_run_id(conn, args)
        if run_id is None:
            print("Error: no runs found in the database.", file=_sys.stderr)
            _sys.exit(1)

        run = get_run(conn, run_id)
        if run is None:
            print(f"Error: run {run_id} does not exist.", file=_sys.stderr)
            _sys.exit(1)

        key = args.key
        records = get_records_for_key(conn, run_id, key)

        _hr(f"INSPECT {key!r}  (run {run_id})")
        print(f"  Run      : {run_id} ({run['label'] or 'no label'})")
        print(f"  Created  : {run['created_at']}")
        print(f"  Git      : {run['git_commit']}")
        print(f"  Records  : {len(records)} document record(s) for this key")

        if not records:
            print(f"\n  No document entity records found for {key!r} in run {run_id}.")
        else:
            _print_document_records(conn, records)

        # Corpus entity lookup - try direct match, then absorbed-key search.
        corpus_entity = get_corpus_entity(conn, run_id, key)
        absorbed = False
        if corpus_entity is None:
            corpus_entity = _find_absorbing_entity(conn, run_id, key)
            absorbed = corpus_entity is not None

        if corpus_entity is not None:
            _print_corpus_entity(corpus_entity, absorbed=absorbed, lookup_key=key)
        else:
            _hr("CORPUS ENTITY")
            print(f"  No corpus entity found for {key!r}.")
            print("  (Key may be suppressed or not yet reconciled.)")

        _hr()
    finally:
        conn.close()


def _print_document_records(conn, records) -> None:
    """Print detailed output for each document entity record.

    Args:
        conn: Active database connection (for evidence reconstruction).
        records: List of DocumentEntityRecord objects to display.
    """
    for i, r in enumerate(records):
        _hr(f"DOCUMENT RECORD {i + 1}/{len(records)}")
        print(f"  Key              : {r.identity.normalized_key}")
        print(f"  Document         : {r.identity.document_anchor.path}")
        print(f"  Record ID        : {r.identity.record_id}")
        print(f"  Surface forms    : {r.identity.surface_forms}")
        print(f"  Bucket           : {r.current_state.bucket.value}")
        print(f"  Category         : {r.current_state.winning_category.value}")
        print(f"  Resolved         : {r.current_state.resolved}")

        pt = r.promotion_trace
        print(f"  Confidence       : {pt.confidence_score:.3f}")
        print(f"  Entityhood       : {r.classification_trace.entityhood.score:.3f}"
              f"  (accepted={r.classification_trace.entityhood.accepted})")
        print(f"  Rule tier        : {pt.rule_tier}")
        print(f"  Scene count      : {pt.scene_count}")
        print(f"  Occurrences      : {r.source_evidence.occurrence_count}")
        print(f"  Attribution      : {pt.attribution_count}")
        print(f"  Possessive       : {pt.possessive_count}")
        print(f"  TF-IDF           : {pt.tfidf_score:.3f}")

        if pt.suppression_reason is not None:
            print(f"  Suppression      : {pt.suppression_reason.value}")
            print(f"  Detail           : {pt.bucket_detail}")

        # Evidence windows with reconstructed context.
        windows = r.source_evidence.evidence_windows
        if windows:
            print(f"\n  Evidence windows ({len(windows)}):")
            for w in windows:
                intro = "[INTRO] " if w.is_first_introduction else "        "
                attr = " [ATTR]" if w.has_attribution else ""
                speaker = f" speaker={w.speaker}" if w.speaker else ""
                print(
                    f"    {intro}{w.entity_key!r}{attr}{speaker}"
                    f"  (span {w.anchor.span_ordinal},"
                    f" chars {w.anchor.start_char}-{w.anchor.end_char})"
                )

                ctx = reconstruct_evidence_context(
                    conn,
                    w.anchor.path,
                    w.anchor.start_char,
                    w.anchor.end_char,
                )
                if ctx["context_before"].strip():
                    before = ctx["context_before"].replace("\n", " ").strip()
                    print(f"      before: {before[:80]!r}")
                if ctx["mention"]:
                    print(f"      >>{ctx['mention']!r}<<")
                if ctx["context_after"].strip():
                    after = ctx["context_after"].replace("\n", " ").strip()
                    print(f"      after : {after[:80]!r}")


def _print_corpus_entity(entity, *, absorbed: bool, lookup_key: str) -> None:
    """Print corpus entity details.

    Args:
        entity: The CorpusEntity to display.
        absorbed: Whether the lookup key was absorbed into this entity.
        lookup_key: The key the user originally searched for.
    """
    _hr("CORPUS ENTITY")
    if absorbed:
        print(f"  (key {lookup_key!r} was absorbed into this entity)")
    print(f"  Canonical key    : {entity.canonical_key}")
    print(f"  Category         : {entity.dominant_category.value}")
    print(f"  Confidence       : {entity.aggregate_confidence:.3f}")
    print(f"  Review required  : {entity.review_required}")
    print(f"  Source keys      : {entity.source_keys}")
    print(f"  Surface forms    : {entity.canonical_surface_forms}")
    if entity.absorbed_surface_forms:
        print(f"  Absorbed forms   : {entity.absorbed_surface_forms}")
    if entity.conflicting_categories:
        cats = [c.value for c in entity.conflicting_categories]
        print(f"  Conflicts        : {cats}")
    if entity.reasons:
        print(f"  Reasons          : {entity.reasons}")


def cmd_compare_runs(args: argparse.Namespace) -> None:
    """Execute the compare-runs subcommand.

    Joins document entity records from two runs on (normalized_key,
    document_path) and reports new entities, removed entities, bucket
    changes, category changes, and significant confidence deltas.

    Args:
        args: Parsed CLI arguments containing old_run_id, new_run_id, and db.
    """
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database '{db_path}' does not exist.", file=_sys.stderr)
        _sys.exit(1)

    conn = initialize_db(db_path)
    try:
        old_run = get_run(conn, args.old_run_id)
        new_run = get_run(conn, args.new_run_id)
        if old_run is None:
            print(f"Error: run {args.old_run_id} does not exist.", file=_sys.stderr)
            _sys.exit(1)
        if new_run is None:
            print(f"Error: run {args.new_run_id} does not exist.", file=_sys.stderr)
            _sys.exit(1)

        old_records = get_records_for_run(conn, args.old_run_id)
        new_records = get_records_for_run(conn, args.new_run_id)

        old_by_key = {
            (r.identity.normalized_key, r.identity.document_anchor.path): r
            for r in old_records
        }
        new_by_key = {
            (r.identity.normalized_key, r.identity.document_anchor.path): r
            for r in new_records
        }

        old_keys = set(old_by_key)
        new_keys = set(new_by_key)
        shared_keys = old_keys & new_keys

        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)

        bucket_changes = []
        category_changes = []
        confidence_changes = []

        for k in sorted(shared_keys):
            old_r = old_by_key[k]
            new_r = new_by_key[k]

            old_bucket = old_r.current_state.bucket.value
            new_bucket = new_r.current_state.bucket.value
            if old_bucket != new_bucket:
                bucket_changes.append((k, old_bucket, new_bucket))

            old_cat = old_r.current_state.winning_category.value
            new_cat = new_r.current_state.winning_category.value
            if old_cat != new_cat:
                category_changes.append((k, old_cat, new_cat))

            old_conf = old_r.promotion_trace.confidence_score
            new_conf = new_r.promotion_trace.confidence_score
            delta = new_conf - old_conf
            if abs(delta) > 0.05:
                confidence_changes.append((k, old_conf, new_conf, delta))

        # -- Output --
        _hr(f"COMPARE run {args.old_run_id} -> run {args.new_run_id}")
        print(
            f"  Old: run {old_run['run_id']}"
            f" ({old_run['label'] or 'no label'},"
            f" {old_run['created_at'][:10]},"
            f" {old_run['document_count']} docs)"
        )
        print(
            f"  New: run {new_run['run_id']}"
            f" ({new_run['label'] or 'no label'},"
            f" {new_run['created_at'][:10]},"
            f" {new_run['document_count']} docs)"
        )

        _hr(f"NEW ENTITIES ({len(added)})")
        if added:
            for nk, dp in added:
                r = new_by_key[(nk, dp)]
                print(
                    f"  {nk:24s} {dp:30s} {r.current_state.bucket.value:12s}"
                    f" {r.current_state.winning_category.value:12s}"
                    f" {r.promotion_trace.confidence_score:.3f}"
                )
        else:
            print("  (none)")

        _hr(f"REMOVED ENTITIES ({len(removed)})")
        if removed:
            for nk, dp in removed:
                r = old_by_key[(nk, dp)]
                print(
                    f"  {nk:24s} {dp:30s} {r.current_state.bucket.value:12s}"
                    f" {r.current_state.winning_category.value:12s}"
                    f" {r.promotion_trace.confidence_score:.3f}"
                )
        else:
            print("  (none)")

        _hr(f"BUCKET CHANGES ({len(bucket_changes)})")
        if bucket_changes:
            for (nk, dp), old_b, new_b in bucket_changes:
                print(f"  {nk:24s} {dp:30s} {old_b} -> {new_b}")
        else:
            print("  (none)")

        _hr(f"CATEGORY CHANGES ({len(category_changes)})")
        if category_changes:
            for (nk, dp), old_c, new_c in category_changes:
                print(f"  {nk:24s} {dp:30s} {old_c} -> {new_c}")
        else:
            print("  (none)")

        _hr(f"CONFIDENCE CHANGES ({len(confidence_changes)}, delta > 0.05)")
        if confidence_changes:
            for (nk, dp), old_c, new_c, delta in confidence_changes:
                sign = "+" if delta > 0 else ""
                print(
                    f"  {nk:24s} {dp:30s}"
                    f" {old_c:.3f} -> {new_c:.3f}  ({sign}{delta:.3f})"
                )
        else:
            print("  (none)")

        _hr("SUMMARY")
        print(
            f"  New: {len(added)}"
            f"  Removed: {len(removed)}"
            f"  Bucket changes: {len(bucket_changes)}"
            f"  Category changes: {len(category_changes)}"
            f"  Confidence changes: {len(confidence_changes)}"
        )
        _hr()
    finally:
        conn.close()


def cmd_run_rescue(args: argparse.Namespace) -> None:
    """Execute the run-rescue subcommand.

    Reads entity records and corpus entities from the store, builds rescue
    task packets, executes them against an LLM provider, and persists
    verdicts to the rescue_verdicts table. The extraction run is never
    mutated - rescue is an immutable overlay.

    Args:
        args: Parsed CLI arguments containing run_id, model, db, and
            optional key filter and label.
    """
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database '{db_path}' does not exist.", file=_sys.stderr)
        _sys.exit(1)

    conn = initialize_db(db_path)
    try:
        run_id = _resolve_run_id(conn, args)
        if run_id is None:
            print("Error: no runs found in the database.", file=_sys.stderr)
            _sys.exit(1)

        run = get_run(conn, run_id)
        if run is None:
            print(f"Error: run {run_id} does not exist.", file=_sys.stderr)
            _sys.exit(1)

        print(f"Loading records for run {run_id}...")
        entity_records = get_records_for_run(conn, run_id)
        corpus_entities = get_corpus_entities_for_run(conn, run_id)
        print(f"  {len(entity_records)} document entity record(s)")
        print(f"  {len(corpus_entities)} corpus entity/entities")

        # Collect document texts needed for evidence window construction.
        doc_paths = {r.identity.document_anchor.path for r in entity_records}
        document_texts: dict[str, str] = {}
        for dp in doc_paths:
            text = get_document_text(conn, dp)
            if text is not None:
                document_texts[dp] = text
        print(f"  {len(document_texts)} document text(s) loaded")

        # Build rescue task packets from store data.
        packets, diagnostics = build_rescue_task_packets_from_records(
            entity_records=entity_records,
            canonical_entities=corpus_entities,
            document_texts=document_texts,
        )

        # Apply --key filter if specified.
        key_filter = getattr(args, "key", None)
        if key_filter:
            packets = [p for p in packets if p.source_object_id == key_filter]
            print(f"  Filtered to key {key_filter!r}: {len(packets)} packet(s)")

        if not packets:
            print("\nNo rescue candidates found. Nothing to do.")
            return

        print(f"\n{len(packets)} rescue task packet(s) to execute")

        # Build the responder from environment or flags.
        model = args.model
        api_key = _os.environ.get("LLM_API_KEY", "")
        base_url = _os.environ.get(
            "LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )

        responder = None
        if api_key:
            responder = make_chat_responder(api_key=api_key, base_url=base_url)
            print(f"Using model {model} via {base_url}")
        else:
            print(
                "WARNING: LLM_API_KEY not set. "
                "All packets will be marked as skipped."
            )

        # Allocate a rescue run ID.
        rescue_run_id = get_next_rescue_run_id(conn, run_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        label = getattr(args, "label", None)

        print(f"Rescue run ID: {rescue_run_id}")
        print(f"Executing {len(packets)} packet(s)...")

        # Execute packets against the LLM provider.
        results = run_task_packets(
            packets,
            model=model,
            provider="chat_completions",
            responder=responder,
        )

        # Pair results with packets by task_id to recover the normalized_key
        # (task_id is a content hash, not the entity key).
        packet_by_id = {p.task_id: p for p in packets}
        rescued_count = 0
        suppressed_count = 0
        failed_count = 0
        skipped_count = 0

        from backend.nlp.types import LLMTaskResultStatus

        for result in results:
            if result.status == LLMTaskResultStatus.FAILED:
                failed_count += 1
                continue
            if result.status == LLMTaskResultStatus.SKIPPED:
                skipped_count += 1
                continue

            payload = result.payload if isinstance(result.payload, dict) else {}
            proposal = payload.get("proposal_payload", {})
            if not isinstance(proposal, dict):
                proposal = {}

            is_valid = payload.get("is_valid", False) is True
            is_rescued = proposal.get("rescue", False) is True and is_valid

            if is_valid:
                if is_rescued:
                    rescued_count += 1
                else:
                    suppressed_count += 1

            packet = packet_by_id.get(result.task_id)
            normalized_key = (
                packet.source_object_id if packet else result.task_id
            )

            persist_rescue_verdict(
                conn,
                rescue_run_id=rescue_run_id,
                run_id=run_id,
                normalized_key=normalized_key,
                rescued=is_rescued,
                model=model,
                created_at=timestamp,
                label=label,
                entity_type=proposal.get("type_hint"),
                canonical_name=proposal.get("canonical_name"),
                confidence=proposal.get("confidence"),
                rationale=proposal.get("rationale"),
            )

        _hr("RESCUE SUMMARY")
        print(f"  Run ID         : {run_id}")
        print(f"  Rescue run ID  : {rescue_run_id}")
        print(f"  Model          : {model}")
        print(f"  Packets        : {len(packets)}")
        print(f"  Rescued        : {rescued_count}")
        print(f"  Suppressed     : {suppressed_count}")
        print(f"  Failed         : {failed_count}")
        print(f"  Skipped        : {skipped_count}")
        if label:
            print(f"  Label          : {label}")
        _hr()
    finally:
        conn.close()


def _build_report_text(
    run: dict,
    run_id: int,
    bucket_counts: dict[str, int],
    total_records: int,
    category_counts: list[tuple[str, int]],
    rescue_verdicts: list[dict] | None,
) -> str:
    """Render the report as plain text.

    Args:
        run: Run metadata dict.
        run_id: The run ID.
        bucket_counts: Mapping of bucket name to record count.
        total_records: Total number of document entity records.
        category_counts: List of (category, count) tuples for corpus entities.
        rescue_verdicts: Rescue verdicts if overlay requested, else None.

    Returns:
        Formatted text report string.
    """
    lines: list[str] = []
    width = 72

    def hr(title: str = "") -> None:
        if title:
            pad = width - len(title) - 2
            lines.append(f"\n-- {title} " + "-" * max(pad, 2))
        else:
            lines.append("-" * width)

    hr("RUN METADATA")
    lines.append(f"  Run ID          : {run_id}")
    lines.append(f"  Label           : {run['label'] or '(none)'}")
    lines.append(f"  Created         : {run['created_at']}")
    lines.append(f"  Git commit      : {run['git_commit']}")
    lines.append(f"  Documents       : {run['document_count']}")

    hr("ENTITY RECORDS BY BUCKET")
    lines.append(f"  Total           : {total_records}")
    lines.append(f"  Promoted        : {bucket_counts.get('promoted', 0)}")
    lines.append(f"  Review-only     : {bucket_counts.get('review_only', 0)}")
    lines.append(f"  Suppressed      : {bucket_counts.get('suppressed', 0)}")

    hr("CORPUS ENTITIES BY CATEGORY")
    total_ce = sum(count for _, count in category_counts)
    lines.append(f"  Total           : {total_ce}")
    for cat, count in category_counts:
        lines.append(f"  {cat:18s}: {count}")

    if rescue_verdicts is not None:
        hr("RESCUE OVERLAY")
        rescued = [v for v in rescue_verdicts if v["rescued"]]
        rejected = [v for v in rescue_verdicts if not v["rescued"]]
        lines.append(f"  Rescue run ID   : {rescue_verdicts[0]['rescue_run_id'] if rescue_verdicts else '(none)'}")
        lines.append(f"  Model           : {rescue_verdicts[0]['model'] if rescue_verdicts else '(none)'}")
        lines.append(f"  Verdicts        : {len(rescue_verdicts)}")
        lines.append(f"  Rescued         : {len(rescued)}")
        lines.append(f"  Rejected        : {len(rejected)}")

        if rescued:
            lines.append("")
            lines.append("  Rescued entities:")
            for v in sorted(rescued, key=lambda x: x["normalized_key"]):
                etype = v.get("entity_type") or ""
                cname = v.get("canonical_name") or ""
                conf = v.get("confidence")
                conf_str = f"  conf={conf:.2f}" if conf is not None else ""
                extra = ""
                if etype:
                    extra += f"  type={etype}"
                if cname:
                    extra += f"  name={cname}"
                lines.append(f"    {v['normalized_key']:24s}{conf_str}{extra}")

    hr()
    return "\n".join(lines) + "\n"


def _build_report_json(
    run: dict,
    run_id: int,
    bucket_counts: dict[str, int],
    total_records: int,
    category_counts: list[tuple[str, int]],
    rescue_verdicts: list[dict] | None,
) -> str:
    """Render the report as JSON.

    Args:
        run: Run metadata dict.
        run_id: The run ID.
        bucket_counts: Mapping of bucket name to record count.
        total_records: Total number of document entity records.
        category_counts: List of (category, count) tuples for corpus entities.
        rescue_verdicts: Rescue verdicts if overlay requested, else None.

    Returns:
        JSON string.
    """
    import json

    report: dict = {
        "run_id": run_id,
        "label": run["label"],
        "created_at": run["created_at"],
        "git_commit": run["git_commit"],
        "document_count": run["document_count"],
        "entity_records": {
            "total": total_records,
            "by_bucket": bucket_counts,
        },
        "corpus_entities": {
            "total": sum(count for _, count in category_counts),
            "by_category": {cat: count for cat, count in category_counts},
        },
    }

    if rescue_verdicts is not None:
        rescued = [v for v in rescue_verdicts if v["rescued"]]
        rejected = [v for v in rescue_verdicts if not v["rescued"]]
        report["rescue_overlay"] = {
            "rescue_run_id": rescue_verdicts[0]["rescue_run_id"] if rescue_verdicts else None,
            "model": rescue_verdicts[0]["model"] if rescue_verdicts else None,
            "total_verdicts": len(rescue_verdicts),
            "rescued_count": len(rescued),
            "rejected_count": len(rejected),
            "rescued_entities": [
                {
                    "normalized_key": v["normalized_key"],
                    "entity_type": v.get("entity_type"),
                    "canonical_name": v.get("canonical_name"),
                    "confidence": v.get("confidence"),
                }
                for v in sorted(rescued, key=lambda x: x["normalized_key"])
            ],
        }

    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


def cmd_report(args: argparse.Namespace) -> None:
    """Execute the report subcommand.

    Generates a human-readable or JSON report of extraction state for a
    run, with optional rescue verdict overlay. Output goes to stdout or
    to a file if --output is specified.

    Args:
        args: Parsed CLI arguments containing run_id, db, format, output,
            and optional rescue_run_id.
    """
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database '{db_path}' does not exist.", file=_sys.stderr)
        _sys.exit(1)

    conn = initialize_db(db_path)
    try:
        run_id = _resolve_run_id(conn, args)
        if run_id is None:
            print("Error: no runs found in the database.", file=_sys.stderr)
            _sys.exit(1)

        run = get_run(conn, run_id)
        if run is None:
            print(f"Error: run {run_id} does not exist.", file=_sys.stderr)
            _sys.exit(1)

        # Bucket counts from document entity records.
        bucket_rows = conn.execute(
            "SELECT bucket, COUNT(*) as cnt "
            "FROM document_entity_records WHERE run_id = ? GROUP BY bucket",
            (run_id,),
        ).fetchall()
        bucket_counts = {row[0]: row[1] for row in bucket_rows}
        total_records = sum(bucket_counts.values())

        # Category counts from corpus entities.
        cat_rows = conn.execute(
            "SELECT dominant_category, COUNT(*) as cnt "
            "FROM corpus_entities WHERE run_id = ? "
            "GROUP BY dominant_category ORDER BY dominant_category",
            (run_id,),
        ).fetchall()
        category_counts = [(row[0], row[1]) for row in cat_rows]

        # Optional rescue overlay.
        rescue_verdicts = None
        rescue_run_id = getattr(args, "rescue_run_id", None)
        if rescue_run_id is not None:
            rescue_verdicts = get_rescue_verdicts(
                conn, run_id, rescue_run_id=rescue_run_id
            )
            if not rescue_verdicts:
                print(
                    f"Warning: no rescue verdicts found for "
                    f"rescue_run_id={rescue_run_id} on run {run_id}.",
                    file=_sys.stderr,
                )
                rescue_verdicts = []

        # Render.
        fmt = getattr(args, "format", "text")
        if fmt == "json":
            output = _build_report_json(
                run, run_id, bucket_counts, total_records,
                category_counts, rescue_verdicts,
            )
        else:
            output = _build_report_text(
                run, run_id, bucket_counts, total_records,
                category_counts, rescue_verdicts,
            )

        # Write to file or stdout.
        output_path = getattr(args, "output", None)
        if output_path:
            Path(output_path).write_text(output, encoding="utf-8")
            print(f"Report written to {output_path}")
        else:
            print(output, end="")
    finally:
        conn.close()


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

    # -- delete-run --
    delete_run_parser = subparsers.add_parser(
        "delete-run",
        help="Delete a pipeline run and all dependent data.",
    )
    delete_run_parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="ID of the run to delete.",
    )
    delete_run_parser.add_argument(
        "--db",
        default="data/extraction.db",
        help="Path to the SQLite database (default: data/extraction.db).",
    )

    # -- inspect --
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Show all records and evidence for a normalized entity key.",
    )
    inspect_parser.add_argument(
        "key",
        help="Normalized entity key to look up.",
    )
    inspect_parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run ID to inspect (defaults to latest run).",
    )
    inspect_parser.add_argument(
        "--db",
        default="data/extraction.db",
        help="Path to the SQLite database (default: data/extraction.db).",
    )
    # -- compare-runs --
    compare_runs_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare two pipeline runs side by side.",
    )
    compare_runs_parser.add_argument(
        "old_run_id",
        type=int,
        help="Baseline run ID.",
    )
    compare_runs_parser.add_argument(
        "new_run_id",
        type=int,
        help="Run ID to compare against the baseline.",
    )
    compare_runs_parser.add_argument(
        "--db",
        default="data/extraction.db",
        help="Path to the SQLite database (default: data/extraction.db).",
    )
    # -- run-rescue --
    rescue_parser = subparsers.add_parser(
        "run-rescue",
        help="Run LLM rescue on suppressed entities.",
    )
    rescue_parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Extraction run ID to rescue against (defaults to latest run).",
    )
    rescue_parser.add_argument(
        "--model",
        required=True,
        help="Model identifier for the LLM provider.",
    )
    rescue_parser.add_argument(
        "--db",
        default="data/extraction.db",
        help="Path to the SQLite database (default: data/extraction.db).",
    )
    rescue_parser.add_argument(
        "--key",
        default=None,
        help="Filter to a single normalized entity key.",
    )
    rescue_parser.add_argument(
        "--label",
        default=None,
        help="Optional human-readable label for this rescue run.",
    )
    # -- report --
    report_parser = subparsers.add_parser(
        "report",
        help="Generate a report for a run.",
    )
    report_parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="Run ID to report on (defaults to latest run).",
    )
    report_parser.add_argument(
        "--db",
        default="data/extraction.db",
        help="Path to the SQLite database (default: data/extraction.db).",
    )
    report_parser.add_argument(
        "--rescue-run-id",
        type=int,
        default=None,
        help="Overlay rescue verdicts from this rescue run.",
    )
    report_parser.add_argument(
        "--output",
        default=None,
        help="Write report to file instead of stdout.",
    )
    report_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "run-corpus": cmd_run_corpus,
        "delete-run": cmd_delete_run,
        "inspect": cmd_inspect,
        "compare-runs": cmd_compare_runs,
        "run-rescue": cmd_run_rescue,
        "report": cmd_report,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
