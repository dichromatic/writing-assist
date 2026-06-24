"""Temporary script - dump the extraction store in the same format as the
manuscript corpus report so the database contents can be visually verified
against the known-good pipeline output.

Usage:
    uv run --project backend python dump_db.py [--db data/extraction.db] [--run-id 1]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def _hr(title: str = "") -> None:
    width = 72
    if title:
        pad = width - len(title) - 2
        print(f"\n-- {title} " + "-" * max(pad, 2))
    else:
        print("-" * width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump extraction store contents.")
    parser.add_argument("--db", default="data/extraction.db")
    parser.add_argument("--run-id", type=int, default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Pick the run.
    if args.run_id is not None:
        run = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (args.run_id,)
        ).fetchone()
    else:
        run = conn.execute(
            "SELECT * FROM runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()

    if run is None:
        print("No runs found in the database.")
        return

    run_id = run["run_id"]

    # Counts by bucket.
    bucket_rows = conn.execute(
        "SELECT bucket, COUNT(*) as cnt "
        "FROM document_entity_records WHERE run_id = ? GROUP BY bucket",
        (run_id,),
    ).fetchall()
    bucket_counts = {r["bucket"]: r["cnt"] for r in bucket_rows}

    total_records = sum(bucket_counts.values())

    # Corpus entity count.
    ce_count = conn.execute(
        "SELECT COUNT(*) FROM corpus_entities WHERE run_id = ?", (run_id,)
    ).fetchone()[0]

    # Document list.
    docs = conn.execute("SELECT path FROM documents ORDER BY path").fetchall()

    _hr("CORPUS")
    print(f"  Run ID            : {run_id}")
    print(f"  Label             : {run['label'] or '(none)'}")
    print(f"  Created           : {run['created_at']}")
    print(f"  Git commit        : {run['git_commit']}")
    print(f"  Documents         : {run['document_count']}")
    print(f"  Entity records    : {total_records}")
    print(f"  Canonical entities: {ce_count}")
    print(f"  Promoted records  : {bucket_counts.get('promoted', 0)}")
    print(f"  Review-only       : {bucket_counts.get('review_only', 0)}")
    print(f"  Suppressed        : {bucket_counts.get('suppressed', 0)}")

    _hr("FILES")
    for d in docs:
        print(f"  {d['path']}")

    # Category counts from corpus entities.
    cat_rows = conn.execute(
        "SELECT dominant_category, COUNT(*) as cnt "
        "FROM corpus_entities WHERE run_id = ? GROUP BY dominant_category "
        "ORDER BY dominant_category",
        (run_id,),
    ).fetchall()

    _hr("CATEGORY COUNTS")
    for r in cat_rows:
        print(f"  {r['dominant_category']:20s} count={r['cnt']}")

    # Corpus entities grouped by category.
    entities = conn.execute(
        "SELECT * FROM corpus_entities WHERE run_id = ? "
        "ORDER BY dominant_category, aggregate_confidence DESC",
        (run_id,),
    ).fetchall()

    _hr("CANONICAL ENTITIES BY CATEGORY")
    current_cat = None
    for e in entities:
        cat = e["dominant_category"]
        if cat != current_cat:
            current_cat = cat
            print(f"  [{cat}]")

        key = e["canonical_key"]
        conf = e["aggregate_confidence"]
        docs_n = e["supporting_document_count"]
        review = "REVIEW" if e["review_required"] else "OK"

        source_keys = json.loads(e["source_keys"])
        absorbed = json.loads(e["absorbed_surface_forms"])
        conflicts = json.loads(e["conflicting_categories"])

        extras = ""
        if conflicts:
            extras += f" conflicts={','.join(conflicts)}"
        if len(source_keys) > 1 or (source_keys and source_keys[0] != key):
            extras += f" keys={','.join(sorted(source_keys))}"
        if absorbed:
            extras += f" absorbed_surfaces={','.join(sorted(absorbed))}"

        print(
            f"  {key:24s} {cat:12s} docs={docs_n:<3d} conf={conf:.3f}  "
            f"{review}{extras}"
        )

    # Promoted entities detail.
    promoted = conn.execute(
        "SELECT normalized_key, document_path, confidence_score, "
        "  winning_category, scene_count, occurrence_count, "
        "  entityhood_score, promotion_trace "
        "FROM document_entity_records "
        "WHERE run_id = ? AND bucket = 'promoted' "
        "ORDER BY confidence_score DESC",
        (run_id,),
    ).fetchall()

    _hr("PROMOTED RECORDS")
    for r in promoted:
        pt = json.loads(r["promotion_trace"])
        print(
            f"  {r['normalized_key']:24s} [{r['winning_category']:12s}] "
            f"score={r['confidence_score']:.3f}  tier={pt['rule_tier']}  "
            f"scenes={r['scene_count']}  attr={pt['attribution_count']}  "
            f"poss={pt['possessive_count']}  tfidf={pt['tfidf_score']:.3f}  "
            f"({r['document_path']})"
        )

    # Review-only sample.
    review_only = conn.execute(
        "SELECT normalized_key, document_path, confidence_score, "
        "  winning_category, promotion_trace "
        "FROM document_entity_records "
        "WHERE run_id = ? AND bucket = 'review_only' "
        "ORDER BY confidence_score DESC",
        (run_id,),
    ).fetchall()

    _hr("REVIEW-ONLY RECORDS")
    for r in review_only:
        pt = json.loads(r["promotion_trace"])
        reason = pt.get("bucket_detail", "")
        print(
            f"  {r['normalized_key']:24s} [{r['winning_category']:12s}] "
            f"score={r['confidence_score']:.3f}  reason: {reason[:60]}  "
            f"({r['document_path']})"
        )

    # Suppressed sample.
    suppressed = conn.execute(
        "SELECT normalized_key, document_path, suppression_reason, "
        "  confidence_score, promotion_trace "
        "FROM document_entity_records "
        "WHERE run_id = ? AND bucket = 'suppressed' "
        "ORDER BY confidence_score DESC",
        (run_id,),
    ).fetchall()

    _hr(f"SUPPRESSED RECORDS ({len(suppressed)} total)")
    # Group by suppression reason for summary.
    reason_counts = defaultdict(int)
    for r in suppressed:
        reason_counts[r["suppression_reason"] or "unknown"] += 1

    print("  By reason:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"    {reason:40s} {count}")

    print()
    print("  Top 30 by confidence:")
    for r in suppressed[:30]:
        pt = json.loads(r["promotion_trace"])
        detail = pt.get("bucket_detail", "")
        print(
            f"    {r['normalized_key']:24s} {r['suppression_reason']:35s} "
            f"score={r['confidence_score']:.3f}  {detail[:40]}  "
            f"({r['document_path']})"
        )

    _hr()
    conn.close()


if __name__ == "__main__":
    main()
