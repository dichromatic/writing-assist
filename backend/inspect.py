"""
Pipeline inspection tool - runs the full NLP pipeline on a single document
and prints a human-readable report of the results at each stage.

Usage:
    uv run --project backend python backend/inspect.py <path/to/doc.md>
    uv run --project backend python backend/inspect.py <path/to/doc.txt>

# Diagram omitted - this is a CLI entry point with no significant data flow
# beyond sequentially calling the pipeline stages and printing their output.
"""

from __future__ import annotations

import os as _os
import sys as _sys

# When Python runs this file as `python backend/inspect.py`, it inserts the
# script's directory (backend/) at sys.path[0]. This shadows the stdlib
# `inspect` module - dataclasses.py does `import inspect` and finds this file
# instead, causing a circular import error. Remove the script directory from
# sys.path before any backend imports occur, and ensure the workspace root is
# present so `import backend.*` still resolves correctly.
_script_dir = _os.path.dirname(_os.path.realpath(__file__))
_workspace = _os.path.dirname(_script_dir)
_sys.path = [p for p in _sys.path if p != _script_dir]
if _workspace not in _sys.path:
    _sys.path.insert(0, _workspace)

import sys
from pathlib import Path

from backend.nlp.pipeline import run_document_pipeline
from backend.nlp.lexicon.induction import classify_cluster_categories


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


def main(path: str) -> None:
    raw = Path(path).read_text(encoding="utf-8")
    pipeline = run_document_pipeline(path, raw)
    doc = pipeline.doc
    pre = pipeline.pre
    result = pipeline.bootstrap_result
    attribution_records = pipeline.attribution_records
    bundle = pipeline.promotion_bundle

    _hr("PARSE")
    print(f"  Path     : {doc.path}")
    print(f"  Length   : {len(raw)} chars")
    print(f"  Headings : {len(doc.headings)}")
    print(f"  Paras    : {len(doc.paragraphs)}")
    print(f"  Scenes   : {len(doc.scenes)}  (breaks: {len(doc.scene_breaks)})")
    print(f"  Sections : {len(doc.sections)}")

    if doc.scenes:
        print()
        for scene in doc.scenes:
            spans_label = f"{len(scene.span_ordinals)} spans"
            print(f"    Scene {scene.scene_index}: chars {scene.start_char}-{scene.end_char}  [{spans_label}]")

    _hr("PREPROCESS")
    total_tokens = sum(len(toks) for toks in pre.tokens_by_span.values())
    print(f"  Spans tokenised : {len(pre.tokens_by_span)}")
    print(f"  Tokens          : {total_tokens}")
    print(f"  Sentences       : {len(pre.sentences)}")
    print(f"  Quote spans     : {len(pre.quote_spans)}")

    if pre.quote_spans:
        print()
        for q in pre.quote_spans[:5]:
            print(f"    [{q.start_char}-{q.end_char}] {_truncate(q.inner_text, 60)!r}")
        if len(pre.quote_spans) > 5:
            print(f"    ... and {len(pre.quote_spans) - 5} more")

    _hr("BOOTSTRAP")
    print(f"  Passes run      : {result.passes_run}")
    print(f"  New per pass    : {result.new_entries_per_pass}")
    print(f"  Lexicon entries : {len(result.lexicon)}")
    print(f"  Clusters        : {len(result.clusters)}")
    print(f"  Candidates      : {len(result.candidates)}")

    # Re-classify clusters now that attribution evidence is available.
    # Bootstrap-time classification uses an empty attribution set because
    # attribution runs after bootstrapping. classify_clusters applies the
    # attribution tie-break so that speakers are never mislabelled as places.
    corrected_categories = classify_cluster_categories(result.clusters, pre, attribution_records)

    if result.lexicon:
        print()
        for entry in sorted(result.lexicon, key=lambda e: e.normalized_phrase):
            category = corrected_categories.get(entry.normalized_phrase, entry.category)
            print(f"    [{category.value}]  {entry.normalized_phrase!r}  "
                  f"(pass {entry.induction_pass}, n={entry.occurrence_count},"
                  f" rules={entry.rule_sources})")

    _hr("ATTRIBUTION")
    print(f"  Records : {len(attribution_records)}")
    if attribution_records:
        print()
        for r in attribution_records:
            print(f"    {r.speaker_key!r}  via {r.pattern}"
                  f"  (span {r.quote_anchor.span_ordinal},"
                  f" chars {r.quote_anchor.start_char}-{r.quote_anchor.end_char})")

    _hr("PROMOTION")
    print(f"  Promoted    : {len(bundle.promoted)}")
    print(f"  Review-only : {len(bundle.review_only)}")
    print(f"  Suppressed  : {len(bundle.suppressed)}")
    print(f"  Evidence windows : {len(bundle.evidence_windows)}")

    if bundle.promoted:
        print()
        print("  PROMOTED:")
        for p in sorted(bundle.promoted, key=lambda c: -c.confidence_score):
            sig = p.signals
            print(f"    {p.cluster.normalized_key!r:20s}  score={p.confidence_score:.3f}"
                  f"  tier={sig.rule_tier}  scenes={sig.scene_count}"
                  f"  attr={sig.attribution_count}  poss={sig.possessive_count}"
                  f"  tfidf={sig.tfidf_score:.3f}")

    if bundle.review_only:
        print()
        print("  REVIEW-ONLY:")
        for r in sorted(bundle.review_only, key=lambda c: -c.confidence_score):
            print(f"    {r.cluster.normalized_key!r:20s}  score={r.confidence_score:.3f}"
                  f"  reason: {r.reason[:60]}")

    if bundle.suppressed:
        print()
        print("  SUPPRESSED:")
        for s in bundle.suppressed:
            print(f"    {s.cluster.normalized_key!r:20s}  {s.reason.value}  {s.detail[:50]}")

    # ------------------------------------------------------------------
    # Evidence windows for promoted clusters
    # ------------------------------------------------------------------
    if bundle.promoted:
        _hr("EVIDENCE WINDOWS (promoted only)")
        promoted_keys = {p.cluster.normalized_key for p in bundle.promoted}
        for w in bundle.evidence_windows:
            if w.entity_key not in promoted_keys:
                continue
            intro = "[INTRO] " if w.is_first_introduction else "        "
            attr  = " [ATTR]" if w.has_attribution else ""
            print(f"\n  {intro}{w.entity_key!r}{attr}"
                  f"  (span {w.anchor.span_ordinal},"
                  f" chars {w.anchor.start_char}-{w.anchor.end_char})")
            if w.context_before.strip():
                print(f"    before: {_truncate(w.context_before.strip(), 70)!r}")
            print(f"    >>{raw[w.anchor.start_char:w.anchor.end_char]!r}<<")
            if w.context_after.strip():
                print(f"    after : {_truncate(w.context_after.strip(), 70)!r}")

    _hr()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python backend/inspect.py <path/to/doc.md>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
