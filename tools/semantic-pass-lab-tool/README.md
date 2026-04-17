# Semantic Pass Lab Tool

This is an experiment-only tool for the `semantic-pass-lab` branch.

It takes current deterministic evidence from the example corpus, builds semantic
pass prompt bundles, and can optionally send those bundles to NVIDIA NIM using a
guided JSON schema.

## Purpose

This tool is not production integration.

It exists to answer:

- what evidence packet shape should a later semantic pass consume
- what noisy evidence still survives into a broad or curated bundle
- whether structured JSON output is viable for later semantic consolidation

## Offline Mode

Generate lab packets and prompts only:

```bash
docker-compose run --rm workspace cargo run --manifest-path tools/semantic-pass-lab-tool/Cargo.toml
```

Outputs:

- `logs/semantic-pass-lab/summary.log`
- `logs/semantic-pass-lab/<file>/<shape>/packet.json`
- `logs/semantic-pass-lab/<file>/<shape>/prompt.txt`

## Live NIM Mode

Required environment:

- `NIM_API_KEY`

Optional environment:

- `NIM_RUN_LIVE=1`
- `NIM_MODEL`
- `NIM_BASE_URL`
- `NIM_TIMEOUT_SECS`
- `NIM_MAX_RETRIES`
- `NIM_MAX_TOKENS`

Example:

```bash
export NIM_API_KEY=...
export NIM_RUN_LIVE=1
export NIM_MODEL=moonshotai/kimi-k2-instruct-0905
export NIM_TIMEOUT_SECS=300
export NIM_MAX_RETRIES=3
export NIM_MAX_TOKENS=2000
docker-compose run --rm \
  -e NIM_API_KEY \
  -e NIM_RUN_LIVE \
  -e NIM_MODEL \
  -e NIM_TIMEOUT_SECS \
  -e NIM_MAX_RETRIES \
  -e NIM_MAX_TOKENS \
  -e SEMANTIC_PASS_FILES='1. Radiant Firth.md' \
  -e SEMANTIC_PASS_BUNDLE_SHAPES=curated \
  workspace cargo run --manifest-path tools/semantic-pass-lab-tool/Cargo.toml
```

Default model:

- `moonshotai/kimi-k2-instruct-0905`

Default endpoint:

- `https://integrate.api.nvidia.com/v1/chat/completions`

Default timeout:

- `180` seconds

Default retries:

- `2` retries after the initial attempt

Default max tokens:

- `2000`

The live run writes additional outputs:

- `logs/semantic-pass-lab/<file>/<shape>/response.json`

## Prompt Size Guardrails

The lab keeps the on-disk packet JSON complete for inspection, but the live
prompt is intentionally compressed to avoid provider-side failures on very large
planning and reference files.

Current guardrails:

- cap linked-evidence summaries shown per cluster
- cap structured fields shown per prompt
- cap definitions shown per prompt
- cap section summary seeds shown per prompt
- truncate long lines in the prompt view

## Output Contract

The tool asks the model for structured JSON with:

- `proposed_entities`
- `proposed_relationships`
- `proposed_terminology`
- `rejected_evidence`
- `open_questions`

The request uses NVIDIA's `guided_json` extension instead of plain
`response_format=json_object`.

The lab parser also tolerates two common non-compliant response shapes:

- fenced ```json blocks
- leading explanatory text before the first JSON object
- invisible leading Unicode markers before JSON, such as UTF BOM / zero-width markers

When parsing still fails, the tool now reports:

- the specific `serde_json` error from each parse strategy
- the leading Unicode codepoints in the normalized content
