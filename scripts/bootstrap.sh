#!/usr/bin/env sh
set -eu

cd frontend && npm install && cd ..
uv sync --project backend --group dev
