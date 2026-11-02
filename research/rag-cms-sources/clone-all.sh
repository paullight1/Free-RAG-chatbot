#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
echo "Cloning all 26 RAG / CMS repositories as submodules. This can be a large download."
git submodule sync --recursive
git submodule update --init --recursive --jobs 4
echo "Done. Code is under research/rag-cms-sources/."
