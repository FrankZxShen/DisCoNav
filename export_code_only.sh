#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$ROOT_DIR/_code_export}"

mkdir -p "$OUT_DIR"

TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

find "$ROOT_DIR" \
  \( -path "$OUT_DIR" \
  -o -path "$ROOT_DIR/.git" \
  -o -path "$ROOT_DIR/__pycache__" \
  -o -path "$ROOT_DIR/data" \
  -o -path "$ROOT_DIR/datasets" \
  -o -path "$ROOT_DIR/runs" \
  -o -path "$ROOT_DIR/outputs" \
  -o -path "$ROOT_DIR/results" \
  -o -path "$ROOT_DIR/logs" \
  -o -path "$ROOT_DIR/checkpoints" \
  -o -path "$ROOT_DIR/weights" \
  -o -path "$ROOT_DIR/models" \
  -o -path "$ROOT_DIR/model" \) -prune -o \
  -type f \( \
  -name "*.py" -o -name "*.pyi" -o -name "*.sh" -o -name "*.bash" -o -name "*.zsh" \
  -o -name "*.c" -o -name "*.cc" -o -name "*.cpp" -o -name "*.cxx" -o -name "*.cu" \
  -o -name "*.h" -o -name "*.hh" -o -name "*.hpp" -o -name "*.hxx" -o -name "*.inl" \
  -o -name "*.cmake" -o -name "CMakeLists.txt" -o -name "Makefile" -o -name "*.mk" \
  -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.toml" -o -name "*.ini" -o -name "*.cfg" \
  -o -name "*.txt" -o -name "*.md" -o -name "*.rst" \) \
  -printf '%P\0' > "$TMP_LIST"

rsync -a --files-from="$TMP_LIST" --from0 "$ROOT_DIR"/ "$OUT_DIR"/
