#!/usr/bin/env bash
# Download snap-research's locomo10.json to the benchmark's data dir.
set -euo pipefail

# Repo root resolved from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST="$REPO_ROOT/benchmarks/datasets/locomo/data/locomo10.json"

mkdir -p "$(dirname "$DEST")"

curl -fsSL -o "$DEST" \
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

# Quick sanity check
python3 -c "import json,sys; d=json.load(open('$DEST')); assert len(d)==10, f'expected 10 convs, got {len(d)}'; print(f'Downloaded {len(d)} conversations')"

echo "Downloaded LoCoMo to $DEST"
