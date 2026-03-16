#!/usr/bin/env bash
# Print the best experiment across all campaigns and generate winner.md
set -euo pipefail
cd "$(dirname "$0")/.."
scripts/dev_node.sh eval_file scripts/winner.exs
