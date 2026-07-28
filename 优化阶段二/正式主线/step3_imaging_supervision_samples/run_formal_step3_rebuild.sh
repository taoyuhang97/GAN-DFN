#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/build_formal_rebuild_groups.py"
python3 "${SCRIPT_DIR}/validate_formal_rebuild_delivery.py"
