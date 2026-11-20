#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_WINDOW_M=8000 exec bash "${SCRIPT_DIR}/run_step9_5km_dfn_scale_sections.sh"
