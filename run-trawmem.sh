#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env TRAWMEM_PROJECT_DIR="${TRAWMEM_PROJECT_DIR:-${PROJECT_DIR}}" \
  bash "${PROJECT_DIR}/run-trawmem-role.sh" "$@"
