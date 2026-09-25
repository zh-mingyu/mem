#!/usr/bin/env bash

# Run all three TrawMem benchmarks in one shard.  The caller owns one GPU;
# each benchmark is executed sequentially with its own output directory and
# LanceDB path while reusing the same local vLLM environment.

set -eo pipefail
export PYTHONNOUSERSITE=1

PROJECT_DIR="${TRAWMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
ROOT="${DATA_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)}"
BASE_OUTPUT="${TRAWMEM_OUTPUT_ROOT:-${PROJECT_DIR}/outputs/full_qwen38_${SLURM_JOB_ID:-local}_s${TRAWMEM_SHARD_INDEX:-0}}"

LOCOMO_DATASET="${TRAWMEM_LOCOMO_DATASET:-${TRAWMEM_DATASET_PATH:-${ROOT}/baseline/benchmark/LoCoMo/data/locomo10.json}}"
LONGMEMEVAL_DATASET="${TRAWMEM_LONGMEMEVAL_DATASET:-${ROOT}/data/longmemeval/longmemeval_s.json}"
MEMGALLERY_DATASET="${TRAWMEM_MEMGALLERY_DATASET:-${ROOT}/Mem-Gallery-main/benchmark/data/data}"

test -s "${LOCOMO_DATASET}"
test -s "${LONGMEMEVAL_DATASET}"
if [[ -d "${MEMGALLERY_DATASET}" ]]; then
  find "${MEMGALLERY_DATASET}" -type f \( -name '*.json' -o -name '*.jsonl' \) -print -quit | grep -q .
else
  test -s "${MEMGALLERY_DATASET}"
fi

run_one() {
  local benchmark="$1"
  local dataset="$2"
  local out="${BASE_OUTPUT}/${benchmark}"
  mkdir -p "${out}"
  TRAWMEM_BENCHMARK="${benchmark}" \
  TRAWMEM_DATASET_PATH="${dataset}" \
  TRAWMEM_OUTPUT_DIR="${out}" \
  TRAWMEM_PROJECT_DIR="${PROJECT_DIR}" \
  DATA_ROOT="${ROOT}" \
    /bin/bash "${PROJECT_DIR}/run-trawmem-role.sh"
}

run_one locomo "${LOCOMO_DATASET}"
run_one longmemeval "${LONGMEMEVAL_DATASET}"
run_one memgallery "${MEMGALLERY_DATASET}"

touch "${BASE_OUTPUT}/TRAWMEM_FULL_COMPLETED"
