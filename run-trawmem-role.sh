#!/usr/bin/env bash

# TrawMem evaluation. ``TRAWMEM_BENCHMARK`` selects one of
# locomo, longmemeval, or memgallery; the full three-benchmark worker invokes
# this script once per benchmark so every run has an isolated memory bank.
# ``TRAWMEM_LLM_BACKEND=remote`` skips local vLLM and uses an OpenAI-compatible
# relay while retaining the local embedding model.

#SBATCH --job-name=TrawMem
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/TrawMem-%j.out
#SBATCH --error=logs/TrawMem-%j.err

set -eo pipefail
export PYTHONNOUSERSITE=1
RUN_STARTED_AT=$(date +%s)

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

if [[ -n "${TRAWMEM_PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="${TRAWMEM_PROJECT_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run-trawmem.sh" ]]; then
  PROJECT_DIR="${SLURM_SUBMIT_DIR}"
else
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
DEFAULT_DATA_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
WORKSPACE_ROOT="${TRAWMEM_WORKSPACE_ROOT:-${PROJECT_DIR}}"
if [[ -n "${TRAWMEM_PYTHON:-}" ]]; then
  PYTHON="${TRAWMEM_PYTHON}"
else
  PYTHON="$(command -v python3 || command -v python || true)"
fi
test -n "${PYTHON}" || { echo "Python 3 is required; set TRAWMEM_PYTHON." >&2; exit 2; }
LLM_BACKEND="${TRAWMEM_LLM_BACKEND:-local}"
VLLM_PYTHON="${TRAWMEM_VLLM_PYTHON:-${PYTHON}}"
MODEL_PATH="${TRAWMEM_MODEL_PATH:-}"
EMBEDDING_PATH="${TRAWMEM_EMBEDDING_PATH:-}"
if [[ "${LLM_BACKEND}" == "remote" ]]; then
  TOKENIZER_PATH="${TRAWMEM_TOKENIZER_MODEL_PATH:-}"
else
  TOKENIZER_PATH="${TRAWMEM_TOKENIZER_MODEL_PATH:-${MODEL_PATH}}"
fi
TOKENIZER_ENCODING="${TRAWMEM_TOKENIZER_ENCODING:-}"
BENCHMARK="${TRAWMEM_BENCHMARK:-locomo}"
DATASET="${TRAWMEM_DATASET_PATH:-}"
MODEL_NAME="${TRAWMEM_LLM_MODEL:-qwen3.8-27b}"
API_KEY="${TRAWMEM_API_KEY:-trawmem-local}"
JOB_ID="${SLURM_JOB_ID:-0}"
PORT="${TRAWMEM_PORT:-$((28000 + JOB_ID % 1000))}"
WORKSPACE_ROLE_SCOPE="${TRAWMEM_WORKSPACE_ROLE_SCOPE:-full}"
JUDGE_ENABLED_RAW="${TRAWMEM_LLM_JUDGE:-1}"
case "${BENCHMARK}" in
  locomo|longmemeval|memgallery) ;;
  *) echo "Invalid TRAWMEM_BENCHMARK: ${BENCHMARK}" >&2; exit 2 ;;
esac
case "${WORKSPACE_ROLE_SCOPE}" in
  full|selection-only) ;;
  *) echo "Invalid TRAWMEM_WORKSPACE_ROLE_SCOPE: ${WORKSPACE_ROLE_SCOPE}" >&2; exit 2 ;;
esac
case "${LLM_BACKEND}" in
  local|remote) ;;
  *) echo "Invalid TRAWMEM_LLM_BACKEND: ${LLM_BACKEND}" >&2; exit 2 ;;
esac
if [[ "${LLM_BACKEND}" == "remote" ]]; then
  REMOTE_BASE_URL="${TRAWMEM_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}"
  REMOTE_API_KEY="${TRAWMEM_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}"
  test -n "${REMOTE_BASE_URL}" || { echo "Remote backend requires TRAWMEM_OPENAI_BASE_URL." >&2; exit 2; }
  test -n "${REMOTE_API_KEY}" || { echo "Remote backend requires TRAWMEM_OPENAI_API_KEY." >&2; exit 2; }
  export TRAWMEM_OPENAI_BASE_URL="${REMOTE_BASE_URL}"
  export TRAWMEM_OPENAI_API_KEY="${REMOTE_API_KEY}"
else
  REMOTE_BASE_URL=""
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  DEFAULT_RUN_DIR="${PROJECT_DIR}/outputs/trawmem_${WORKSPACE_ROLE_SCOPE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  DEFAULT_RUN_DIR="${PROJECT_DIR}/outputs/trawmem_${WORKSPACE_ROLE_SCOPE}_${SLURM_JOB_ID}"
fi
RUN_DIR="${TRAWMEM_OUTPUT_DIR:-${DEFAULT_RUN_DIR}}"
CONFIG_DIR="${RUN_DIR}/config"
CACHE_ROOT="${TRAWMEM_CACHE_DIR:-${WORKSPACE_ROOT}/cache}"
TMP_ROOT="${TRAWMEM_TMP_DIR:-${WORKSPACE_ROOT}/tmp}"

if [[ "${PYTHON}" == */* ]]; then
  test -x "${PYTHON}" || { echo "Python executable is not runnable: ${PYTHON}" >&2; exit 2; }
fi
if [[ "${LLM_BACKEND}" == "local" ]]; then
  if [[ "${VLLM_PYTHON}" == */* ]]; then
    test -x "${VLLM_PYTHON}" || { echo "VLLM Python executable is not runnable: ${VLLM_PYTHON}" >&2; exit 2; }
  fi
  test -d "${MODEL_PATH}"
fi
test -d "${EMBEDDING_PATH}"
if [[ -d "${DATASET}" ]]; then
  find "${DATASET}" -type f \( -name '*.json' -o -name '*.jsonl' \) -print -quit | grep -q .
else
  test -s "${DATASET}"
fi
mkdir -p "${RUN_DIR}" "${CONFIG_DIR}" "${PROJECT_DIR}/logs" \
  "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" \
  "${CACHE_ROOT}/xdg" "${TMP_ROOT}"

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${CONFIG_DIR}:${PROJECT_DIR}"
export NLTK_DATA="${DATA_ROOT}/nltk_data"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
export TRAWMEM_TIMING=1

BUILD_WORKERS="${TRAWMEM_BUILD_WORKERS:-4}"
TEST_WORKERS="${TRAWMEM_TEST_WORKERS:-8}"
WORKSPACE_ENABLED_RAW="${TRAWMEM_ENABLE_MEMORY_WORKSPACE:-True}"
case "$(lowercase "${WORKSPACE_ENABLED_RAW}")" in
  1|true|yes|on) ENABLE_MEMORY_WORKSPACE="True" ;;
  *) ENABLE_MEMORY_WORKSPACE="False" ;;
esac
WORKSPACE_MAX_EVIDENCE="${TRAWMEM_WORKSPACE_MAX_EVIDENCE:-16}"
WORKSPACE_MIN_EVIDENCE="${TRAWMEM_WORKSPACE_MIN_EVIDENCE:-1}"
WORKSPACE_MAX_RELATIONS="${TRAWMEM_WORKSPACE_MAX_RELATIONS:-12}"
WORKSPACE_ACTIVATION_MASS="${TRAWMEM_WORKSPACE_ACTIVATION_MASS:-0.70}"
THREAD_TURN_LIMIT="${TRAWMEM_THREAD_TURN_LIMIT:-10}"
THREAD_ROUTE_TOP_K="${TRAWMEM_THREAD_ROUTE_TOP_K:-16}"
THREAD_ROUTE_PROBE_TOP_K="${TRAWMEM_THREAD_ROUTE_PROBE_TOP_K:-12}"
THREAD_ROUTE_PROBE_MARGIN="${TRAWMEM_THREAD_ROUTE_PROBE_MARGIN:-0.02}"
THREAD_MAX_EXPANDED="${TRAWMEM_THREAD_MAX_EXPANDED:-32}"
THREAD_ADAPTIVE_MAX_EXPANDED="${TRAWMEM_THREAD_ADAPTIVE_MAX_EXPANDED:-24}"
WORKSPACE_QUERY_SELECTION_WEIGHT="${TRAWMEM_WORKSPACE_QUERY_SELECTION_WEIGHT:-0.52}"
WORKSPACE_NODE_SEED_WEIGHT="${TRAWMEM_WORKSPACE_NODE_SEED_WEIGHT:-0.80}"
WORKSPACE_THREAD_DIVERSITY_WEIGHT="${TRAWMEM_WORKSPACE_THREAD_DIVERSITY_WEIGHT:-0.05}"
WORKSPACE_FLOW_TEMPERATURE="${TRAWMEM_WORKSPACE_FLOW_TEMPERATURE:-0.35}"
WORKSPACE_FLOW_MISMATCH_WEIGHT="${TRAWMEM_WORKSPACE_FLOW_MISMATCH_WEIGHT:-1.25}"
WORKSPACE_ROLE_AWARE_RAW="${TRAWMEM_WORKSPACE_ROLE_AWARE:-True}"
case "$(lowercase "${WORKSPACE_ROLE_AWARE_RAW}")" in
  1|true|yes|on) WORKSPACE_ROLE_AWARE="True" ;;
  *) WORKSPACE_ROLE_AWARE="False" ;;
esac
WORKSPACE_ROLE_TRANSPORT_WEIGHT="${TRAWMEM_WORKSPACE_ROLE_TRANSPORT_WEIGHT:-0.22}"
WORKSPACE_ROLE_SELECTION_WEIGHT="${TRAWMEM_WORKSPACE_ROLE_SELECTION_WEIGHT:-0.10}"
WORKSPACE_ROLE_TEMPERATURE="${TRAWMEM_WORKSPACE_ROLE_TEMPERATURE:-0.65}"
SHARD_INDEX="${TRAWMEM_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
NUM_SHARDS="${TRAWMEM_NUM_SHARDS:-1}"
TEST_ARGS=(--shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}")
if [[ "${BENCHMARK}" == "locomo" ]]; then
  TEST_ARGS+=(--parallel-questions --test-workers "${TEST_WORKERS}")
fi
case "$(lowercase "${JUDGE_ENABLED_RAW}")" in
  1|true|yes|on) TEST_ARGS+=(--llm-judge) ;;
esac
if [[ -n "${TRAWMEM_NUM_SAMPLES:-}" ]]; then
  TEST_ARGS+=(--num-samples "${TRAWMEM_NUM_SAMPLES}")
fi

cat > "${CONFIG_DIR}/config.py" <<PY
import os

# Remote runs read the relay token from the process environment so it is not
# written into the persistent per-run config file.
LLM_BACKEND = "${LLM_BACKEND}"
if LLM_BACKEND == "remote":
    OPENAI_API_KEY = os.environ["TRAWMEM_OPENAI_API_KEY"]
    OPENAI_BASE_URL = os.environ["TRAWMEM_OPENAI_BASE_URL"]
else:
    OPENAI_API_KEY = "${API_KEY}"
    OPENAI_BASE_URL = "http://127.0.0.1:${PORT}/v1"
LLM_MODEL = "${MODEL_NAME}"
TOKENIZER_MODEL_PATH = os.environ.get("TRAWMEM_TOKENIZER_MODEL_PATH", "${TOKENIZER_PATH}") or None
TOKENIZER_ENCODING = os.environ.get("TRAWMEM_TOKENIZER_ENCODING", "${TOKENIZER_ENCODING}") or None
EMBEDDING_MODEL = "${EMBEDDING_PATH}"
EMBEDDING_DIMENSION = 1024
ENABLE_THINKING = False
USE_STREAMING = True
USE_JSON_FORMAT = True
WINDOW_SIZE = 20
THREAD_TURN_LIMIT = ${THREAD_TURN_LIMIT}
ENABLE_PARALLEL_PROCESSING = True
MAX_PARALLEL_WORKERS = ${BUILD_WORKERS}
ENABLE_PARALLEL_RETRIEVAL = True
MAX_RETRIEVAL_WORKERS = 8
ENABLE_THREAD_ADDRESS_PLANNER = False
THREAD_ROUTE_TOP_K = ${THREAD_ROUTE_TOP_K}
THREAD_ROUTE_PROBE_TOP_K = ${THREAD_ROUTE_PROBE_TOP_K}
THREAD_ROUTE_PROBE_MARGIN = ${THREAD_ROUTE_PROBE_MARGIN}
THREAD_MAX_HOPS = 3
THREAD_MAX_EXPANDED = ${THREAD_MAX_EXPANDED}
THREAD_ADAPTIVE_MAX_EXPANDED = ${THREAD_ADAPTIVE_MAX_EXPANDED}
THREAD_GRAPH_NEIGHBORS = 5
THREAD_GRAPH_MIN_SIM = 0.25
ENABLE_MEMORY_WORKSPACE = ${ENABLE_MEMORY_WORKSPACE}
WORKSPACE_MAX_EVIDENCE = ${WORKSPACE_MAX_EVIDENCE}
WORKSPACE_MIN_EVIDENCE = ${WORKSPACE_MIN_EVIDENCE}
WORKSPACE_MAX_RELATIONS = ${WORKSPACE_MAX_RELATIONS}
WORKSPACE_ACTIVATION_MASS = ${WORKSPACE_ACTIVATION_MASS}
WORKSPACE_QUERY_SELECTION_WEIGHT = ${WORKSPACE_QUERY_SELECTION_WEIGHT}
WORKSPACE_NODE_SEED_WEIGHT = ${WORKSPACE_NODE_SEED_WEIGHT}
WORKSPACE_THREAD_DIVERSITY_WEIGHT = ${WORKSPACE_THREAD_DIVERSITY_WEIGHT}
WORKSPACE_FLOW_TEMPERATURE = ${WORKSPACE_FLOW_TEMPERATURE}
WORKSPACE_FLOW_MISMATCH_WEIGHT = ${WORKSPACE_FLOW_MISMATCH_WEIGHT}
WORKSPACE_ROLE_AWARE = ${WORKSPACE_ROLE_AWARE}
WORKSPACE_ROLE_SCOPE = "${WORKSPACE_ROLE_SCOPE}"
WORKSPACE_ROLE_TRANSPORT_WEIGHT = ${WORKSPACE_ROLE_TRANSPORT_WEIGHT}
WORKSPACE_ROLE_SELECTION_WEIGHT = ${WORKSPACE_ROLE_SELECTION_WEIGHT}
WORKSPACE_ROLE_TEMPERATURE = ${WORKSPACE_ROLE_TEMPERATURE}
# Keep every shard's thread index isolated from other jobs.
THREAD_LANCEDB_PATH = "${RUN_DIR}/thread_lancedb_data"
THREAD_TABLE_NAME = "event_threads_v3"
JUDGE_API_KEY = OPENAI_API_KEY
JUDGE_BASE_URL = OPENAI_BASE_URL
JUDGE_MODEL = "${TRAWMEM_JUDGE_MODEL:-${MODEL_NAME}}"
JUDGE_TEMPERATURE = ${TRAWMEM_JUDGE_TEMPERATURE:-0.0}
JUDGE_MAX_TOKENS = ${TRAWMEM_JUDGE_MAX_TOKENS:-256}
TOKEN_USAGE_LOG_PATH = "${RUN_DIR}/token_usage.jsonl"
TOKEN_USAGE_SUMMARY_PATH = "${RUN_DIR}/token_usage_summary.json"
ROOT_PROGRESS_PATH = "${RUN_DIR}/root_progress.jsonl"
PY

if [[ "${LLM_BACKEND}" == "remote" ]]; then
  "${PYTHON}" - "${REMOTE_BASE_URL%/}" "${MODEL_NAME}" <<'PY'
import json
import os
import sys
import urllib.request

base_url, model_name = sys.argv[1], sys.argv[2]
keys = list(dict.fromkeys(
    key.strip()
    for key in os.environ["TRAWMEM_OPENAI_API_KEY"].split(",")
    if key.strip()
))
if not keys:
    raise SystemExit("No API keys were provided")

failures = []
for slot, key in enumerate(keys, 1):
    try:
        request = urllib.request.Request(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = {str(item.get("id", "")) for item in payload.get("data", [])}
        if model_name not in models:
            failures.append(f"slot {slot}: requested model unavailable")
            continue
        print(
            f"Remote relay ready: model={model_name}, "
            f"keys={len(keys)}, verified_slot={slot}"
        )
        break
    except Exception as exc:
        reason = getattr(exc, "reason", None)
        detail = str(reason or exc).replace("\n", " ")
        failures.append(f"slot {slot}: {type(exc).__name__}: {detail}")
else:
    raise SystemExit(
        f"No supplied API key can access the requested model; parsed_keys={len(keys)} ("
        + "; ".join(failures)
        + ")"
    )
PY
  if [[ -n "${TOKENIZER_ENCODING}" && -z "${TOKENIZER_PATH}" ]]; then
    "${PYTHON}" - "${TOKENIZER_ENCODING}" <<'PY'
import sys
import tiktoken

name = sys.argv[1]
tiktoken.get_encoding(name)
print(f"Context tokenizer ready: {name}")
PY
  fi
else
PYTHONPATH="" "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}" \
  --dtype bfloat16 \
  --gpu-memory-utilization "${TRAWMEM_GPU_MEMORY_UTILIZATION:-0.88}" \
  --max-model-len "${TRAWMEM_MAX_MODEL_LEN:-32768}" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --enforce-eager \
  --disable-custom-all-reduce \
  --no-enable-log-requests \
  > "${RUN_DIR}/vllm_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 120); do
  if "${PYTHON}" -c "import urllib.request; request=urllib.request.Request('http://127.0.0.1:${PORT}/v1/models', headers={'Authorization': 'Bearer ${API_KEY}'}); urllib.request.urlopen(request, timeout=2).read()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 160 "${RUN_DIR}/vllm_server.log"
    exit 1
  fi
  sleep 5
done
test "${READY}" -eq 1
fi

if [[ "${BENCHMARK}" == "locomo" ]]; then
  TEST_SCRIPT="${PROJECT_DIR}/test_locomo10.py"
  test -f "${TEST_SCRIPT}" || {
    echo "Missing LoCoMo evaluator: ${TEST_SCRIPT}" >&2
    exit 2
  }
  PYTHONUNBUFFERED=1 "${PYTHON}" "${TEST_SCRIPT}" \
    --dataset "${DATASET}" \
    "${TEST_ARGS[@]}" \
    --result-file "${RUN_DIR}/results.json"
else
  TEST_SCRIPT="${PROJECT_DIR}/test_long_benchmarks.py"
  test -f "${TEST_SCRIPT}" || {
    echo "Missing long-benchmark evaluator: ${TEST_SCRIPT}" >&2
    exit 2
  }
  PYTHONUNBUFFERED=1 "${PYTHON}" "${TEST_SCRIPT}" \
    --benchmark "${BENCHMARK}" \
    --dataset "${DATASET}" \
    "${TEST_ARGS[@]}" \
    --result-file "${RUN_DIR}/results.json"
fi

if [[ "${BENCHMARK}" == "locomo" && -s "${RUN_DIR}/results.json" && -s "${RUN_DIR}/token_usage_summary.json" ]]; then
  ELAPSED_SECONDS=$(( $(date +%s) - RUN_STARTED_AT ))
  printf -v ELAPSED "%02d:%02d:%02d" \
    $((ELAPSED_SECONDS / 3600)) \
    $(((ELAPSED_SECONDS % 3600) / 60)) \
    $((ELAPSED_SECONDS % 60))
  MASS_TAG=${WORKSPACE_ACTIVATION_MASS/./}
  "${PYTHON}" "${PROJECT_DIR}/threadworkspace_excel_metrics.py" \
    --results "${RUN_DIR}/results.json" \
    --token-summary "${RUN_DIR}/token_usage_summary.json" \
    --output-dir "${RUN_DIR}" \
    --model "${MODEL_NAME}" \
    --method "TrawMem-${BENCHMARK}-${WORKSPACE_ROLE_SCOPE}-Mass${MASS_TAG}" \
    --elapsed "${ELAPSED}"
fi

touch "${RUN_DIR}/TRAWMEM_COMPLETED"
