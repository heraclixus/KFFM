#!/bin/bash
# Run signature-kernel time-series seeded experiments on CUDA GPUs.
#
# Intended usage on the GPU server:
#   CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 bash scripts/run_signature_timeseries_cuda.sh pilot
#
# By default this launches one worker per visible GPU. Each worker runs its
# assigned experiments sequentially, while different GPUs run in parallel.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}/scripts"

GROUP="${1:-pilot}"
METRIC="${METRIC:-mmd_rbf}"
SEEDS="${SEEDS:-0}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-../outputs/signature_timeseries_cuda}"
BASE="${BASE:-../outputs}"
LOG_DIR="${LOG_DIR:-${OUT}/logs}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SIGNATURE_KERNEL_STRICT="${SIGNATURE_KERNEL_STRICT:-1}"
SIGNATURE_KERNEL_FORCE_CPU="${SIGNATURE_KERNEL_FORCE_CPU:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [[ -n "${PYTHON_CMD:-}" ]]; then
  read -r -a PYTHON_ARR <<< "${PYTHON_CMD}"
elif [[ -n "${PYTHON:-}" ]]; then
  PYTHON_ARR=("${PYTHON}")
elif [[ -x "../.venv/bin/python" ]]; then
  PYTHON_ARR=("../.venv/bin/python")
else
  PYTHON_ARR=("python")
fi

if [[ -n "${GPUS:-}" ]]; then
  GPU_RAW="${GPUS//,/ }"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_RAW="${CUDA_VISIBLE_DEVICES//,/ }"
else
  GPU_RAW="0 1 2 4 5 6"
fi
read -r -a GPU_LIST <<< "${GPU_RAW}"

RUNS=()
TASKS=()

add_run() {
  RUNS+=("$1|$2")
}

case "$GROUP" in
  pilot)
    add_run heston signature_ll
    add_run heston signature_sig05
    add_run heston signature_d2
    add_run aemet signature
    add_run aemet signature_ll
    add_run expr_genes signature
    add_run expr_genes signature_ll
    ;;
  heston)
    add_run heston signature
    add_run heston signature_ll
    add_run heston signature_d2
    add_run heston signature_ll_d2
    add_run heston signature_sig05
    add_run heston signature_sig2
    add_run heston signature_sig5
    ;;
  table1)
    add_run aemet signature
    add_run expr_genes signature
    add_run economy signature
    ;;
  verify)
    add_run aemet signature
    add_run aemet signature_ll
    add_run expr_genes signature
    add_run expr_genes signature_ll
    add_run economy signature
    add_run economy signature_ll
    ;;
  all)
    add_run heston signature
    add_run heston signature_ll
    add_run heston signature_d2
    add_run heston signature_ll_d2
    add_run heston signature_sig05
    add_run heston signature_sig2
    add_run heston signature_sig5
    add_run aemet signature
    add_run aemet signature_ll
    add_run aemet signature_sig05
    add_run expr_genes signature
    add_run expr_genes signature_ll
    add_run expr_genes signature_sig05
    add_run economy signature
    add_run economy signature_ll
    ;;
  *)
    echo "Unknown group: ${GROUP}" >&2
    echo "Expected one of: pilot, heston, table1, verify, all" >&2
    exit 2
    ;;
esac

for run in "${RUNS[@]}"; do
  IFS='|' read -r dataset method <<< "${run}"
  for seed in ${SEEDS}; do
    TASKS+=("${dataset}|${method}|${seed}")
  done
done

mkdir -p "${LOG_DIR}"

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set GPUS='0 1 2 4 5 6' or CUDA_VISIBLE_DEVICES=0,1,2,4,5,6." >&2
  exit 2
fi

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "No tasks selected." >&2
  exit 2
fi

MAX_JOBS="${MAX_JOBS:-${#GPU_LIST[@]}}"
if (( MAX_JOBS > ${#GPU_LIST[@]} )); then
  MAX_JOBS="${#GPU_LIST[@]}"
fi
if (( MAX_JOBS > ${#TASKS[@]} )); then
  MAX_JOBS="${#TASKS[@]}"
fi
if (( MAX_JOBS < 1 )); then
  MAX_JOBS=1
fi

log_progress() {
  echo "[$(date)] $*" | tee -a "${LOG_DIR}/progress.log"
}

print_config() {
  {
    echo "[$(date)] group=${GROUP}"
    echo "python=${PYTHON_ARR[*]}"
    echo "metric=${METRIC}"
    echo "seeds=${SEEDS}"
    echo "device=${DEVICE}"
    echo "gpus=${GPU_LIST[*]}"
    echo "workers=${MAX_JOBS}"
    echo "output=${OUT}"
    echo "base_outputs=${BASE}"
    echo "strict_signature=${SIGNATURE_KERNEL_STRICT}"
    echo "force_cpu_signature=${SIGNATURE_KERNEL_FORCE_CPU}"
    echo "skip_existing=${SKIP_EXISTING}"
    echo "tasks=${#TASKS[@]}"
    for task in "${TASKS[@]}"; do
      echo "task=${task}"
    done
  } | tee "${LOG_DIR}/run_config.txt"
}

preflight_gpu() {
  local gpu="$1"
  log_progress "PREFLIGHT gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  SIGNATURE_KERNEL_STRICT="${SIGNATURE_KERNEL_STRICT}" \
  SIGNATURE_KERNEL_FORCE_CPU="${SIGNATURE_KERNEL_FORCE_CPU}" \
  PYTHONFAULTHANDLER=1 \
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_ARR[@]}" -u - <<'PY'
import torch
from optimal_transport import create_kernel

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() >= 1, "No CUDA device visible"

kernel = create_kernel(
    "signature",
    time_aug=True,
    lead_lag=True,
    dyadic_order=1,
    static_kernel_type="rbf",
    static_kernel_sigma=1.0,
    add_basepoint=True,
    normalize=True,
    max_seq_len=16,
    max_batch=2,
)
assert getattr(kernel, "_pysiglib_available", False), "pySigLib is not available"

x = torch.randn(2, 1, 16, device="cuda")
K = kernel(x, x)
print("signature_kernel_shape", tuple(K.shape))
print("signature_kernel_device", K.device)
print("signature_kernel_finite", torch.isfinite(K).all().item())
assert K.device.type == "cuda", K.device
assert torch.isfinite(K).all().item(), "signature kernel returned non-finite values"
PY
}

run_task() {
  local worker="$1"
  local gpu="$2"
  local dataset="$3"
  local method="$4"
  local seed="$5"
  local log="${LOG_DIR}/${dataset}-${method}-s${seed}-gpu${gpu}.log"

  log_progress "START worker=${worker} gpu=${gpu} dataset=${dataset} method=${method} seed_index=${seed}"

  local cmd=(
    "${PYTHON_ARR[@]}" -u run_seeded_experiments.py
    --dataset "${dataset}"
    --kernel "${method}"
    --metric "${METRIC}"
    --seed "${seed}"
    --device "${DEVICE}"
    --output_dir "${OUT}"
    --base_outputs_dir "${BASE}"
  )
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q SIGNATURE_KERNEL_STRICT=%q SIGNATURE_KERNEL_FORCE_CPU=%q PYTHONPATH=%q ' \
      "${gpu}" "${SIGNATURE_KERNEL_STRICT}" "${SIGNATURE_KERNEL_FORCE_CPU}" "${PROJECT_ROOT}:\${PYTHONPATH:-}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" \
  SIGNATURE_KERNEL_STRICT="${SIGNATURE_KERNEL_STRICT}" \
  SIGNATURE_KERNEL_FORCE_CPU="${SIGNATURE_KERNEL_FORCE_CPU}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  PYTHONFAULTHANDLER=1 \
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  "${cmd[@]}" > "${log}" 2>&1
  local status=$?

  if grep -E "Falling back to RBF|SignatureKernel failed|pySigLib not found" "${log}" >/dev/null 2>&1; then
    log_progress "FAIL worker=${worker} gpu=${gpu} dataset=${dataset} method=${method} seed_index=${seed} reason=signature_fallback log=${log}"
    return 90
  fi

  if [[ "${status}" == "0" ]]; then
    log_progress "DONE worker=${worker} gpu=${gpu} dataset=${dataset} method=${method} seed_index=${seed}"
  else
    log_progress "FAIL worker=${worker} gpu=${gpu} dataset=${dataset} method=${method} seed_index=${seed} status=${status} log=${log}"
  fi

  return "${status}"
}

worker_loop() {
  local worker="$1"
  local gpu="$2"
  local n_workers="$3"
  local failures=0
  local i task dataset method seed

  for ((i = worker; i < ${#TASKS[@]}; i += n_workers)); do
    task="${TASKS[$i]}"
    IFS='|' read -r dataset method seed <<< "${task}"
    if ! run_task "${worker}" "${gpu}" "${dataset}" "${method}" "${seed}"; then
      failures=$((failures + 1))
    fi
  done

  return "${failures}"
}

print_config

if [[ "${DRY_RUN}" == "1" ]]; then
  for ((worker = 0; worker < MAX_JOBS; worker++)); do
    gpu="${GPU_LIST[$worker]}"
    worker_loop "${worker}" "${gpu}" "${MAX_JOBS}" || true
  done
  exit 0
fi

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  for ((worker = 0; worker < MAX_JOBS; worker++)); do
    preflight_gpu "${GPU_LIST[$worker]}" | tee "${LOG_DIR}/preflight-gpu${GPU_LIST[$worker]}.log"
  done
fi

log_progress "LAUNCH group=${GROUP} workers=${MAX_JOBS} tasks=${#TASKS[@]}"

pids=()
for ((worker = 0; worker < MAX_JOBS; worker++)); do
  gpu="${GPU_LIST[$worker]}"
  worker_loop "${worker}" "${gpu}" "${MAX_JOBS}" &
  pids+=("$!")
  log_progress "WORKER worker=${worker} gpu=${gpu} pid=${pids[$(( ${#pids[@]} - 1 ))]}"
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

log_progress "FINISH group=${GROUP} worker_failures=${failures} output=${OUT}"

if [[ "${failures}" != "0" ]]; then
  exit 1
fi
