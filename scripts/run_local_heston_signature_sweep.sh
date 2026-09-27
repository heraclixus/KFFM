#!/bin/bash
# Local-only Heston signature-kernel sweep.
#
# The SLURM environment has not had a working PySigLib install, so this script
# runs in the local `signature` conda environment and refuses silent
# signature-to-RBF fallback. By default it keeps four method/seed tasks in
# flight; set PARALLEL=1 to force sequential execution.
#
# Examples:
#   bash scripts/run_local_heston_signature_sweep.sh pilot
#   nohup bash scripts/run_local_heston_signature_sweep.sh pilot > outputs/local_heston_signature_sweep/nohup.log 2>&1 &
#   SEEDS="0 1 2 3 4 5 6 7 8 9" bash scripts/run_local_heston_signature_sweep.sh full
#   PARALLEL=4 nohup bash scripts/run_local_heston_signature_sweep.sh full > outputs/local_heston_signature_sweep/nohup.log 2>&1 &

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}/scripts"

GROUP="${1:-pilot}"
CONDA_ENV="${CONDA_ENV:-signature}"
DEVICE="${DEVICE:-cpu}"
METRIC="${METRIC:-mmd_rbf}"
OUT="${OUT:-../outputs/local_heston_signature_sweep}"
BASE="${BASE:-../outputs/outputs}"
LOG_DIR="${LOG_DIR:-${OUT}/logs}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PARALLEL="${PARALLEL:-4}"

if ! [[ "${PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${PARALLEL}" -lt 1 ]]; then
  echo "PARALLEL must be a positive integer, got: ${PARALLEL}" >&2
  exit 2
fi

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export SIGNATURE_KERNEL_STRICT="${SIGNATURE_KERNEL_STRICT:-1}"
export SIGNATURE_KERNEL_FORCE_CPU="${SIGNATURE_KERNEL_FORCE_CPU:-0}"
export KFFM_SKIP_PLOTS="${KFFM_SKIP_PLOTS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

METHODS=(
  hsig_r05_s1_l64
  hsig_r05_s1_l128
  hsig_r01_s1_l128
  hsig_r1_s1_l128
  hsig_r05_s05_l128
  hsig_r05_s2_l128
  hsig_r05_s1_ll_l128
  hsig_r05_s1_d2_l128
)

case "${GROUP}" in
  pilot)
    DEFAULT_SEEDS="0"
    ;;
  full)
    DEFAULT_SEEDS="0 1 2 3 4 5 6 7 8 9"
    ;;
  *)
    echo "Unknown group: ${GROUP}" >&2
    echo "Expected one of: pilot, full" >&2
    exit 2
    ;;
esac

SEEDS="${SEEDS:-${DEFAULT_SEEDS}}"

mkdir -p "${LOG_DIR}"

log_progress() {
  echo "[$(date)] $*" | tee -a "${LOG_DIR}/progress.log"
}

run_one() {
  local method="$1"
  local seed_index="$2"
  local log="${LOG_DIR}/heston-${method}-s${seed_index}.log"

  log_progress "START method=${method} seed_index=${seed_index} metric=${METRIC} device=${DEVICE}"

  local cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}" python -u run_seeded_experiments.py
    --dataset heston
    --kernel "${method}"
    --metric "${METRIC}"
    --seed "${seed_index}"
    --device "${DEVICE}"
    --output_dir "${OUT}"
    --base_outputs_dir "${BASE}"
  )
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}" > "${log}" 2>&1
  local status=$?

  if grep -E "fall back to RBF|Falling back to RBF|pySigLib not found|SignatureKernel failed" "${log}" >/dev/null 2>&1; then
    log_progress "FAIL method=${method} seed_index=${seed_index} reason=signature_fallback log=${log}"
    return 90
  fi

  if [[ "${status}" == "0" ]]; then
    log_progress "DONE method=${method} seed_index=${seed_index}"
  else
    log_progress "FAIL method=${method} seed_index=${seed_index} status=${status} log=${log}"
  fi

  return "${status}"
}

{
  echo "[$(date)] group=${GROUP}"
  echo "conda_env=${CONDA_ENV}"
  echo "device=${DEVICE}"
  echo "metric=${METRIC}"
  echo "output=${OUT}"
  echo "base_outputs=${BASE}"
  echo "strict_signature=${SIGNATURE_KERNEL_STRICT}"
  echo "force_cpu_signature=${SIGNATURE_KERNEL_FORCE_CPU}"
  echo "skip_plots=${KFFM_SKIP_PLOTS}"
  echo "skip_existing=${SKIP_EXISTING}"
  echo "parallel=${PARALLEL}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
  echo "openblas_num_threads=${OPENBLAS_NUM_THREADS}"
  echo "methods=${#METHODS[@]}"
  echo "seeds=${SEEDS}"
  for method in "${METHODS[@]}"; do
    echo "method=${method}"
  done
} | tee "${LOG_DIR}/run_config.txt"

seed_count=0
for _ in ${SEEDS}; do
  seed_count=$((seed_count + 1))
done
total_tasks=$((${#METHODS[@]} * seed_count))
log_progress "PLAN configs=${#METHODS[@]} seeds=${seed_count} total_runs=${total_tasks}"

failures=0
attempted=0

batch_pids=()
batch_labels=()

wait_for_batch() {
  local pid
  local label
  local status
  local i

  for i in "${!batch_pids[@]}"; do
    pid="${batch_pids[$i]}"
    label="${batch_labels[$i]}"
    if wait "${pid}"; then
      status=0
    else
      status=$?
      failures=$((failures + 1))
      log_progress "BATCH_FAIL label=${label} status=${status}"
      if [[ "${FAIL_FAST}" == "1" ]]; then
        local j
        for j in "${!batch_pids[@]}"; do
          if [[ "${j}" != "${i}" ]]; then
            kill "${batch_pids[$j]}" >/dev/null 2>&1 || true
          fi
        done
        exit 1
      fi
    fi
  done

  batch_pids=()
  batch_labels=()
}

for seed in ${SEEDS}; do
  for method in "${METHODS[@]}"; do
    attempted=$((attempted + 1))

    if [[ "${DRY_RUN}" == "1" || "${PARALLEL}" == "1" ]]; then
      if ! run_one "${method}" "${seed}"; then
        failures=$((failures + 1))
        if [[ "${FAIL_FAST}" == "1" ]]; then
          exit 1
        fi
      fi
      continue
    fi

    run_one "${method}" "${seed}" &
    pid=$!
    batch_pids+=("${pid}")
    batch_labels+=("${method}:s${seed}")
    log_progress "LAUNCHED pid=${pid} method=${method} seed_index=${seed}"

    if [[ "${#batch_pids[@]}" -ge "${PARALLEL}" ]]; then
      wait_for_batch
    fi
  done
done

if [[ "${#batch_pids[@]}" -gt 0 ]]; then
  wait_for_batch
fi

log_progress "FINISHED group=${GROUP} attempted=${attempted} failures=${failures} output=${OUT}"

if [[ "${failures}" != "0" ]]; then
  exit 1
fi
