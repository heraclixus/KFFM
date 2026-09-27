#!/bin/bash
# Run local signature-kernel pilots sequentially in the `signature` conda env.
#
# These are intentionally local-only: the current SLURM environment does not
# have a working PySigLib install, so server signature jobs fall back to RBF.

set -uo pipefail

cd "$(dirname "$0")"

GROUP="${1:-pilot}"
CONDA_ENV="${CONDA_ENV:-signature}"
# CPU is the safest default in the local `signature` env: PySigLib works there,
# while gpytorch GP-prior sampling currently hits a CPU/MPS device mismatch.
DEVICE="${DEVICE:-cpu}"
METRIC="${METRIC:-mmd_rbf}"
SEEDS="${SEEDS:-0}"
OUT="${OUT:-../outputs/local_signature_gap_pilots}"
BASE="${BASE:-../outputs/outputs}"
LOG_DIR="${LOG_DIR:-${OUT}/logs}"
DRY_RUN="${DRY_RUN:-0}"
FAIL_FAST="${FAIL_FAST:-0}"

export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

mkdir -p "$LOG_DIR"

RUNS=()

add_run() {
  RUNS+=("$1 $2")
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
    echo "Expected one of: pilot, heston, verify, all" >&2
    exit 2
    ;;
esac

run_one() {
  local dataset="$1"
  local method="$2"
  local seed="$3"
  local log="${LOG_DIR}/${dataset}-${method}-s${seed}.log"

  echo "[$(date)] START dataset=${dataset} method=${method} seed_index=${seed} device=${DEVICE}" | tee -a "${LOG_DIR}/progress.log"

  local cmd=(
    conda run -n "$CONDA_ENV" python run_seeded_experiments.py
    --dataset "$dataset"
    --kernel "$method"
    --metric "$METRIC"
    --seed "$seed"
    --device "$DEVICE"
    --output_dir "$OUT"
    --base_outputs_dir "$BASE"
    --skip-existing
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}" > "$log" 2>&1
  local status=$?

  if [[ "$status" == "0" ]]; then
    echo "[$(date)] DONE dataset=${dataset} method=${method} seed_index=${seed}" | tee -a "${LOG_DIR}/progress.log"
  else
    echo "[$(date)] FAIL dataset=${dataset} method=${method} seed_index=${seed} status=${status}; see ${log}" | tee -a "${LOG_DIR}/progress.log"
  fi

  return "$status"
}

failures=0
count=0

for run in "${RUNS[@]}"; do
  read -r dataset method <<< "$run"
  for seed in $SEEDS; do
    count=$((count + 1))
    if ! run_one "$dataset" "$method" "$seed"; then
      failures=$((failures + 1))
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit 1
      fi
    fi
  done
done

echo "[$(date)] Finished group=${GROUP}; attempted=${count}; failures=${failures}; output=${OUT}" | tee -a "${LOG_DIR}/progress.log"

if [[ "$failures" != "0" ]]; then
  exit 1
fi
