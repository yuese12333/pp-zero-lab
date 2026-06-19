#!/usr/bin/env bash
# pp-zero-lab deployment helper.
#
# Usage:
#   bash scripts/deploy.sh check
#   bash scripts/deploy.sh install
#   bash scripts/deploy.sh core
#   bash scripts/deploy.sh extended
#   bash scripts/deploy.sh full
#   bash scripts/deploy.sh viz
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

METRICS_HEADER='config,category,stage,num_gpus,micro_batches,mem_per_gpu_gb,throughput_samples_s,bubble_ratio,comm_volume'
SWEEP_HEADER='sweep_type,schedule,num_stages,micro_batches,n_layer,n_embd,params,bubble_theory,bubble_measured,bubble_abs_err,mem_per_gpu_gb,throughput_samples_s,step_time_ms'
COMBO_HEADER='exp,pipeline_stages_P,zero_stage,dp_degree_N,total_gpus,params_total,params_per_stage,mem_per_gpu_gb,vs_baseline_x'
TRADEOFF_HEADER='exp,num_stages_K,micro_batches_M,bubble_ratio,effective_compute_pct'

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf '[warn] %s\n' "$*" >&2
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash scripts/deploy.sh <command>

Commands:
  check       Check Python, key packages, CUDA and DeepSpeed visibility.
  install     Install PyTorch CUDA wheel and requirements.txt packages.
  core        Regenerate results/metrics.csv: measured rows + ZeRO theory rows.
  extended    Regenerate results/sweep.csv, combo.csv and tradeoff.csv.
  full        Run check, core and extended.
  viz         Start local static server for viz/index.html.
  status      Print result file row counts and visualization artifact list.

Environment variables:
  PYTHON       Python executable, default: python
  STEPS        Forwarded to scripts/run_all.sh, default there: 20
  WARMUP       Forwarded to scripts/run_all.sh, default there: 2
  BATCH_SIZE   Forwarded to scripts/run_all.sh, default there: 4
  HOST         Visualization server host, default: 127.0.0.1
  PORT         Visualization server port, default: 8765
  SKIP_TORCH   Set to 1 for install command to skip PyTorch wheel install.
EOF
}

require_python() {
  command -v "$PYTHON" >/dev/null 2>&1 || die "Python executable not found: $PYTHON"
}

check_env() {
  require_python
  log "Python"
  "$PYTHON" --version

  log "Package visibility"
  "$PYTHON" - <<'PY'
import importlib.util
mods = ["torch", "numpy", "pandas", "deepspeed"]
for name in mods:
    spec = importlib.util.find_spec(name)
    print(f"{name:10s}: {'ok' if spec else 'missing'}")
PY

  log "CUDA / DeepSpeed runtime"
  "$PYTHON" - <<'PY'
try:
    import torch
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", exc)

try:
    import deepspeed
    print("deepspeed version:", deepspeed.__version__)
except Exception as exc:
    print("deepspeed import failed:", exc)
PY

  if ! command -v deepspeed >/dev/null 2>&1; then
    warn "deepspeed command not found; scripts/run_all.sh will skip ZeRO measured rows."
  fi
}

install_deps() {
  require_python
  log "Upgrade pip"
  "$PYTHON" -m pip install --upgrade pip

  if [[ "${SKIP_TORCH:-0}" != "1" ]]; then
    log "Install PyTorch CUDA wheel"
    "$PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  else
    warn "SKIP_TORCH=1, skip PyTorch wheel install."
  fi

  log "Install project requirements"
  "$PYTHON" -m pip install -r requirements.txt
}

reset_csv() {
  local path="$1"
  local header="$2"
  mkdir -p "$(dirname "$path")"
  printf '%s\n' "$header" > "$path"
}

run_core() {
  require_python
  log "Run measured core experiments"
  PYTHON="$PYTHON" bash scripts/run_all.sh

  log "Append ZeRO multi-GPU theory rows"
  "$PYTHON" src/theory.py
}

run_extended() {
  require_python
  log "Reset extended CSV files"
  reset_csv "$ROOT/results/sweep.csv" "$SWEEP_HEADER"
  reset_csv "$ROOT/results/combo.csv" "$COMBO_HEADER"
  reset_csv "$ROOT/results/tradeoff.csv" "$TRADEOFF_HEADER"

  log "Run sweep experiments"
  "$PYTHON" src/sweep.py

  log "Run PP x ZeRO combo and tradeoff modeling"
  "$PYTHON" src/combo.py
}

show_status() {
  log "Result files"
  "$PYTHON" - <<'PY'
from pathlib import Path
root = Path.cwd()
for rel in ["results/metrics.csv", "results/sweep.csv", "results/combo.csv", "results/tradeoff.csv"]:
    path = root / rel
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = max(0, len(lines) - 1)
        print(f"{rel:24s} rows={rows:<4d} size={path.stat().st_size}")
    else:
        print(f"{rel:24s} missing")
PY

  log "Visualization artifacts"
  if [[ -d "$ROOT/viz" ]]; then
    find "$ROOT/viz" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort || ls -1 "$ROOT/viz"
  else
    warn "viz directory missing"
  fi
}

serve_viz() {
  require_python
  [[ -f "$ROOT/viz/index.html" ]] || die "viz/index.html not found"
  log "Serving visualization"
  printf 'Open: http://%s:%s/viz/index.html\n' "$HOST" "$PORT"
  "$PYTHON" -m http.server "$PORT" --bind "$HOST"
}

main() {
  local command="${1:-}"
  case "$command" in
    check)
      check_env
      ;;
    install)
      install_deps
      ;;
    core)
      run_core
      show_status
      ;;
    extended)
      run_extended
      show_status
      ;;
    full)
      check_env
      run_core
      run_extended
      show_status
      ;;
    viz)
      serve_viz
      ;;
    status)
      show_status
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      usage
      die "Unknown command: $command"
      ;;
  esac
}

main "$@"
