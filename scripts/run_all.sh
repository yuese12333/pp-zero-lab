#!/usr/bin/env bash
# 一键跑全部实验并覆盖写入 results/metrics.csv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
STEPS="${STEPS:-20}"
WARMUP="${WARMUP:-2}"
BATCH="${BATCH_SIZE:-4}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_all.sh

Run core measured experiments and overwrite results/metrics.csv.

Environment variables:
  PYTHON       Python executable, default: python
  STEPS        Training steps, default: 20
  WARMUP       Warmup steps, default: 2
  BATCH_SIZE   Batch size, default: 4
EOF
  exit 0
fi

METRICS_CSV="$ROOT/results/metrics.csv"
mkdir -p "$(dirname "$METRICS_CSV")"
echo "==> reset $METRICS_CSV"
printf '%s\n' 'config,category,stage,num_gpus,micro_batches,mem_per_gpu_gb,throughput_samples_s,bubble_ratio,comm_volume' > "$METRICS_CSV"

echo "==> baseline"
"$PYTHON" src/train.py --config baseline --steps "$STEPS" --warmup-steps "$WARMUP" --batch-size "$BATCH"

for cfg in zero1 zero2 zero3; do
  echo "==> $cfg"
  if command -v deepspeed >/dev/null 2>&1 && "$PYTHON" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    deepspeed --num_gpus=1 src/train.py --config "$cfg" --steps "$STEPS" --warmup-steps "$WARMUP" --batch-size "$BATCH" --deepspeed
  else
    echo "[skip] $cfg: 需要 CUDA + deepspeed"
  fi
done

for cfg in gpipe 1f1b; do
  echo "==> $cfg"
  "$PYTHON" src/train.py --config "$cfg" --steps "$STEPS" --warmup-steps "$WARMUP" --batch-size "$BATCH" --num-stages 4 --micro-batches 8
done

echo "==> Done. See results/metrics.csv"
