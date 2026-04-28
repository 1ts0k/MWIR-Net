#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CKPT="checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt"
CUDA_ID=0
SHUTDOWN=0
INCLUDE_GTRAIN=0
SKIP_LPIPS=0
SKIP_DERAIN=0
SKIP_DEHAZE=0
SKIP_ABLATIONS=0

DERAIN_OUTPUT="outputs/mwirnet_output_final_plain_multisplit"
DEHAZE_OUTPUT="outputs/mwirnet_output_final_plain_multisplit_dehaze"
ZERO_PROMPT_OUTPUT="outputs/mwirnet_output_ablation_zero_prompt"
NO_CA_OUTPUT="outputs/mwirnet_output_ablation_no_channel_attention"
GTRAIN_OUTPUT="outputs/mwirnet_output_gtrain_plain"

usage() {
  cat <<'EOF'
Usage: bash tools/run_remaining_experiments.sh [options]

Runs the remaining MWIR-Net thesis experiments:
  1. Plain multi-split deraining evaluation
  2. Plain multi-split dehazing evaluation
  3. Rain100L prompt ablations
  4. LPIPS evaluation for the generated outputs

Options:
  --ckpt PATH          Checkpoint path. Default: checkpoints/mwirnet_ckpt_stage2_charb_edge002/epoch=1-step=856.ckpt
  --cuda ID           CUDA device id. Default: 0
  --include-gtrain    Also prepare and run GT-RAIN-test plain inference. This can be slow.
  --skip-lpips        Skip LPIPS evaluation.
  --skip-derain       Skip plain multi-split deraining.
  --skip-dehaze       Skip plain multi-split dehazing.
  --skip-ablations    Skip Rain100L ablation evaluations.
  --shutdown          Shut down the server after all steps finish successfully.
  -h, --help          Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --cuda)
      CUDA_ID="$2"
      shift 2
      ;;
    --include-gtrain)
      INCLUDE_GTRAIN=1
      shift
      ;;
    --skip-lpips)
      SKIP_LPIPS=1
      shift
      ;;
    --skip-derain)
      SKIP_DERAIN=1
      shift
      ;;
    --skip-dehaze)
      SKIP_DEHAZE=1
      shift
      ;;
    --skip-ablations)
      SKIP_ABLATIONS=1
      shift
      ;;
    --shutdown)
      SHUTDOWN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/remaining_experiments_${RUN_ID}"
LOG_FILE="${LOG_DIR}/run.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

run_step() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  echo "[$(date '+%F %T')] $*"
  "$@"
  echo "[$(date '+%F %T')] finished: ${name}"
}

lpips_step() {
  local label="$1"
  local mode="$2"
  local pred_dir="$3"
  local target_dir="$4"

  if [[ ! -d "$pred_dir" ]]; then
    echo "Skip LPIPS ${label}: missing pred dir ${pred_dir}"
    return
  fi
  run_step "LPIPS ${label}" \
    python tools/evaluate_lpips.py \
      --mode "$mode" \
      --pred_dir "$pred_dir" \
      --target_dir "$target_dir"
}

echo "Project root: $PROJECT_ROOT"
echo "Checkpoint: $CKPT"
echo "CUDA: $CUDA_ID"
echo "Log file: $LOG_FILE"

if [[ "$SKIP_DERAIN" -eq 0 ]]; then
  run_step "Derain plain multi-split" \
    python test.py \
      --cuda "$CUDA_ID" \
      --mode 1 \
      --ckpt_path "$CKPT" \
      --output_path "${DERAIN_OUTPUT}/" \
      --derain_splits Rain100L Rain100H Test100 Test1200 Test2800
else
  echo "Skipping derain plain multi-split."
fi

if [[ "$SKIP_DEHAZE" -eq 0 ]]; then
  run_step "Dehaze plain multi-split" \
    python test.py \
      --cuda "$CUDA_ID" \
      --mode 2 \
      --ckpt_path "$CKPT" \
      --output_path "${DEHAZE_OUTPUT}/" \
      --dehaze_splits outdoor nyuhaze500
else
  echo "Skipping dehaze plain multi-split."
fi

if [[ "$SKIP_ABLATIONS" -eq 0 ]]; then
  run_step "Ablation zero_prompt on Rain100L" \
    python test.py \
      --cuda "$CUDA_ID" \
      --mode 1 \
      --ckpt_path "$CKPT" \
      --output_path "${ZERO_PROMPT_OUTPUT}/" \
      --derain_splits Rain100L \
      --ablation_mode zero_prompt

  run_step "Ablation no_channel_attention on Rain100L" \
    python test.py \
      --cuda "$CUDA_ID" \
      --mode 1 \
      --ckpt_path "$CKPT" \
      --output_path "${NO_CA_OUTPUT}/" \
      --derain_splits Rain100L \
      --ablation_mode no_channel_attention
else
  echo "Skipping ablations."
fi

if [[ "$INCLUDE_GTRAIN" -eq 1 ]]; then
  run_step "Prepare GT-RAIN-test" \
    python tools/prepare_gtrain_test.py --split test

  run_step "GT-RAIN-test plain inference" \
    python test.py \
      --cuda "$CUDA_ID" \
      --mode 1 \
      --ckpt_path "$CKPT" \
      --output_path "${GTRAIN_OUTPUT}/" \
      --derain_splits GT-RAIN-test
fi

if [[ "$SKIP_LPIPS" -eq 0 ]]; then
  lpips_step "Rain100L plain" derain "${DERAIN_OUTPUT}/derain" "test/derain/Rain100L/target"
  lpips_step "Rain100H plain" derain "${DERAIN_OUTPUT}/derain_Rain100H" "test/derain/Rain100H/target"
  lpips_step "Test100 plain" derain "${DERAIN_OUTPUT}/derain_Test100" "test/derain/Test100/target"
  lpips_step "Test1200 plain" derain "${DERAIN_OUTPUT}/derain_Test1200" "test/derain/Test1200/target"
  lpips_step "Test2800 plain" derain "${DERAIN_OUTPUT}/derain_Test2800" "test/derain/Test2800/target"
  lpips_step "SOTS outdoor plain" dehaze "${DEHAZE_OUTPUT}/dehaze_outdoor" "test/dehaze/outdoor/target"
  lpips_step "SOTS nyuhaze500 plain" dehaze "${DEHAZE_OUTPUT}/dehaze_nyuhaze500" "test/dehaze/nyuhaze500/target"
  lpips_step "zero_prompt Rain100L" derain "${ZERO_PROMPT_OUTPUT}/derain" "test/derain/Rain100L/target"
  lpips_step "no_channel_attention Rain100L" derain "${NO_CA_OUTPUT}/derain" "test/derain/Rain100L/target"
else
  echo "Skipping LPIPS evaluation."
fi

echo
echo "All requested experiments finished successfully."
echo "Log saved to: $LOG_FILE"

if [[ "$SHUTDOWN" -eq 1 ]]; then
  echo "Shutdown requested. Powering off now."
  sync
  shutdown -h now
fi
