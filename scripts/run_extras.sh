#!/usr/bin/env bash
# Post-training analyses + the simulated-reviewer retraining round.
#   nohup bash scripts/run_extras.sh > logs/run_extras.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
SEEDS=${SEEDS:-"0 1 2"}
STATUS=logs/STATUS_extras.txt
say() { echo "$(date '+%F %T')  $*" | tee -a "$STATUS"; }
stage() { local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then say "SKIP   $name"; return 0; fi
  say "START  $name"; local t0=$SECONDS
  if "$@" > "logs/$name.log" 2>&1; then say "OK     $name ($(( (SECONDS-t0)/60 )) min)"
  else say "FAILED $name -> logs/$name.log"; fi; }

say "=== extras start ==="
stage 20_error_analysis   results/error_analysis/failures_100.csv  uv run python -m fieldasr.analysis.errors
stage 21_privacy_report   results/privacy_report.json              uv run python -m fieldasr.privacy.report
stage 22_review_queue     data/review/queue.csv                    uv run python -m fieldasr.analysis.make_review_queue
for s in $SEEDS; do for arm in replay_only random60 targeted60; do
  stage 23_retrain_${arm}_s$s results/retrain_${arm}_s$s/metrics.json \
    uv run python -m fieldasr.train.retrain --config configs/retrain_$arm.yaml --seed $s
done; done
stage 29_aggregate /nonexistent uv run python -m fieldasr.aggregate
say "=== extras finished ==="; touch logs/DONE_extras
