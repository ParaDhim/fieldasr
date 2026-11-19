#!/usr/bin/env bash
# Overnight pipeline. Run:  nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
# Resumable: a stage is skipped if its result file exists. A failed stage never stops the run.
# Override via env:  SEEDS="0 1 2"  W2V_SEEDS="0"  AUG_SEEDS="0"
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
       HF_HUB_DISABLE_PROGRESS_BARS=1
SEEDS=${SEEDS:-"0 1 2"}; W2V_SEEDS=${W2V_SEEDS:-"0"}; AUG_SEEDS=${AUG_SEEDS:-"0"}
mkdir -p logs results
STATUS=logs/STATUS.txt
say() { echo "$(date '+%F %T')  $*" | tee -a "$STATUS"; }

stage() {  # stage <log-name> <done-marker> <cmd...>
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then say "SKIP   $name (done: $marker)"; return 0; fi
  say "START  $name"; local t0=$SECONDS
  if "$@" > "logs/$name.log" 2>&1; then
    say "OK     $name ($(( (SECONDS-t0)/60 )) min)"
  else
    say "FAILED $name after $(( (SECONDS-t0)/60 )) min -> see logs/$name.log"
  fi
}
whisper() { uv run python -m fieldasr.train.whisper_ft --config "configs/$1.yaml" --seed "$2"; }

say "=== pipeline start ==="
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | tee -a "$STATUS"

stage 01_prepare            data/processed/qc_report.json  uv run fieldasr prepare
stage 02_build_evalsets     data/evalsets/manifest.csv     uv run fieldasr build-evalsets
stage 03_zeroshot           results/whisper_zeroshot_s0/metrics.json \
      uv run python -m fieldasr.train.zeroshot --config configs/whisper_zeroshot.yaml

# seed 0 of the main comparison first, so core results exist early
stage 04_full_s0            results/whisper_full_s0/metrics.json      whisper whisper_full 0
stage 05_lora_s0            results/whisper_lora_s0/metrics.json      whisper whisper_lora 0
for s in $AUG_SEEDS; do
  stage 06_full_aug_s$s     results/whisper_full_aug_s$s/metrics.json whisper whisper_full_aug $s
done

# deployment: CPU benchmark runs while nothing else is running, so latency is clean
stage 07_export_ct2         results/deploy/ct2/int8/model.bin \
      uv run python -m fieldasr.deploy.export --model-dir results/whisper_full_s0/model --out-dir results/deploy/ct2
stage 08_bench_cpu          results/deploy/bench.json \
      uv run python -m fieldasr.deploy.bench --ct2-dir results/deploy/ct2

# LoRA scored best on clean audio, so export + benchmark it too (the model card ships this one)
stage 08b_export_ct2_lora   results/deploy_lora/ct2/int8/model.bin \
      uv run python -m fieldasr.deploy.export --model-dir results/whisper_lora_s0/model --out-dir results/deploy_lora/ct2
stage 08c_bench_cpu_lora    results/deploy_lora/bench.json \
      uv run python -m fieldasr.deploy.bench --ct2-dir results/deploy_lora/ct2 --out results/deploy_lora/bench.json

for s in $SEEDS; do
  [ "$s" = 0 ] && continue
  stage 09_full_s$s         results/whisper_full_s$s/metrics.json     whisper whisper_full $s
  stage 10_lora_s$s         results/whisper_lora_s$s/metrics.json     whisper whisper_lora $s
done
for s in $W2V_SEEDS; do
  stage 11_w2v2_s$s         results/w2v2_xlsr_s$s/metrics.json \
        uv run python -m fieldasr.train.w2v2_ft --config configs/w2v2_xlsr.yaml --seed "$s"
done

stage 12_aggregate          /nonexistent  uv run python -m fieldasr.aggregate
say "=== pipeline finished ==="; touch logs/DONE
