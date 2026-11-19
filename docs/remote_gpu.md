# Running on a remote GPU (Colab / Kaggle / rented box)

Everything is driven from the command line, so any box with an NVIDIA GPU (>= 16 GB; the
runs here used an RTX A6000 48 GB) works. Whisper-small full fine-tune: ~11 min, LoRA: ~14 min
(incl. evaluation of 9 conditions), XLS-R-300M: ~78 min on the A6000.

```bash
# 1. get the code + env  (rented box / Colab terminal / Kaggle notebook shell)
git clone <repo> fieldasr && cd fieldasr
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv sync --extra train --extra deploy --extra app

# 2. ffmpeg WITH AMR-NB (needed to build the phone-codec eval sets). No sudo? static build:
mkdir -p ~/.local/opt/ffmpeg && cd ~/.local/opt/ffmpeg
curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ --strip-components=1
ln -sf $PWD/ffmpeg ~/.local/bin/ffmpeg && cd -
ffmpeg -hide_banner -encoders | grep amr     # must list libopencore_amrnb

# 3. data (FLEURS sw_ke, ~1.5 GB) -> data/raw/fleurs_sw/{train,dev,test}.{tsv,tar.gz}
mkdir -p data/raw/fleurs_sw && cd data/raw/fleurs_sw
for s in train dev test; do
  curl -sLO https://huggingface.co/datasets/google/fleurs/resolve/main/data/sw_ke/audio/$s.tar.gz
  curl -sLo $s.tsv https://huggingface.co/datasets/google/fleurs/resolve/main/data/sw_ke/$s.tsv
done; cd -

# 4. run detached so a dropped SSH/Colab tab does not kill it, then follow the logs
nohup bash scripts/run_all.sh    > logs/run_all.log    2>&1 &
nohup bash scripts/run_extras.sh > logs/run_extras.log 2>&1 &   # after run_all finishes
tail -f logs/STATUS.txt logs/04_full_s0.log
```

* **Colab:** use a GPU runtime; run the same commands in `!bash` cells (or the terminal tab).
  Colab disks are ephemeral: copy `results/` and `mlflow.db` to Drive when done.
* **Kaggle:** enable internet + GPU in the notebook settings; write to `/kaggle/working`.
* Resume after a crash: re-run the same script, finished stages are skipped.
* Smoke-test first: `uv run python -m fieldasr.train.whisper_ft --config configs/whisper_full.yaml --model openai/whisper-tiny --max-steps 6 --eval-limit 8 --out-root /tmp/smoke`.
* Pull results back: `rsync -av box:fieldasr/results/ results/` (exclude `*/model` to skip ~1 GB per model).
