# FieldASR

Swahili speech recognition under **field conditions** (noisy, low-quality phone audio): data QC,
field-realistic evaluation (WER/CER **per condition**), Whisper fine-tuning (LoRA vs full, 3 seeds),
a wav2vec2 baseline, error analysis, a review tool + retraining round, PII scrubbing, and INT8 CPU
deployment (offline CLI + Dockerised API).

Corpus: Google FLEURS `sw_ke` (read speech, CC-BY-4.0). Read [docs/dataset_card.md](docs/dataset_card.md)
for the limitations before quoting any number: noise/codecs are **simulated**, speech is **read**,
and FLEURS has **no speaker IDs or accent labels**.

## Results (test split, WER %, mean +- std over 3 seeds unless noted)

| condition | zero-shot Whisper-small | full FT | **LoRA** | full FT + field aug (1 seed) | XLS-R CTC (1 seed) |
|---|---|---|---|---|---|
| clean | 84.7 | 25.2 +- 0.1 | **21.4 +- 0.3** | 24.3 | 22.9 |
| mp3_8k | 91.5 | 32.0 +- 0.5 | 27.6 +- 0.8 | 28.2 | 27.9 |
| amr_nb_8k | 149.7 | 52.5 +- 1.9 | 50.2 +- 1.7 | 51.7 | **41.5** |
| babble_snr10 | 129.6 | 43.1 +- 0.4 | 45.9 +- 4.0 | **38.8** | 46.2 |
| pink_snr10 | 113.3 | 41.9 +- 1.1 | 38.3 +- 0.4 | 38.9 | 39.2 |
| cheap_mic_reverb | 115.5 | 55.9 +- 1.4 | 51.8 +- 1.8 | **47.8** | 65.9 |
| pink_snr0 | 191.4 | 100.4 +- 7.6 | 96.3 +- 5.3 | 95.4 | **73.6** |
| babble_snr0 | 232.5 | 131.7 +- 2.6 | 141.6 +- 4.2 | 125.7 | **107.6** |
| domain_terms (numerals/dates/health/school) | 81.2 | 25.6 +- 0.1 | 22.1 +- 1.0 | 25.1 | 25.0 |

CER, gender slice and per-domain-tag numbers: [results/summary.md](results/summary.md).
**The headline "clean" number is misleading:** error roughly doubles under phone-codec or
cheap-mic conditions and is ~100% at 0 dB noise. WER > 100% = insertions (Whisper hallucinates).

Key takeaways (details and caveats in the docs):
* Fine-tuning takes Swahili WER 85% -> 21-25%. LoRA beats full FT on clean audio (21.4 vs 25.2).
* Whisper's low-SNR failure is **insertions/hallucination**; the CTC model's is **deletions**.
* Field augmentation helps moderate degradation (cheap-mic 55.9 -> 47.8) but not 0 dB; one seed.
* **INT8 quantisation did not hurt accuracy** and is ~1.3-1.8x faster on CPU (see below).
* Adding ~60 field-condition corrections helps noisy conditions; *targeted* selection was not clearly
  better than random ([docs/retraining_round.md](docs/retraining_round.md); the reviewer there is a
  **simulated oracle**, no human corrected anything yet).

## CPU deployment (CTranslate2 / faster-whisper, 8 threads, 100 clips per condition)
**LoRA model (shipped):**

| condition | WER fp32 | WER int8 | dWER | lat fp32 (s) | lat int8 (s) | RTF fp32 | RTF int8 |
|---|---|---|---|---|---|---|---|
| clean | 0.230 | 0.230 | -0.001 | 1.71 | 1.14 | 0.121 | 0.081 |
| babble_snr10 | 0.445 | 0.414 | -0.030 | 2.44 | 1.51 | 0.173 | 0.107 |
| amr_nb_8k | 0.413 | 0.411 | -0.002 | 2.37 | 1.55 | 0.168 | 0.110 |

**Full fine-tune (seed 0), same protocol:**

| condition | WER fp32 | WER int8 | dWER | lat fp32 (s) | lat int8 (s) | RTF fp32 | RTF int8 |
|---|---|---|---|---|---|---|---|
| clean | 0.252 | 0.254 | +0.003 | 1.64 | 1.17 | 0.117 | 0.083 |
| babble_snr10 | 0.426 | 0.425 | -0.001 | 2.05 | 1.15 | 0.146 | 0.082 |
| amr_nb_8k | 0.442 | 0.434 | -0.008 | 1.95 | 1.56 | 0.139 | 0.111 |

Subsets of the test conditions, 100 clips each (WER here is on that subset, so it differs from the full-test table). Model size: INT8 242 MB vs FP32 928 MB. RTF = processing time / audio duration (0.08 = 12x faster than real time).

**Quantisation did not hurt accuracy**: WER differences are -0.1 to -3.0 points (INT8 is slightly *better* on some), which at n=100 per condition is within noise, not a real gain; hallucination-prone clips dominate the variance. Latency is single-stream on 8 CPU threads, greedy decoding, no VAD.


## Reproduce everything
```bash
make reproduce      # setup + tests + all training/eval/export/benchmark + analyses (resumable)
```
Needs: `uv`, an NVIDIA GPU (>= 16 GB), `ffmpeg` with `libopencore_amrnb` on PATH, FLEURS files in
`data/raw/fleurs_sw/` ([docs/remote_gpu.md](docs/remote_gpu.md) has the exact commands, incl. no-sudo
ffmpeg and remote GPU use). Measured on one RTX A6000: all training + eval ~3 h, retraining round ~45 min.
Stages log to `logs/*.log`, status in `logs/STATUS*.txt`, runs tracked in MLflow (`mlflow.db`;
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`). Configs are in `configs/`; seeds 0-2.

| piece | command |
|---|---|
| data prep + QC + leakage check | `uv run fieldasr prepare` |
| field eval sets | `uv run fieldasr build-evalsets` |
| train one run | `uv run python -m fieldasr.train.whisper_ft --config configs/whisper_lora.yaml --seed 0` |
| error analysis + listening sheet | `uv run python -m fieldasr.analysis.errors` |
| review tool | `make review` (http://127.0.0.1:7860) |
| offline transcribe | `uv run fieldasr transcribe clip.wav --model-dir results/deploy_lora/ct2/int8` |
| API | `make api`, or `docker build -t fieldasr . && docker run -p 8000:8000 -v $PWD/results/deploy_lora/ct2/int8:/models/int8:ro fieldasr` |
| tests / lint / types | `make test lint` (32 tests, ruff, mypy) |

## Docs
[dataset card](docs/dataset_card.md) | [model card](docs/model_card.md) | [error analysis](docs/error_analysis.md) |
[retraining round](docs/retraining_round.md) | [data handling](docs/data_handling.md) | [remote GPU](docs/remote_gpu.md)

## Honest status / not done
* **Human listening pass over ~100 failures: not done.** Taxonomy is auto-tagged; the sheet is ready
  (`results/error_analysis/listening_sheet.html`).
* **No real reviewer corrections:** the retraining round uses an oracle reviewer (clearly labelled).
* **Docker image was written but not built here** (no access to the Docker daemon); its dependency
  set was verified in a clean venv (no torch) and the API was run with the real INT8 model.
* No speaker-level leakage audit (FLEURS has no speaker IDs); no accent or child-speech subset.
* PII scrubber recall is unmeasured on real data (FLEURS has none); see the data-handling note.
* wav2vec2 and the augmented run have a single seed; zero-shot is deterministic.
