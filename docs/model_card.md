# Model card: FieldASR Swahili (Whisper-small + LoRA, INT8 CTranslate2)

## Model
`openai/whisper-small` (244M) fine-tuned with LoRA (r=32, alpha=64, dropout 0.05 on q/k/v/out/fc1/fc2;
13.0M trainable adapter parameters, ~5% on top of the frozen base; adapters merged after
training), 5 epochs (910 steps, ~10 min), bs 16, lr 5e-4, bf16, on 2925 FLEURS `sw_ke` train clips (12.4 h) that
passed the hard QC filters (clipping, >30 s, duplicates, empty/unreadable removed; `low_level` kept).
Deployed as CTranslate2 INT8 (`results/deploy_lora/ct2/int8`). Seed 0 is the shipped checkpoint.

## Intended use
Research/prototyping of Swahili transcription of clean-ish, read-style speech on CPU, and as a
baseline for field-robustness work. **Not** for clinical, legal, safety or child-data decisions, and
not validated for conversational speech, other Swahili varieties (only Kenyan-read FLEURS), or
code-switched speech.

## Training data
FLEURS `sw_ke` train (see [dataset card](dataset_card.md)). Read Wikipedia-style sentences; ~13.5 h
before QC; no private data. Gender-balanced in train, 72% male in test.

## Evaluation (test split, 3 seeds, WER % / CER %)
| condition | WER | CER |
|---|---|---|
| clean | 21.4 +- 0.3 | 6.0 +- 0.3 |
| mp3_8k | 27.6 +- 0.8 | 8.4 +- 0.4 |
| pink noise 10 dB | 38.3 +- 0.4 | 14.3 +- 0.6 |
| babble 10 dB | 45.9 +- 4.0 | 19.8 +- 2.5 |
| AMR-NB 8 kHz | 50.2 +- 1.7 | 21.5 +- 1.4 |
| cheap mic + reverb | 51.8 +- 1.8 | 19.3 +- 1.1 |
| pink noise 0 dB | 96.3 +- 5.3 | 47.9 +- 2.5 |
| babble 0 dB | 141.6 +- 4.2 | 88.9 +- 3.4 |
| domain terms | 22.1 +- 1.0 | 6.8 +- 0.9 |
Female clean WER 16.0, male 23.4 (test is 136 F / 351 M clips; gap not explained, possibly speaker mix).
Zero-shot Whisper-small is 84.7% on clean, so the fine-tune is a large gain; it is still far from usable at
low SNR.

## CPU deployment
faster-whisper / CTranslate2, 8 CPU threads, 100 clips per condition:

| condition | WER fp32 | WER int8 | dWER | lat fp32 (s) | lat int8 (s) | RTF fp32 | RTF int8 |
|---|---|---|---|---|---|---|---|
| clean | 0.230 | 0.230 | -0.001 | 1.71 | 1.14 | 0.121 | 0.081 |
| babble_snr10 | 0.445 | 0.414 | -0.030 | 2.44 | 1.51 | 0.173 | 0.107 |
| amr_nb_8k | 0.413 | 0.411 | -0.002 | 2.37 | 1.55 | 0.168 | 0.110 |

Subsets of the test conditions, 100 clips each (WER here is on that subset, so it differs from the full-test table). Model size: INT8 242 MB vs FP32 928 MB. RTF = processing time / audio duration (0.08 = 12x faster than real time).

**Quantisation did not hurt accuracy**: WER differences are -0.1 to -3.0 points (INT8 is slightly *better* on some), which at n=100 per condition is within noise, not a real gain; hallucination-prone clips dominate the variance. Latency is single-stream on 8 CPU threads, greedy decoding, no VAD.


## Known failure modes
* **Insertions/hallucination at low SNR** (WER > 100% at babble 0 dB): repeated phrases or the
  background talker's speech. Use duration-based length caps / no-speech gating in production.
* English-origin names and loanwords are mangled or "translated" back to English.
* Numerals: inconsistent digit/word output; lost under noise.
* Word-boundary errors in agglutinative forms (WER penalises more than CER).
* Output is cased and punctuated; evaluation normalises case/punctuation on both sides.

## Limitations and ethics
Simulated degradations only; read speech only; no accent/age labels so no subgroup guarantees;
single dataset. Fine-tuned weights can memorise training sentences; voices are personal data (see
[data handling](data_handling.md)). The built-in PII scrubber is heuristic and not a
de-identification guarantee.
