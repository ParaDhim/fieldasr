# Review -> retraining round (simulated reviewer)

**Important: no human has corrected any transcript yet.** The reviewer here is an *oracle*: the
FLEURS reference text stands in for a corrected transcript. This validates the machinery and gives
an upper bound on what correcting N clips can do; it does not measure real reviewer effort or
real annotation noise.

## Pipeline (all real code, runs end to end)
1. `make_review_queue`: dev clips (211, speakers never in test) are degraded with random field
   augmentation (`augment.random_field_aug`), transcribed by the base model (Whisper-small + LoRA
   s0), ranked by item WER, worst-first -> `data/review/queue.csv` (path + model draft).
2. **Review app** (`make review`): reviewer plays the clip, edits the draft, flags bad audio,
   saves; "Export" writes PII-scrubbed `data/review/corrected.csv`
   (bad-audio/empty rows dropped). Tested: `tests/test_review_store.py`.
3. `fieldasr.train.retrain` continues fine-tuning the base on `corrections x3 + 200 replay`
   train clips (3 epochs, lr 2e-5) and re-runs all 9 eval conditions. It accepts the app's real
   export via `corrections:` in the config; rows with `pii_redacted=True` are excluded by default.

## Controlled comparison (base = Whisper-small LoRA s0; test-set WER %, mean +- std, 3 retrain seeds)
Three arms share identical replay data, so differences are due to the 60 added clips:
`replay_only` (no corrections), `random60` (60 random augmented dev clips), `targeted60` (the 60
worst failures).

| condition | base (LoRA s0) | replay only | random 60 | targeted 60 |
|---|---|---|---|---|
| clean | 21.0 | 22.0 +- 0.0 | 21.5 +- 0.6 | 21.3 +- 0.4 |
| pink_snr10 | 38.5 | 38.7 +- 0.1 | 35.8 +- 0.7 | 36.9 +- 1.6 |
| pink_snr0 | 96.1 | 86.0 +- 7.4 | 74.9 +- 2.2 | 72.0 +- 0.8 |
| babble_snr10 | 42.5 | 43.7 +- 3.1 | 36.7 +- 0.1 | 37.4 +- 0.1 |
| babble_snr0 | 137.8 | 114.6 +- 6.2 | 94.3 +- 3.9 | 103.4 +- 12.3 |
| amr_nb_8k | 48.6 | 46.9 +- 2.6 | 41.8 +- 3.2 | 41.0 +- 1.2 |
| mp3_8k | 27.0 | 27.9 +- 0.4 | 27.4 +- 0.1 | 26.9 +- 0.4 |
| cheap_mic_reverb | 51.0 | 51.8 +- 0.6 | 50.6 +- 0.4 | 49.5 +- 0.7 |
| domain_terms | 20.9 | 22.2 +- 0.2 | 21.4 +- 0.3 | 22.2 +- 0.9 |

## What this does and does not show
* **Adding ~60 field-condition clips helps under noise/codec:** vs the replay-only control,
  pink 0 dB 86.0 -> 72.0, AMR 46.9 -> 41.0, babble 10 dB 43.7 -> 37.4 (targeted).
* **Choosing the worst failures is not clearly better than choosing random clips.** Targeted is
  a bit better on pink 0 dB, AMR, cheap-mic, worse on babble 0 dB (103 +- 12 vs 94 +- 4); most
  gaps between the two arms are inside the seed spread. The gain is mostly *field-condition
  exposure*, not *failure selection*.
* **Continued training costs a little clean accuracy:** replay-only already moves clean 21.0 -> 22.0
  (the base was a single LoRA seed; the continued-FT recipe is not tuned).
* The base is one seed; the three seeds vary only the retraining data order/replay sample.
* The "corrections" are *augmented* dev audio, so this partly re-tests the augmentation finding
  from `whisper_full_aug` rather than the value of human correction per se.
* Numbers outside noise: 0 dB conditions stay near or above ~72% WER; nothing here makes them usable.
