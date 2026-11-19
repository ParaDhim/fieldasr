# Dataset card: FieldASR evaluation + training data (Swahili)

## Source
[Google FLEURS](https://huggingface.co/datasets/google/fleurs), config `sw_ke` (Swahili, Kenya).
CC-BY-4.0. Read speech: ~2.5k Wikipedia-derived sentences (FLoRes) read aloud by volunteers.
We did **not** collect any data. Common Voice was the first choice but is no longer served from
the HF hub.

| split | clips | hours | female / male clips | QC-pass clips |
|---|---|---|---|---|
| train | 3070 | 13.5 | 1599 / 1471 | 2753 |
| dev | 211 | 0.8 | 59 / 152 | 157 |
| test | 487 | 1.9 | 136 / 351 | 389 |

## Splits and leakage
FLEURS has **no speaker IDs**, so we cannot construct or audit speaker-disjoint splits ourselves.
We rely on the official splits (documented by the FLEURS authors as speaker-disjoint) and check
what we can: exact transcript overlap and audio-hash overlap between every split pair = **0**
(`data/processed/qc_report.json`). Speaker-embedding leakage checks are *not* done (limitation).
Note the test split is 72% male clips.

## QC (`fieldasr prepare`)
Flags per clip: `clipping` (>0.1% samples at full scale), `too_short`/`too_long` (<1 s, >30 s),
`mostly_silence` (gain-relative: >70% frames 35 dB below the clip's loud level), `low_level`
(peak < 0.01), `low_snr` (percentile SNR < 10 dB), `empty_transcript`, `duplicate_audio`.
Found: clipping 73 (train only), too_long 72 train + 1 dev, low_level 172/54/98, low_snr 1,
duplicate_audio 0. Training excludes clipping, too_long, duplicates, empty and unreadable clips (2925 train clips,
12.4 h remain; `low_level` is informational and kept, so this exceeds the strict all-flags-pass count of 2753).
`multi_reader` (same sentence read by several speakers) is informational: it is normal in FLEURS.
Lesson: an absolute -50 dB silence threshold flagged ~12% of train clips falsely, because
recording gain varies by ~40 dB across clips; a gain-relative threshold fixed it.

## Field-condition evaluation sets (`fieldasr build-evalsets`)
Derived from the 487 test clips (all <= 30 s), seeded and deterministic:
`clean`; `pink_snr10`, `pink_snr0` (pink noise); `babble_snr10`, `babble_snr0` (4 overlapping
other-speaker utterances); `amr_nb_8k` (AMR-NB 7.4 kbps @ 8 kHz via ffmpeg/opencore round trip);
`mp3_8k` (16 kbps MP3 @ 8 kHz); `cheap_mic_reverb` (synthetic RIR RT60 0.6 s, 250-5500 Hz
band-pass, tanh clipping, 10-bit quantisation, hiss at 30 dB SNR); `domain_terms` (121 clean
clips whose text has numerals, dates, health or school vocabulary; keyword/regex selected).

### Limitations (read before citing numbers)
* **Synthetic degradation.** Noise is synthetic pink/babble, reverb is a simulated RIR, the "cheap
  mic" is a filter chain. Real handset audio will differ; these sets are a stress test, not a
  proxy for deployment WER.
* **Read speech, not conversation.** Wikipedia-style sentences, careful prosody, many
  English-origin proper nouns. Expect worse WER on spontaneous speech.
* **Domain subset is a text filter,** not health/school speech. It is read encyclopaedic text that
  happens to contain these words (121 clips; subtags are small).
* **No accent or child-speech labels** exist in FLEURS; we slice by gender only. No child-speech
  subset was built.
* Babble is made from the same corpus's speakers (a model can latch onto the interfering talker).
