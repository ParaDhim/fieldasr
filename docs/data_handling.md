# Data-handling note

## What is stored (this repo)
| item | where | contains | notes |
|---|---|---|---|
| FLEURS audio + transcripts | `data/raw`, `data/processed` | public CC-BY read speech, volunteer voices | no personal data collected by us; voices are still biometric-ish |
| derived eval audio | `data/evalsets`, `data/dev_field` | degraded copies of the above | regenerable, safe to delete |
| model checkpoints | `results/*/model` | weights fine-tuned on the above | can memorise training transcripts |
| hypotheses | `results/*/hyps.csv` | model outputs + references | |
| review corrections | `data/review/corrections.jsonl` | **raw reviewer text (local only)** | append-only; may contain PII a reviewer typed |
| review export | `data/review/corrected.csv` | **PII-scrubbed** text + audio path | this is the only file meant to leave the reviewer's machine |
| experiment logs | `mlflow.db`, `logs/` | params, metrics, stdout | no audio, no transcripts |

## Inference service
`fieldasr transcribe` is fully offline. The API writes each upload to a temp file that is deleted
right after decoding; audio and transcripts are **not** persisted or logged; responses are
PII-scrubbed unless `?redact=false`. Put it behind TLS and authentication; neither is built in.

## Retention (proposed defaults; not enforced by code)
Raw field audio: delete after transcription/QC unless there is documented consent for retention
(max 90 days). Reviewer corrections log: delete after the export is validated. Scrubbed corrected
data and models: kept for the project; re-evaluate when consent terms change. Anything in
`data/review/corrections.jsonl` is raw and should be treated like the source audio.

## PII scrubbing (`fieldasr.privacy.scrub`)
Regex + gazetteer + honorific heuristics for phone numbers (TZ/KE/UG/RW formats), e-mails, URLs,
ID-like strings (long digit runs, mixed alphanumerics >= 6 chars) and names. Replaced with
`[PHONE] [EMAIL] [URL] [ID] [NAME]`.
* **Not a de-identification guarantee.** It is not an NER model. Name recall is limited to a
  ~45-name gazetteer + honorifics ("Bwana X", "Dkt. X", "Mama X"); other names, nicknames,
  locations, school names, dates of birth and free-text identifiers pass through.
* **Measured on FLEURS:** 62/3768 transcripts flagged (43 name, 21 ID hits). FLEURS has no
  private data, so these are all false positives (public figures like "John Grant", aircraft
  codes like "PNG CG4684"). **This corpus cannot measure recall**; recall evidence is only the
  unit tests (`tests/test_scrub.py`) on synthetic examples. Over-redaction of public figures is
  accepted on purpose.
* Scrubbing text does nothing for the **audio**, which still carries the speaker's voice and any
  names spoken in it.

## Risks to flag for health or child data
* **Voice is personal data.** Speaker identity is recoverable from audio and from model
  weights/embeddings; children's voices are a special category in most regimes (GDPR art. 8,
  COPPA, Kenya DPA 2019, Tanzania PDPA 2022). Parental/guardian consent and a lawful basis are
  needed *before* collection, not after.
* **Health context makes transcripts sensitive** (diagnoses, medications, HIV status, names of
  patients) and name-scrubbing will miss indirect identifiers. Use a trained NER + human review
  before any sharing; avoid sending audio to third-party APIs.
* **Model memorisation:** fine-tuned Whisper can regurgitate training sentences; do not publish
  weights trained on non-public health or child audio.
* **Annotation exposes data to reviewers:** reviewers hear raw audio. Use named, access-controlled,
  confidentiality-bound reviewers; do not run the review app on a public interface (it binds to
  127.0.0.1 by default).
* **Cross-border transfer and retention limits** apply to cloud GPUs (Colab/Kaggle/rented boxes):
  do not put non-public audio there without a data-processing agreement.
* **Bias:** gender-skewed or accent-skewed evaluation hides failures on under-represented groups;
  FLEURS test is 72% male and has no accent or age labels.
