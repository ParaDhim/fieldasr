"""Convert a fine-tuned HF Whisper dir to CTranslate2 (FP32 and INT8)."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperProcessor

    # transformers>=5 folds the feature-extractor config into processor_config.json;
    # ctranslate2 / faster-whisper expect a standalone preprocessor_config.json.
    WhisperProcessor.from_pretrained(a.model_dir).feature_extractor.save_pretrained(a.model_dir)

    for q in ("float32", "int8"):
        dst = Path(a.out_dir) / q
        if (dst / "model.bin").exists():
            continue
        TransformersConverter(
            a.model_dir, copy_files=["tokenizer.json", "preprocessor_config.json"]
        ).convert(str(dst), quantization=q, force=True)
        print(f"exported {q} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
