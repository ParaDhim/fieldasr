"""Shared training utilities: config, seeding, training-set loading, MLflow wrapper."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

BAD_FLAGS = {"clipping", "too_long", "unreadable", "duplicate_audio", "empty_transcript"}


def parse_args(desc: str) -> tuple[dict[str, Any], argparse.Namespace]:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests")
    ap.add_argument("--eval-limit", type=int, default=None, help="clips per condition")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--train-manifest", default="data/processed/manifest_qc.csv")
    ap.add_argument("--eval-manifest", default="data/evalsets/manifest.csv")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.model:
        cfg["model"] = a.model
    return cfg, a


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(manifest: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(manifest, keep_default_na=False)
    df = df[df.split == split]
    ok = df["flags"].map(lambda f: not (set(str(f).split(",")) & BAD_FLAGS))
    return df[ok & (df.duration <= 30)].reset_index(drop=True)


class Tracker:
    """MLflow wrapper that never kills a run if tracking fails."""

    def __init__(self, run_name: str, params: dict[str, Any]):
        self.ok = True
        try:
            import mlflow

            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("fieldasr")
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({k: str(v) for k, v in params.items()})
            self.mlflow = mlflow
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow disabled] {e}", flush=True)
            self.ok = False

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.ok:
            try:
                self.mlflow.log_metrics(
                    {k.replace("=", "_"): v for k, v in metrics.items()}, step=step
                )
            except Exception as e:  # noqa: BLE001
                print(f"[mlflow log failed] {e}", flush=True)

    def end(self) -> None:
        if self.ok:
            self.mlflow.end_run()


def log_eval(tracker: Tracker, metrics: dict) -> None:
    flat = {}
    for cond, m in metrics.items():
        flat[f"wer/{cond}"] = m["wer"]
        flat[f"cer/{cond}"] = m["cer"]
    tracker.log({k: v for k, v in flat.items() if v == v})
