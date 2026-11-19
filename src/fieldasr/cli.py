"""fieldasr command line."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="FieldASR pipeline")


@app.callback()
def _root() -> None:
    """FieldASR pipeline."""


@app.command()
def prepare(raw: Path = Path("data/raw/fleurs_sw"), out: Path = Path("data/processed")) -> None:
    """Extract FLEURS audio, run QC, check split leakage, write manifests + QC report."""
    from fieldasr.data import fleurs, qc, splits

    audio = out / "audio"
    fleurs.extract(raw, audio)
    qc_df = qc.run_qc(fleurs.load_manifest(raw, audio))
    out.mkdir(parents=True, exist_ok=True)
    qc_df.assign(flags=qc_df["flags"].map(",".join)).to_csv(out / "manifest_qc.csv", index=False)
    summary = qc.summarize(qc_df)
    leak = splits.check_leakage(qc_df)
    (out / "qc_report.json").write_text(
        json.dumps(
            {
                "flag_counts": summary.to_dict(),
                "leakage": leak,
                "n": qc_df.groupby("split").size().to_dict(),
                "n_pass": qc_df.groupby("split")["qc_pass"].sum().astype(int).to_dict(),
            },
            indent=2,
        )
    )
    typer.echo(summary.to_string())
    typer.echo(f"leakage: {leak}")


@app.command("build-evalsets")
def build_evalsets(
    manifest: Path = Path("data/processed/manifest_qc.csv"),
    out: Path = Path("data/evalsets"),
) -> None:
    """Build noise / codec / cheap-mic+reverb / domain-term variants of the test split."""
    import pandas as pd

    from fieldasr.data import evalsets

    df = evalsets.build(pd.read_csv(manifest, keep_default_na=False), out)
    typer.echo(df.groupby("condition").size().to_string())


@app.command()
def transcribe(
    files: list[Path] = typer.Argument(..., help="audio files (wav/mp3/amr/...)"),
    model_dir: Path = typer.Option(Path("results/deploy/ct2/int8"), help="CTranslate2 model dir"),
    redact: bool = typer.Option(True, help="scrub names/phones/IDs from output"),
    threads: int = 4,
    jsonl: bool = typer.Option(False, help="emit one JSON object per file"),
) -> None:
    """Offline CPU transcription (no network, nothing stored)."""
    import json

    from fieldasr.deploy.infer import Transcriber

    tr = Transcriber(model_dir, threads=threads)
    for f in files:
        r = tr(f, redact=redact)
        typer.echo(
            json.dumps(Transcriber.to_dict(r), ensure_ascii=False)
            if jsonl
            else f"{f.name}\t{r.text}\t[RTF {r.rtf:.2f}]"
        )


if __name__ == "__main__":
    app()
