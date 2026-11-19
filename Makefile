.PHONY: setup test lint reproduce train extras report review api
setup:        ## pinned env (needs ffmpeg with libopencore_amrnb on PATH for the codec sets)
	uv sync --extra train --extra deploy --extra app
test:
	uv run pytest -q
lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy src
train:        ## data prep, eval sets, all training, CT2 export + CPU benchmark (resumable)
	bash scripts/run_all.sh
extras:       ## error analysis, privacy scan, simulated-reviewer retraining round
	bash scripts/run_extras.sh
reproduce: setup test train extras   ## every number in the README / docs
	uv run python -m fieldasr.aggregate
review:
	uv run python -m fieldasr.review.app --queue data/review/queue.csv
api:
	MODEL_DIR=results/deploy_lora/ct2/int8 uv run uvicorn fieldasr.deploy.api:app --port 8000
