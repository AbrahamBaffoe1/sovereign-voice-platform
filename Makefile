DATA_ROOT ?= data/bootstrap
ARTIFACTS_ROOT ?= artifacts/bootstrap
EXPERIMENTS_ROOT ?= artifacts/experiments/asr
MIN_FREE_GB ?= 0
LANGUAGE ?= all
TASK ?= both

.PHONY: install install-all data-install dev test lint typecheck run compose-up compose-down validate corpus-plan corpus-v0 corpus-v0-strict asr-plan asr-baseline asr-baseline-strict

install:
	python -m pip install -e .

install-all:
	python -m pip install -e '.[asr,tts-chatterbox,data,training,training-asr,dev]'

data-install:
	python -m pip install -e '.[data]'

dev:
	python -m pip install -e '.[dev]'

run:
	PYTHONPATH=services/api uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

test:
	pytest -q

lint:
	ruff check .

typecheck:
	mypy services/api/app training

validate:
	python -m compileall -q services training tests scripts

corpus-plan:
	python -m training.data.bootstrap --language $(LANGUAGE) --task $(TASK) --include-eval --data-root $(DATA_ROOT) --artifacts-root $(ARTIFACTS_ROOT) --min-free-gb $(MIN_FREE_GB) --dry-run

corpus-v0:
	python -m training.data.bootstrap --language $(LANGUAGE) --task $(TASK) --include-eval --data-root $(DATA_ROOT) --artifacts-root $(ARTIFACTS_ROOT) --min-free-gb $(MIN_FREE_GB)

corpus-v0-strict:
	python -m training.data.bootstrap --language $(LANGUAGE) --task $(TASK) --require-eval --data-root $(DATA_ROOT) --artifacts-root $(ARTIFACTS_ROOT) --min-free-gb $(MIN_FREE_GB)

asr-plan:
	python -m training.asr.run_baseline --language $(LANGUAGE) --artifacts-root $(ARTIFACTS_ROOT) --output-root $(EXPERIMENTS_ROOT)

asr-baseline:
	python -m training.asr.run_baseline --language $(LANGUAGE) --artifacts-root $(ARTIFACTS_ROOT) --output-root $(EXPERIMENTS_ROOT) --execute

asr-baseline-strict:
	python -m training.asr.run_baseline --language $(LANGUAGE) --artifacts-root $(ARTIFACTS_ROOT) --output-root $(EXPERIMENTS_ROOT) --require-external-eval --execute

compose-up:
	docker compose up --build

compose-down:
	docker compose down
