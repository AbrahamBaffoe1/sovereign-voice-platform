DATA_ROOT ?= data/bootstrap
ARTIFACTS_ROOT ?= artifacts/bootstrap
EXPERIMENTS_ROOT ?= artifacts/experiments/asr
TTS_READINESS_ROOT ?= artifacts/tts-readiness
EXECUTION_ROOT ?= /srv/sovereign-voice
MIN_FREE_GB ?= 0
LANGUAGE ?= all
TASK ?= both
PYTHON_BIN ?= python3.11
ASR_VENV ?= .venvs/asr
CHATTERBOX_VENV ?= .venvs/tts-chatterbox

.PHONY: install install-asr install-tts-chatterbox data-install dev test lint typecheck run compose-up compose-down validate corpus-plan corpus-v0 corpus-v0-strict asr-plan asr-baseline asr-baseline-strict tts-readiness tts-readiness-strict real-execution real-execution-strict

install:
	python -m pip install -e .

install-asr:
	$(PYTHON_BIN) -m venv $(ASR_VENV)
	$(ASR_VENV)/bin/python -m pip install --upgrade pip wheel
	PIP_CONSTRAINT=constraints/training-cu124.txt $(ASR_VENV)/bin/python -m pip install -e '.[asr,data,training,training-asr,dev]'
	$(ASR_VENV)/bin/python -m pip check

install-tts-chatterbox:
	$(PYTHON_BIN) -m venv $(CHATTERBOX_VENV)
	$(CHATTERBOX_VENV)/bin/python -m pip install --upgrade pip wheel
	PIP_CONSTRAINT=constraints/chatterbox-cu124.txt $(CHATTERBOX_VENV)/bin/python -m pip install -e '.[tts-chatterbox]'
	$(CHATTERBOX_VENV)/bin/python -m pip check

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
	python -m training.asr.run_baseline --language $(LANGUAGE) --artifacts-root $(ARTIFACTS_ROOT) --output-root $(EXPERIMENTS_ROOT) --resume --execute

asr-baseline-strict:
	python -m training.asr.run_baseline --language $(LANGUAGE) --artifacts-root $(ARTIFACTS_ROOT) --output-root $(EXPERIMENTS_ROOT) --require-external-eval --resume --execute

tts-readiness:
	python -m training.tts.readiness --language $(LANGUAGE) --artifacts-root $(ARTIFACTS_ROOT) --output-root $(TTS_READINESS_ROOT)

tts-readiness-strict:
	python -m training.tts.readiness --language $(LANGUAGE) --artifacts-root $(ARTIFACTS_ROOT) --output-root $(TTS_READINESS_ROOT) --strict

real-execution:
	python -m training.execution.run_pipeline --workspace $(EXECUTION_ROOT) --language $(LANGUAGE) --min-free-gb $(MIN_FREE_GB)

real-execution-strict:
	python -m training.execution.run_pipeline --workspace $(EXECUTION_ROOT) --language $(LANGUAGE) --min-free-gb $(MIN_FREE_GB) --require-external-eval

compose-up:
	docker compose up --build

compose-down:
	docker compose down
