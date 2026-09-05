.PHONY: install install-all dev test lint typecheck run compose-up compose-down validate

install:
	python -m pip install -e .

install-all:
	python -m pip install -e '.[asr,tts-chatterbox,training,dev]'

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

compose-up:
	docker compose up --build

compose-down:
	docker compose down
