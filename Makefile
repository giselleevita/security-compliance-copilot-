.PHONY: run test eval lint docker-run docker-down

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

eval:
	python evals/run_eval.py

lint:
	ruff check app/ tests/ evals/

docker-run:
	docker compose up --build

docker-down:
	docker compose down
