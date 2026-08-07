.PHONY: install backend frontend test build-sandbox smoke

install:
	uv sync
	cd frontend && npm install

backend:
	uv run uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev -- --host 0.0.0.0

test:
	uv run pytest -q
	cd frontend && npm run typecheck

build-sandbox:
	docker build -t $${SANDBOX_IMAGE:-data-copilot-sandbox:latest} sandbox

smoke:
	uv run python scripts/smoke.py
