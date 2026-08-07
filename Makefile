.PHONY: install backend frontend test build-sandbox smoke e2e

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

# Playwright end-to-end: drives the real UI against a mock-provider backend
# with real Docker sandboxes. Starts its own servers on ports 8100/5174.
e2e:
	cd frontend && npm run test:e2e

# Same suite plus a live-provider run against the real OpenAI model
# (consumes API credits; reads OPENAI_API_KEY from .env).
e2e-live:
	cd frontend && E2E_LIVE=1 npm run test:e2e
