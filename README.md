# Data Copilot

Container-native analytics copilot POC built from `PRD.txt`. It provides dataset
upload and profiling, a persistent IPython execution sandbox, deterministic
offline agent tooling, dashboard/PDF artifacts, preview proxying, and a React
workbench.

## Quick start

Requirements:

- Python 3.11+
- Node.js 20+
- Docker Desktop or another reachable Docker daemon

```bash
uv sync
cd frontend && npm install && cd ..
cp .env.example .env
docker build -t data-copilot-sandbox:latest sandbox
```

Start the gateway and UI in separate terminals:

```bash
uv run uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
cd frontend && npm run dev
```

Open `http://localhost:5173`. The default `LLM_PROVIDER=mock` mode is
deterministic and needs no API key. Set `LLM_PROVIDER=anthropic` or `openai`
and provide the corresponding key to use a provider adapter.

## Demo

1. Upload `sample_data/sales.csv` from the left explorer.
2. Ask **Build a dashboard**.
3. Ask **Create a PDF whitepaper report**.
4. Use the canvas tabs and execution inspector to inspect the generated results.

The live smoke test covers the same flow:

```bash
uv run python scripts/smoke.py
```

## Architecture

- `backend/app`: FastAPI gateway, session lifecycle, profiling, provider
  adapters, agent tool dispatch, and preview proxy.
- `sandbox`: Docker image and persistent `sandboxd` IPython/file/process service.
- `frontend`: Vite, React, TypeScript, Zustand, and the workbench UI.
- `sessions`: local JSONL trajectory logs and fake-sandbox workspaces.

Each Docker session gets its own volume and container with CPU, memory, PID,
capability, `no-new-privileges`, and `/tmp` restrictions. The default sandbox
bridge disables IP masquerading and publishes only ephemeral loopback ports
needed by the local gateway. Set `SANDBOX_ALLOW_EGRESS=true` only when outbound
network access is explicitly required.

## Validation

```bash
uv run pytest -q
cd frontend && npm run typecheck && npm run build
docker build -t data-copilot-sandbox:latest sandbox
```

The backend tests use `FakeSandbox`; the smoke test validates the real
Docker-backed session, upload/profile response, dashboard preview, PDF download,
and cleanup.
