# Data Copilot: Run and Test Instructions

These instructions assume commands are run from the repository root.

## Prerequisites

Install or make available:

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Node.js 20 or newer and npm
- Docker Desktop or another reachable Docker daemon

On WSL, Docker Desktop must have WSL integration enabled for the distro running
this project.

Verify the tools:

```bash
python --version
uv --version
node --version
npm --version
docker info
```

## One-time setup

Install the backend and frontend dependencies, create the local environment
file, and build the sandbox image:

```bash
uv sync
npm --prefix frontend ci
cp .env.example .env
docker build -t data-copilot-sandbox:latest sandbox
```

The default `.env` configuration uses `LLM_PROVIDER=mock`. This mode is
deterministic and does not require an API key. To use a real provider, install
the optional provider dependencies and edit `.env`:

```bash
uv sync --extra providers
```

Set `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`, or
`LLM_PROVIDER=openai` with `OPENAI_API_KEY`.

The backend creates the `data-copilot-sandbox` Docker network automatically on
the first real session. No manual network creation is required.

## Run the app locally

Start each service in a separate terminal.

### Terminal 1: backend

```bash
uv run uvicorn app.main:app \
  --app-dir backend \
  --host 127.0.0.1 \
  --port 8000
```

Check that it is ready:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

The interactive API documentation is available at
`http://127.0.0.1:8000/docs`.

### Terminal 2: frontend

Set the API URL when starting Vite. The frontend does not define a Vite
development proxy, so this environment variable is required when the backend
and frontend run on different ports:

```bash
cd frontend
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```

Open `http://localhost:5173`.

To use a different backend address, replace the value of `VITE_API_URL`. Restart
the Vite process after changing it.

## Manual smoke test

1. Open `http://localhost:5173`.
2. Upload `sample_data/sales.csv` using the left-hand explorer.
3. Ask **Build a dashboard**.
4. Confirm that the dashboard appears in the canvas.
5. Ask **Create a PDF whitepaper report**.
6. Confirm that the PDF appears and can be downloaded.
7. Inspect the execution events and generated artifacts in the workbench.

## Automated tests and validation

### Backend unit tests

The backend tests use `FakeSandbox`, so they do not require Docker:

```bash
uv run pytest -q
```

### Frontend checks

Run the TypeScript check and production build:

```bash
cd frontend
npm run typecheck
npm run build
cd ..
```

### Live smoke test

Keep the backend running in another terminal, then run:

```bash
uv run python scripts/smoke.py
```

With Docker available and the sandbox image built, this checks:

- backend health
- session creation and cleanup
- CSV upload and profiling as `df_1`
- dashboard generation and preview proxying
- PDF generation and download

The script uses `http://127.0.0.1:8000` by default. Override it when needed:

```bash
BACKEND_URL=http://127.0.0.1:8000 uv run python scripts/smoke.py
```

If the backend is unavailable, or if Docker is unavailable, the script reports
the relevant checks as skipped. A skipped Docker check is not a full
Docker-backed validation.

## Makefile shortcuts

From the repository root:

```bash
make install          # Install Python and frontend dependencies
make build-sandbox   # Build the Docker sandbox image
make test             # Run backend tests and frontend typecheck
make smoke            # Run the live smoke test
```

The equivalent service commands are:

```bash
make backend
make frontend
```

When using `make frontend`, set the API URL in the environment first:

```bash
VITE_API_URL=http://127.0.0.1:8000 make frontend
```

## Troubleshooting

### The frontend cannot create a session

Make sure the backend is running on port 8000 and restart Vite with:

```bash
VITE_API_URL=http://127.0.0.1:8000 npm --prefix frontend run dev
```

### Session creation returns a Docker error

Check Docker and rebuild the image:

```bash
docker info
docker build -t data-copilot-sandbox:latest sandbox
```

### A port is already in use

Stop the process using port 8000 or 5173, or start the corresponding service
on another port and update `VITE_API_URL` or `.env` accordingly.

### Provider authentication errors

The default provider is `mock`. If a real provider is selected in `.env`,
confirm that the matching API key is set and that the optional dependencies
were installed with `uv sync --extra providers`.
