FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SANDBOX_WORKSPACE=/workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 shared-mime-info \
    fonts-dejavu && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    pandas polars numpy scipy pyarrow openpyxl xlrd \
    streamlit dash nicegui fastapi uvicorn bottle \
    typst weasyprint reportlab python-pptx matplotlib seaborn plotly kaleido \
    ipykernel jupyter_client
WORKDIR /app
COPY sandboxd /app/sandboxd
RUN mkdir -p /workspace && useradd --create-home --uid 10001 sandbox \
    && chown -R sandbox:sandbox /workspace /app
USER sandbox
EXPOSE 7000 8000 8080 8501 8502
CMD ["uvicorn", "sandboxd.server:app", "--host", "0.0.0.0", "--port", "7000"]
