# Runtime image for Asymmetric expert system simulation (Random and Backpressure policies)
FROM python:3.11-slim

WORKDIR /app

# Copy project and install (pyproject.toml has runtime deps only for smaller image)
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Default: run CLI (override with e.g. docker run ... single --lam 0.8 --a 0.5 ...)
ENTRYPOINT ["python", "-m", "expert_sim.cli"]
CMD ["--help"]
