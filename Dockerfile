FROM python:3.11-slim

# Create non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies first (cache-friendly layer)
RUN pip install --no-cache-dir uv
COPY --chown=user pyproject.toml ./
RUN uv pip install --system --no-cache \
        pydantic fastapi uvicorn openenv-core openai requests

# Copy project files
COPY --chown=user server/ ./server/
COPY --chown=user env.py models.py inference.py openenv.yaml README.md ./

USER user

# Hugging Face Spaces port
EXPOSE 7860

# Start the OpenEnv server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
