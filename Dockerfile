FROM python:3.11-slim

WORKDIR /app

# Install basic system tools (curl is useful for health checks)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install dependencies
RUN pip install uv
COPY pyproject.toml .
COPY uv.lock .
RUN uv sync

# Streamlit runs on 8501 by default
EXPOSE 8501

# Healthcheck to ensure Streamlit is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to launch the app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
