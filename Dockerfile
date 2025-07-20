# Use Python 3.12 base image
FROM python:3.12-bookworm

# Install system dependencies for OpenCV/TensorFlow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Rust-based package manager)
RUN pip install uv && \
    uv venv -p python3.12 /opt/venv

# Set up environment
ENV PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Copy project files
WORKDIR /app

# First copy dependency files (for caching)
COPY pyproject.toml ./

# Install dependencies using uv (cached unless pyproject.toml changes)
RUN uv pip install -e . && \
    rm -rf /root/.cache/pip  # Alternative cache cleanup

# Now copy the rest of the application
COPY server.py .
COPY utils/ ./utils/
COPY model/ ./model/
COPY models/ ./models/

# Clean up Python cache
RUN find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /opt/venv -name '*.pyc' -delete

# Run as non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uv", "run", "server.py", "--host", "0.0.0.0", "--port", "8000"]
