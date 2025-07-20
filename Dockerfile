# Use Python 3.12 base image
FROM python:3.12-bookworm

# Install uv (Rust-based package manager)
RUN pip install uv && \
    uv venv -p python3.12 /opt/venv

# Set up environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
WORKDIR /app

# First copy dependency files (for caching)
COPY pyproject.toml ./

# Install dependencies using uv (cached unless pyproject.toml changes)
RUN uv pip install -e .

# Now copy the rest of the application
COPY server.py .
COPY utils/ ./utils/
COPY model/ ./model/
COPY models/ ./models/

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uv", "run", "server.py", "--host", "0.0.0.0", "--port", "8000"]
