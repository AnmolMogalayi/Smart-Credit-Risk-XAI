# Use the latest Python 3.13 image
FROM python:3.13-slim

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Setup a non-root user (good for HuggingFace/Cloud)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install requirements first (to use Docker cache)
COPY --chown=user backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of your app folders
COPY --chown=user backend/ ./backend/
COPY --chown=user models/ ./models/
COPY --chown=user data/ ./data/

# Set working directory to backend so uvicorn finds main.py
WORKDIR /app/backend

# Expose the port (7860 is standard for AI spaces)
EXPOSE 7860

# Command to start your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]