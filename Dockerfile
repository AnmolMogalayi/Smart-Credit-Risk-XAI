# Use Python 3.13 to match your local dev environment
FROM python:3.13-slim

# System deps for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — HuggingFace requirement
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies (using Docker cache for speed)
COPY --chown=user backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project folders into the container
COPY --chown=user backend/           ./backend/
COPY --chown=user models/            ./models/
COPY --chown=user data/              ./data/
COPY --chown=user index.html         ./frontend/index.html

# Ensure the app can find the 'backend' package and 'models' folder
ENV PYTHONPATH=/app

# Move into the backend folder to start the server
WORKDIR /app/backend

# HuggingFace Spaces MUST use port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]