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
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project folders into the container
COPY --chown=user . .

# Ensure the app can find the 'backend' package and 'models' folder
ENV PYTHONPATH=/app

# Streamlit/HuggingFace Spaces port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]