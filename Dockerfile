# Dockerfile cho TTS Serverless (Non-GPU)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy TTS modules và handler
COPY openai_tts.py .
COPY edge_tts.py .
COPY gemini_tts.py .
COPY tts_handler.py .

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from tts_handler import health_check; success, msg = health_check(); exit(0 if success else 1)"

# Run handler
CMD ["python", "tts_handler.py"]
