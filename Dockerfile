# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install PocketBase
RUN curl -L https://github.com/pocketbase/pocketbase/releases/download/v0.20.0/pocketbase_0.20.0_linux_amd64.zip -o pocketbase.zip \
    && unzip pocketbase.zip \
    && chmod +x pocketbase \
    && mv pocketbase /usr/local/bin/ \
    && rm pocketbase.zip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create directories for models and data
RUN mkdir -p /app/data /app/logs /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_OLLAMA=true
ENV OLLAMA_MODEL=qwen2.5:14b-instruct
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose both ports (API and PocketBase)
EXPOSE 8000 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Create startup script inline
RUN echo 'import subprocess\n\
import time\n\
import os\n\
\n\
# Start PocketBase in background\n\
print("Starting PocketBase...")\n\
pocketbase = subprocess.Popen(["/usr/local/bin/pocketbase", "serve", "--http=0.0.0.0:8090", "--dir=/app/pb_data"])\n\
\n\
# Wait for PocketBase to be ready\n\
time.sleep(5)\n\
print("PocketBase started on port 8090")\n\
\n\
# Start main app\n\
print("Starting main application...")\n\
import app\n\
app.main()\n' > start_services.py

# Run both services
CMD ["python", "start_services.py"]