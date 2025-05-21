# Development Dockerfile for Gradio Emotion App
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Gradio's default port
EXPOSE 7860

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0

# For development: allow running as root, but you can add a non-root user if desired
# USER appuser

# Run the app (do not use --reload, but you can mount code with -v for live changes)
CMD ["python", "app.py"] 