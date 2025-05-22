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

# Run application
CMD ["python", "app.py"] 
