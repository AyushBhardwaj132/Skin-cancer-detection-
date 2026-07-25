# Production Dockerfile for ISIC 2024 Skin Cancer Detection API & Web Application
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command starts FastAPI backend server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
