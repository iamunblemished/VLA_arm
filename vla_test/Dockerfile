FROM python:3.10-slim

# Layer 1: System deps (Kept the same)
RUN apt-get update && apt-get install -y \
    libgl1 libglx-mesa0 libglib2.0-0 gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: The hardware stack
# Dropped Torch entirely! Added TFLite and OpenCV
RUN pip install --no-cache-dir tflite-runtime opencv-python-headless

# Layer 3: The tokenizer stack
RUN pip install --no-cache-dir transformers accelerate pillow requests numpy

WORKDIR /app
ENV TRANSFORMERS_CACHE=/app/model_cache
