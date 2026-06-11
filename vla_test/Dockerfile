FROM python:3.10-slim

# Layer 1: System deps
RUN apt-get update && apt-get install -y \
    libgl1 libglx-mesa0 libglib2.0-0 gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: The hardware stack (Swapped tflite-runtime for tensorflow)
RUN pip install --no-cache-dir tensorflow opencv-python-headless

# Layer 3: The tokenizer stack (Pinned numpy to <2)
RUN pip install --no-cache-dir transformers accelerate pillow requests "numpy<2"

WORKDIR /app
ENV TRANSFORMERS_CACHE=/app/model_cache
