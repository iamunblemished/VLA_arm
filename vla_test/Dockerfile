FROM python:3.10-slim

# Layer 1: System deps
RUN apt-get update && apt-get install -y \
    libgl1 libglx-mesa0 libglib2.0-0 gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: The heavy hitters (Torch first)
# Using --no-cache-dir is MANDATORY here
RUN pip install --no-cache-dir torch torchvision

# Layer 3: The rest of the stack
RUN pip install --no-cache-dir transformers accelerate pillow requests numpy

WORKDIR /app
ENV TRANSFORMERS_CACHE=/app/model_cache
