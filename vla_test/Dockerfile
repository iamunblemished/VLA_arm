FROM python:3.10-slim

# Layer 1: System deps
RUN apt-get update && apt-get install -y \
    libgl1 libglx-mesa0 libglib2.0-0 gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Hardware stack (NPU Delegate requires TensorFlow)
RUN pip install --no-cache-dir tensorflow opencv-python-headless

# Layer 3: The Preprocessing stack (Tiny CPU-only Torch for the AutoProcessor)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Layer 4: The Tokenizer stack
RUN pip install --no-cache-dir transformers accelerate pillow requests numpy

WORKDIR /app
ENV TRANSFORMERS_CACHE=/app/model_cache
