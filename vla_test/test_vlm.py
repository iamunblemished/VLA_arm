import torch
import requests
import time
import psutil
import os
import sys
from io import BytesIO
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    TextStreamer
)

# Optimization for Dragon Q6A
torch.set_num_threads(8)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

class LayerTimer:
    def __init__(self):
        self.start_time = time.time()
        self.last_mark = time.time()

timer = LayerTimer()

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"--- [START] Initial RAM: {get_ram():.2f} MB ---")

# 1. Component Assembly (Use AutoProcessor to ensure internal sync)
# We pass do_image_splitting=False here to ensure the tokenizer knows the vision output size
processor = AutoProcessor.from_pretrained(MODEL_ID, do_image_splitting=False)

# 2. Load Model in float32
print("Loading model weights (FP32)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float32, 
    device_map="cpu"
)

# 3. Timing Hooks
def create_hook(name):
    def hook(module, input, output):
        curr_time = time.time()
        delta = curr_time - timer.last_mark
        total = curr_time - timer.start_time
        sys.stdout.write(f"  -> [PROGRESS] {name} done | +{delta:.2f}s | Total: {total:.2f}s | RAM: {get_ram():.1f}MB\n")
        sys.stdout.flush()
        timer.last_mark = curr_time
    return hook

print("Attaching timing hooks...")
for name, module in model.named_modules():
    if name.endswith(".layers") and isinstance(module, torch.nn.ModuleList):
        for i, layer in enumerate(module):
            if i % 5 == 0 or i == len(module) - 1:
                layer.register_forward_hook(create_hook(f"Layer_{i}"))
        break

def run_inference():
    # 4. Capture from your Vision Server
    url = "http://localhost:5000/capture"
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("Successfully captured frame from Vision Server.")
    except Exception as e:
        print(f"Capture failed: {e}. Using dummy image.")
        image = Image.new('RGB', (224, 224), color=(50, 50, 50))

    # 5. Correct Chat Template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Object name?"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # 6. Critical Fix: Pass do_image_splitting here as well
    inputs = processor(
        text=prompt, 
        images=[image], 
        return_tensors="pt",
        do_image_splitting=False 
    ).to(torch.float32)

    print(f"\n--- [PHASE 1] Pre-filling (224p | FP32) ---")
    
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
    timer.start_time = time.time()
    timer.last_mark = time.time()
    
    with torch.no_grad():
        model.generate(
            **inputs, 
            max_new_tokens=25, 
            do_sample=False, 
            streamer=streamer,
            use_cache=True
        )

def run_loop():
    print("\n--- [VLA MODE ACTIVE] Entering Control Loop ---")
    while True:
        try:
            # 1. Perception
            run_inference()
            
            # 2. Reasoning (Add logic here to map "glass" to a command)
            # e.g., if "glass" in output: send_to_arm("TARGET_GLASS")
            
            # 3. Frequency Control
            # Give the CPU a 100ms breather to prevent thermal throttling
            time.sleep(0.1) 
        except KeyboardInterrupt:
            print("\nShutting down VLA Brain...")
            break

if __name__ == "__main__":
    run_loop()
