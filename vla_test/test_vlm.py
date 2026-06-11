import cv2
import numpy as np
import requests
import time
from PIL import Image
from transformers import AutoProcessor
import tflite_runtime.interpreter as tflite

# Configuration
SERVER_URL = "http://127.0.0.1:5000/frame"
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
TFLITE_MODEL_PATH = "smolvlm_hexagon.elf.tflite"

print("1. Loading Hugging Face Processor (for Tokenization)...")
# We still need the processor to turn text into numbers and numbers back to text
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("2. Loading TFLite Model onto Qualcomm Hexagon NPU...")
try:
    # Attempt to load the QNN hardware delegate
    qnn_delegate = tflite.load_delegate('libQnnTFLiteDelegate.so')
    interpreter = tflite.Interpreter(
        model_path=TFLITE_MODEL_PATH, 
        experimental_delegates=[qnn_delegate]
    )
    print("SUCCESS: QNN Hardware Delegate attached!")
except Exception as e:
    print(f"WARNING: Could not load QNN delegate ({e}).")
    print("Ensure libQnnTfLiteDelegate.so is in your LD_LIBRARY_PATH.")
    print("Falling back to CPU TFLite execution for testing...")
    interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model memory successfully allocated on NPU!")

def get_frame():
    """Fetches the latest frame from the vision server."""
    try:
        response = requests.get(SERVER_URL)
        if response.status_code == 200:
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        else:
            print("Failed to get frame from server.")
            return None
    except requests.exceptions.ConnectionError:
        print("Vision server is not running.")
        return None

def process_frame(frame):
    """Passes the frame to the Hexagon NPU and prints the result."""
    # Convert BGR (OpenCV) to RGB (PIL)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 1. Format the Prompt
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the main object in this image."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    # Preprocess inputs
    inputs = processor(text=prompt, images=[image], return_tensors="np", do_image_splitting=False)

    # 2. Enforce Strict Hardware Constraints (Shapes & Data Types)
    # The Hexagon compiler requires exact dimensions and int32 types
    MAX_TOKENS = 76
    
    input_ids = np.zeros((1, MAX_TOKENS), dtype=np.int32)
    attention_mask = np.zeros((1, MAX_TOKENS), dtype=np.int32)
    
    # Copy our prompt into the fixed-size buffer
    seq_len = min(inputs["input_ids"].shape[1], MAX_TOKENS)
    input_ids[0, :seq_len] = inputs["input_ids"][0, :seq_len].astype(np.int32)
    attention_mask[0, :seq_len] = inputs["attention_mask"][0, :seq_len].astype(np.int32)
    
    # Process Image to exactly (1, 1, 3, 512, 512)
    pixel_values = inputs["pixel_values"].astype(np.float32)
    if len(pixel_values.shape) == 4:
        pixel_values = np.expand_dims(pixel_values, axis=1)

    # 3. Dynamic Input Mapping
    # TFLite scrambles input index order, so we dynamically match them by shape
    idx_input_ids = None
    idx_attn_mask = None
    idx_pixel_vals = None
    
    for detail in input_details:
        shape = tuple(detail['shape'])
        if shape == (1, MAX_TOKENS):
            if 'mask' in detail['name'].lower() or 'attention' in detail['name'].lower():
                idx_attn_mask = detail['index']
            else:
                idx_input_ids = detail['index']
        elif len(shape) == 5:
            idx_pixel_vals = detail['index']

    # 4. Generate Text via NPU Loop
    print("\n--- Starting NPU Inference ---")
    generated_tokens = []
    
    # We will generate up to 15 tokens to test it
    for step in range(15):
        if seq_len >= MAX_TOKENS:
            break # Context window is full
            
        # Push variables to the NPU
        interpreter.set_tensor(idx_input_ids, input_ids)
        interpreter.set_tensor(idx_attn_mask, attention_mask)
        interpreter.set_tensor(idx_pixel_vals, pixel_values)
        
        # Fire the Hexagon DSP!
        start_time = time.time()
        interpreter.invoke()
        npu_time = (time.time() - start_time) * 1000
        
        # Pull output logits back from the NPU
        logits = interpreter.get_tensor(output_details[0]['index'])
        
        # Find the most likely next word
        next_token_id = np.argmax(logits[0, seq_len - 1, :])
        generated_tokens.append(next_token_id)
        
        print(f"Token {step+1} generated in {npu_time:.1f}ms")
        
        # Append the new word to our prompt for the next loop
        input_ids[0, seq_len] = next_token_id
        attention_mask[0, seq_len] = 1
        seq_len += 1

    # 5. Decode the final sentence
    output_text = processor.decode(generated_tokens, skip_special_tokens=True)
    print("\n--- Final VLM Output ---")
    print(output_text)
    print("------------------------\n")

if __name__ == "__main__":
    print("Fetching frame from vision server...")
    frame = get_frame()
    if frame is not None:
        print("Successfully captured frame from Vision Server.")
        process_frame(frame)
    else:
        print("No frame to process.")
