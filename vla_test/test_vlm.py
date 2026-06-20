import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
import tensorflow as tf

SERVER_URL = "http://localhost:5000/capture"
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
TFLITE_MODEL_PATH = "smolvlm_hexagon.elf.tflite"

print("1. Loading Tokenizer and Image Processor manually...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# trust_remote_code forces the download of the SmolVLM-specific image code
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print("2. Loading TFLite Model onto Qualcomm Hexagon NPU...")
try:
    qnn_delegate = tf.lite.experimental.load_delegate('libQnnTFLiteDelegate.so')
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, experimental_delegates=[qnn_delegate])
    print("SUCCESS: QNN Hardware Delegate attached!")
except Exception as e:
    print(f"Delegate rejected. Falling back to CPU TFLite execution...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_frame():
    try:
        response = requests.get(SERVER_URL, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("Successfully captured frame from Vision Server.")
        return image
    except Exception as e:
        print(f"Capture failed: {e}")
        return None

def process_frame(image):
    # 1. Format the Prompt
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the main object in this image."}]}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # 2. Process text and image separately to avoid the library crash
    text_inputs = tokenizer(prompt, return_tensors="np")
    image_inputs = image_processor(images=image, return_tensors="np")

    # Extract inputs
    raw_input_ids = text_inputs["input_ids"].astype(np.int32)
    raw_attention_mask = text_inputs["attention_mask"].astype(np.int32)
    pixel_values = image_inputs["pixel_values"].astype(np.float32)

    if len(pixel_values.shape) == 4:
        pixel_values = np.expand_dims(pixel_values, axis=1)

    # 3. Fit into the NPU's compiled memory block (Increased to 128)
    MAX_TOKENS = 128
    input_ids = np.zeros((1, MAX_TOKENS), dtype=np.int32)
    attention_mask = np.zeros((1, MAX_TOKENS), dtype=np.int32)
    
    seq_len = min(raw_input_ids.shape[1], MAX_TOKENS)
    input_ids[0, :seq_len] = raw_input_ids[0, :seq_len]
    attention_mask[0, :seq_len] = raw_attention_mask[0, :seq_len]

    # Map Tensors
    idx_input_ids, idx_attn_mask, idx_pixel_vals = None, None, None
    for detail in input_details:
        shape = tuple(detail['shape'])
        if len(shape) == 5:
            idx_pixel_vals = detail['index']
        elif 'mask' in detail['name'].lower() or 'attention' in detail['name'].lower():
            idx_attn_mask = detail['index']
        else:
            idx_input_ids = detail['index']

    print("\n--- Starting NPU Inference ---")
    generated_tokens = []
    
    # Increased loop from 15 to 30 to allow full sentence generation
    for step in range(30):
        if seq_len >= MAX_TOKENS:
            break
            
        interpreter.set_tensor(idx_input_ids, input_ids)
        interpreter.set_tensor(idx_attn_mask, attention_mask)
        interpreter.set_tensor(idx_pixel_vals, pixel_values)
        
        start_time = time.time()
        interpreter.invoke()
        npu_time = (time.time() - start_time) * 1000
        
        logits = interpreter.get_tensor(output_details[0]['index'])
        next_token_id = np.argmax(logits[0, seq_len - 1, :])
        generated_tokens.append(next_token_id)
        
        print(f"Token {step+1} generated in {npu_time:.1f}ms")
        
        input_ids[0, seq_len] = next_token_id
        attention_mask[0, seq_len] = 1
        seq_len += 1

    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("\n--- Final VLM Output ---")
    print(output_text)
    print("------------------------\n")

if __name__ == "__main__":
    frame = get_frame()
    if frame is not None:
        process_frame(frame)
