import qai_hub as hub

# 1. Connect to the client
client = hub.Client()

# 2. Grab your model
print("Fetching your uploaded model from the cloud...")
cloud_model = hub.get_model("mngyzkv0n")

# 3. Define the target board
device = hub.Device("Dragonwing RB3 Gen 2 Vision Kit")

# 4. Explicitly tell the compiler the dimensions AND the data types
input_specs = {
    "input_ids": ((1, 76), "int64"),
    "attention_mask": ((1, 76), "int64"),
    "pixel_values": ((1, 1, 3, 512, 512), "float32")
}

# --- THE FIX: Add the intermediate tensor flag (space-separated) ---
compiler_options = "--truncate_64bit_io --truncate_64bit_tensors"

print(f"Submitting compile job for {device.name} with options: {compiler_options}...")
compile_job = hub.submit_compile_job(
    model=cloud_model,
    device=device,
    input_specs=input_specs,
    options=compiler_options,
    name="SmolVLM_Hexagon_Final_Compile"
)

# 5. Wait for completion
print("Compiling on Qualcomm cloud... (This will take a few minutes)")
compile_job.wait()

# 6. Bulletproof success check: Just see if a target model exists!
target_model = compile_job.get_target_model()

if target_model is not None:
    print("Downloading final compiled model...")
    target_model.download("smolvlm_hexagon.elf")
    print("\n========================================================")
    print("Success! Your Hexagon model is saved as 'smolvlm_hexagon.elf'")
    print("========================================================")
else:
    print("\nCompile job failed. Status info:")
    print(compile_job.get_status())
    print("Check the logs on the AI Hub dashboard for detailed error messages.")
