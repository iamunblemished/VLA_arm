import qai_hub as hub

# Initialize the client
client = hub.Client()

print("Uploading model... (This may take a while, please be patient)")
# Upload the file directly
model = client.upload_model("smolvlm_exported.pt2", name="SmolVLM_Manual_Upload")

print(f"Upload successful! Model ID: {model.model_id}")
print("You can now find this model in your Workbench project.")
