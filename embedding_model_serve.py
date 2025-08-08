import os
import time
import subprocess
from threading import Thread

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pyngrok import ngrok
from transformers import AutoModel

# === 1. Install necessary packages ===
required_packages = ["fastapi", "transformers", "uvicorn", "pyngrok"]
for pkg in required_packages:
    subprocess.run(["pip", "install", "--upgrade", pkg])

# === 2. Constants ===
MAX_LENGTH = 8192  # max token length for input strings

# === 3. Load embedding model ===
model_name = "jinaai/jina-embeddings-v2-base-en"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# === 4. Create FastAPI app ===
app = FastAPI()

@app.post("/encode")
async def encode_string(request: Request):
    data = await request.json()
    input_string = data.get("input_string")
    
    if not input_string:
        raise HTTPException(status_code=400, detail="Invalid input: 'input_string' required.")
    
    # Truncate to MAX_LENGTH if too long
    if len(input_string) > MAX_LENGTH:
        input_string = input_string[:MAX_LENGTH]

    # Generate embedding vector
    embedding = model.encode(input_string)

    # Return JSON serializable list
    return {"result": embedding.tolist()}

# === 5. Setup ngrok ===
ngrok_token = os.getenv("NGROK_TOKEN", "")
if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
else:
    print("[Warning] No NGROK_TOKEN environment variable found, ngrok may have limitations.")

public_url = ngrok.connect(8001)
print("Public URL:", public_url)

# Save URL to file
with open("public_url_embedding.txt", "w") as f:
    f.write(str(public_url))

# === 6. Run FastAPI app in background thread ===
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8001)

thread = Thread(target=run_app)
thread.start()

# === 7. Keep process alive and print heartbeat ===
try:
    while True:
        time.sleep(60)
        print(f"[Heartbeat] API running at {public_url}")
except KeyboardInterrupt:
    print("Server stopped by user.")
