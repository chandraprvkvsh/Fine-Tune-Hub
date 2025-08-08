import os
import time
import subprocess
from threading import Thread

import torch
from fastapi import FastAPI, Request
from pyngrok import ngrok
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn

# === 1. Install Required Packages ===
required_packages = [
    "fastapi", "transformers", "uvicorn", "accelerate",
    "pyngrok", "bitsandbytes"
]
for pkg in required_packages:
    subprocess.run(["pip", "install", "--upgrade", pkg])

# === 2. Authenticate with Hugging Face ===
hf_token = os.getenv("HF_TOKEN", "")
if hf_token:
    os.system(f'huggingface-cli login --token "{hf_token}"')
else:
    print("[Warning] No Hugging Face token set in environment variable 'HF_TOKEN'.")

# === 3. Load Model and Tokenizer ===
BASE_MODEL = os.getenv("BASE_MODEL", "your-org/your-model-name")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# === 4. Helper: Clean Output ===
def remove_after_tag(text, tag="</s>"):
    return text.split(tag)[0] if tag in text else text

# === 5. FastAPI App ===
app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    context = data.get("context", "")
    abstract = data.get("abstract", "")
    
    instruction = (
        "Craft an intelligent, clear, insightful, and succinct one-line title "
        "for the research paper, drawing inspiration from the context and abstract provided.\n"
    )
    
    prompt = f"<s>[INST] {instruction}\nContext: {context}\nAbstract: {abstract} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = decoded.split("[/INST]")[-1].strip()
    return {"result": remove_after_tag(result)}

# === 6. Launch with ngrok ===
ngrok_token = os.getenv("NGROK_TOKEN", "")
if ngrok_token:
    ngrok.set_auth_token(ngrok_token)
else:
    print("[Warning] No ngrok token found in 'NGROK_TOKEN'. You may hit connection limits.")

public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# Optionally write public URL to a file
with open("public_url.txt", "w") as f:
    f.write(str(public_url))

# === 7. Run FastAPI in Background ===
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = Thread(target=run_app)
thread.start()

# === 8. Keep Alive Loop ===
try:
    while True:
        time.sleep(60)
        print(f"[Heartbeat] Server running at {public_url}")
except KeyboardInterrupt:
    print("Server stopped by user.")
