# ============================================================================
# Kaggle Backend — BioMistral-7B GPU Inference Server
# ============================================================================
# Run this in a Kaggle notebook with GPU enabled.
# It loads BioMistral-7B on GPU and exposes a REST API via ngrok.
#
# SETUP (run these cells first in Kaggle):
#   Cell 1:  !pip install flask pyngrok transformers accelerate bitsandbytes torch
#   Cell 2:  Paste and run this entire script
#
# After running, you'll get a public ngrok URL like:
#   https://xxxx-xx-xx.ngrok-free.app
# Copy that URL and set it as KAGGLE_API_URL in your HF Space secrets.
# ============================================================================

import os, json, threading
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# ── Load BioMistral-7B with 4-bit quantization (fits in T4 16GB) ──
print("Loading BioMistral-7B (4-bit quantization)...")

MODEL_ID = "BioMistral/BioMistral-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"✅ BioMistral-7B loaded on {model.device}!")

# ── Flask API ──
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_ID, "device": str(model.device)})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 120)
    temperature = data.get("temperature", 0.7)
    do_sample = data.get("do_sample", True)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use chat template
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=do_sample and temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        return jsonify({"generated_text": text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Start with ngrok tunnel ──
def run_flask():
    app.run(port=5000)

# Start Flask in background thread
thread = threading.Thread(target=run_flask, daemon=True)
thread.start()

# Set up ngrok tunnel
from pyngrok import ngrok

# Set your ngrok authtoken (get free one at https://dashboard.ngrok.com/get-started/your-authtoken)
# Uncomment and replace with your token:
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")

public_url = ngrok.connect(5000)
print("\n" + "=" * 60)
print(f"🚀 BioMistral-7B API is live!")
print(f"📡 Public URL: {public_url}")
print(f"=" * 60)
print(f"\n👉 Set this as KAGGLE_API_URL in your HF Space secrets:")
print(f"   {public_url}")
print(f"\n🔗 Test: {public_url}/health")
print(f"=" * 60)

# Keep the notebook alive
import time
while True:
    time.sleep(60)
    print(f"[{time.strftime('%H:%M:%S')}] Server running at {public_url}")
