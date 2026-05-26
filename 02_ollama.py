# Ollama — Easiest setup, manages model downloads and serving automatically
#
# INSTALL:
#   brew install ollama
#
# SETUP (run once, before this script):
#   ollama serve          # start the local server (or run Ollama.app)
#   ollama pull qwen3-vl:4b
#
# RUN:
#   python 02_ollama.py

import base64
import json
import urllib.request

MODEL  = "qwen3-vl:4b"
IMAGE  = "test.jpg"
PROMPT = "What is in this image? Describe it in detail."

with open(IMAGE, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

payload = json.dumps({
    "model": MODEL,
    "messages": [{
        "role": "user",
        "content": PROMPT,
        "images": [image_b64],
    }],
    "stream": False,
}).encode()

print(f"Sending request to Ollama ({MODEL}) ...")
req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())

print("\n" + result["message"]["content"])
