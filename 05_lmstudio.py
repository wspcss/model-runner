# LM Studio — OpenAI-compatible local endpoint
#
# INSTALL:
#   pip install openai
#
# SETUP (run once, before this script):
#   1. Download LM Studio from https://lmstudio.ai
#   2. In LM Studio, search for and download: Qwen/Qwen3-VL-4B-Instruct-GGUF
#   3. Load the model and start the local server (Local Server tab → Start Server)
#      Default endpoint: http://localhost:1234/v1
#
# RUN:
#   python 05_lmstudio.py

import base64
from openai import OpenAI

MODEL  = "qwen3-vl-4b-instruct"   # must match the model name shown in LM Studio
IMAGE  = "test.jpg"
PROMPT = "What is in this image? Describe it in detail."

with open(IMAGE, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

print(f"Sending request to LM Studio ({MODEL}) ...")
response = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": PROMPT},
        ],
    }],
    max_tokens=512,
)

print("\n" + response.choices[0].message.content)
