# HuggingFace Transformers — Most flexible, runs straight from the Hub
#
# INSTALL:
#   pip install transformers accelerate torch pillow qwen-vl-utils
#
# MODEL: Downloaded automatically on first run (~8GB, bfloat16)
#   Qwen/Qwen3-VL-4B-Instruct
#
# RUN:
#   python 03_transformers.py

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL  = "Qwen/Qwen3-VL-4B-Instruct"
IMAGE  = "test.jpg"
PROMPT = "What is in this image? Describe it in detail."

print(f"Loading {MODEL} ...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",          # uses MPS (Metal) automatically on Apple Silicon
)
processor = AutoProcessor.from_pretrained(MODEL)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": f"file://{IMAGE}"},
        {"type": "text",  "text": PROMPT},
    ],
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
).to(model.device)

print("Generating response ...\n")
with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=512)

# Decode only the newly generated tokens
generated = output_ids[:, inputs["input_ids"].shape[1]:]
response  = processor.batch_decode(generated, skip_special_tokens=True)[0]

print(response)
