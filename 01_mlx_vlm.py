# MLX-VLM — Fastest on Apple Silicon (uses Metal + unified memory natively)
#
# INSTALL:
#   pip install mlx-vlm
#
# MODEL: Downloaded automatically from HuggingFace on first run (~2.5GB, 4-bit quantized)
#   mlx-community/Qwen3-VL-4B-Instruct-4bit
#
# RUN:
#   python 01_mlx_vlm.py

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL  = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
IMAGE  = "test.jpg"
PROMPT = "What is in this image? Describe it in detail."

print(f"Loading {MODEL} ...")
model, processor = load(MODEL)
config = load_config(MODEL)

formatted = apply_chat_template(processor, config, PROMPT, num_images=1)

print("Generating response ...\n")
response = generate(
    model,
    processor,
    formatted,
    image=IMAGE,
    max_tokens=512,
    verbose=False,
)

print(response)
