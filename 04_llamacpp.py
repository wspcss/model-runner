# llama.cpp — GGUF format, Metal GPU offload via Homebrew CLI
#
# INSTALL:
#   brew install llama.cpp
#
# DOWNLOAD MODEL FILES (run once):
#   pip install huggingface_hub
#   hf download Qwen/Qwen3-VL-4B-Instruct-GGUF \
#       Qwen3VL-4B-Instruct-Q4_K_M.gguf \
#       mmproj-Qwen3VL-4B-Instruct-F16.gguf \
#       --local-dir .
#
# RUN:
#   python 04_llamacpp.py

import subprocess
import shutil
import sys
import os

MODEL_GGUF  = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
MMPROJ_GGUF = "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
IMAGE       = "test.jpg"
PROMPT      = "What is in this image? Describe it in detail."

llama_cli = shutil.which("llama-cli")
if llama_cli is None:
    sys.exit("llama-cli not found — run: brew install llama.cpp")

for f in [MODEL_GGUF, MMPROJ_GGUF, IMAGE]:
    if not os.path.exists(f):
        sys.exit(f"Missing file: {f}")

cmd = [
    llama_cli,
    "--model",      MODEL_GGUF,
    "--mmproj",     MMPROJ_GGUF,
    "--image",      IMAGE,
    "--prompt",     PROMPT,
    "--n-predict",  "512",
    "--n-gpu-layers", "-1",     # offload all layers to Metal
    "--no-display-prompt",
    "--log-disable",
]

print("Running llama-cli ...\n")
result = subprocess.run(cmd, text=True, capture_output=True)
print(result.stdout.strip())
if result.returncode != 0:
    print(result.stderr[-1000:])  # show tail of stderr on failure
