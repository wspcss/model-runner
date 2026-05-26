# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Five standalone Python scripts that each demonstrate a different inference backend for the **Qwen3-VL-4B-Instruct** vision-language model on macOS / Apple Silicon. All scripts load `test.jpg` and answer the same prompt; they differ only in the backend used.

| Script | Backend | Model format |
|---|---|---|
| `01_mlx_vlm.py` | MLX-VLM (fastest on Apple Silicon) | 4-bit from HuggingFace Hub |
| `02_ollama.py` | Ollama REST API | Pulled via `ollama pull` |
| `03_transformers.py` | HuggingFace Transformers + MPS | bfloat16 from HuggingFace Hub |
| `04_llamacpp.py` | llama-cli (Homebrew llama.cpp) | GGUF files in repo root |
| `05_lmstudio.py` | LM Studio OpenAI-compatible endpoint | GGUF loaded in LM Studio GUI |

## Environment

A local `.venv` is present. Activate it before running any script:

```bash
source .venv/bin/activate
```

## Running the scripts

Each script has its own one-time setup documented in a comment block at the top of the file. The short form:

```bash
# MLX-VLM (auto-downloads model on first run, ~2.5 GB)
python 01_mlx_vlm.py

# Ollama (requires ollama serve + ollama pull qwen3-vl:4b first)
python 02_ollama.py

# Transformers (auto-downloads model on first run, ~8 GB)
python 03_transformers.py

# llama.cpp (requires brew install llama.cpp; GGUF files already in repo)
python 04_llamacpp.py

# LM Studio (load model in GUI, start Local Server, then run)
python 05_lmstudio.py
```

## Dependencies

`requirements.txt` groups deps by script. Install all at once or selectively:

```bash
pip install -r requirements.txt          # everything
pip install mlx-vlm                      # 01 only
pip install transformers accelerate torch torchvision pillow qwen-vl-utils  # 03 only
```

`02_ollama.py` uses only stdlib; Ollama itself is installed via `brew install ollama`.  
`04_llamacpp.py` drives the `llama-cli` binary (`brew install llama.cpp`); GGUF model files (`*.gguf`) are already present in the repo root.  
`05_lmstudio.py` calls `http://localhost:1234/v1` using the `openai` package (`pip install openai`); requires LM Studio running with the model loaded and Local Server started.

## Changing the image or prompt

Each script has three constants at the top — `MODEL`, `IMAGE`, and `PROMPT` — that are the only things to change for different inputs.
