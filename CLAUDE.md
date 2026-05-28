# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmarking suite for **Qwen3-VL-4B-Instruct**, a vision-language model, running on macOS / Apple Silicon. It has two layers:

1. **Five standalone demo scripts** (`01_`–`05_`) — each loads `test.jpg` with the same prompt, differing only in backend.
2. **`backend_benchmark.py`** — the main harness that runs all backends, scores responses with an LLM judge, and produces JSON + HTML reports.

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

## Benchmark harness

`backend_benchmark.py` is the primary tool. It reads backends from `local_backends.json` and test cases from `test_case.json`, runs inference on each, scores responses via an LLM judge (OpenRouter), and saves results to `output/`.

```bash
# Run all backends
python backend_benchmark.py --all --api-key $OPENROUTER_API_KEY

# Run a single backend
python backend_benchmark.py --backend MLX-VLM-4bit

# Control output length
python backend_benchmark.py --all --max-tokens 256

# Override judge model
python backend_benchmark.py --all --judge-model openai/gpt-4o
```

Outputs land in `output/` as `backend_results_<label>_<timestamp>.{json,html}`.

### Adding a backend

Add an entry to `local_backends.json`. The `type` field must be one of: `mlx_vlm`, `ollama`, `transformers`, `llamacpp`, `lmstudio`.

### Adding test cases

Edit `test_case.json`. Each entry needs `question`, `image` (path to image file), and optionally `grounded_answer` (used for judge scoring) and `category`.

## Running the standalone scripts

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

## GGUF files

Three quantizations are already present in the repo root:

| File | Size | Used by |
|---|---|---|
| `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | ~2.5 GB | `llamacpp-Q4_K_M` backend |
| `Qwen3VL-4B-Instruct-Q8_0.gguf` | ~4.5 GB | available for custom configs |
| `Qwen3VL-4B-Instruct-F16.gguf` | ~8 GB | available for custom configs |
| `mmproj-Qwen3VL-4B-Instruct-F16.gguf` | — | multimodal projector (llama.cpp) |
| `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` | — | multimodal projector (quantized) |

## Dependencies

`requirements.txt` groups deps by script. Install all at once or selectively:

```bash
pip install -r requirements.txt          # everything
pip install mlx-vlm                      # 01 only
pip install transformers accelerate torch torchvision pillow qwen-vl-utils  # 03 only
```

`02_ollama.py` uses only stdlib; Ollama itself is installed via `brew install ollama`.  
`04_llamacpp.py` drives the `llama-cli` binary (`brew install llama.cpp`).  
`05_lmstudio.py` calls `http://localhost:1234/v1` using the `openai` package; requires LM Studio running with the model loaded and Local Server started.

## Changing the image or prompt

Each standalone script has three constants at the top — `MODEL`, `IMAGE`, and `PROMPT` — that are the only things to change for different inputs. For the benchmark harness, edit `test_case.json`.
