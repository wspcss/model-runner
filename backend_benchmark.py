#!/usr/bin/env python3
"""
Backend Benchmark Script
Benchmarks Qwen3-VL-4B-Instruct across different inference backends on macOS / Apple Silicon.
Uses OpenRouter API for LLM judge scoring.
Supports benchmarking a single backend or all backends from local_backends.json.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import base64
import subprocess
import shutil

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL = "openai/gpt-5.4"

LOCAL_BACKENDS_FILE = "local_backends.json"
TEST_CASES_FILE = "test_case.json"
MAX_TOKENS = 512


def sanitize_name(name):
    return name.replace("/", "_").strip()


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/png")
    return f"data:{mime_type};base64,{encoded}"


def call_openrouter(model, messages, max_tokens):
    import urllib.request
    import urllib.error

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            response_data = json.loads(body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"API error {e.code}: {error_body}")
    elapsed = time.time() - start
    return response_data, elapsed


def judge_response(question, grounded_answer, model_response):
    judge_prompt = (
        f"Given this question:\n\"{question}\"\n\n"
        f"The expected answer is:\n\"{grounded_answer}\"\n\n"
        f"The model responded:\n\"{model_response}\"\n\n"
        f"Rate how accurately the model's response matches the expected answer "
        f"on a scale of 0 to 10, where 0 is completely wrong and 10 is when all points in the expected answer are covered. "
        f"Reply with only a single number."
    )
    messages = [{"role": "user", "content": judge_prompt}]
    try:
        response_data, _ = call_openrouter(JUDGE_MODEL, messages, max_tokens=512)
        choices = response_data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            text = (msg.get("content") or "").strip()
            if not text:
                reasoning = (msg.get("reasoning") or "").strip()
                if reasoning:
                    text = reasoning
            num_match = re.search(r'\b(10|\d)\b', text)
            if num_match:
                return int(num_match.group())
        return None
    except Exception as e:
        print(f"    ⚠️  Judge API call failed: {e}")
        return None


# ── Backend inference functions ───────────────────────────────────────────────

def _run_mlx_inference(cfg, model, processor, image_path, question, max_tokens):
    from mlx_vlm.generate import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(processor, model.config, question, num_images=1)
    start = time.time()
    result = mlx_generate(
        model,
        processor,
        prompt=prompt,
        image=image_path,
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    elapsed = time.time() - start
    return {
        "response_text": result.text.split("</think>")[-1].strip(),
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "generation_tps": result.generation_tps,
        "peak_memory_gb": result.peak_memory,
        "time_seconds": elapsed,
    }


def _run_ollama_inference(cfg, image_path, question, max_tokens):
    import urllib.request

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    endpoint = cfg.get("endpoint", "http://localhost:11434")
    payload = json.dumps({
        "model": cfg["model"],
        "messages": [{"role": "user", "content": question, "images": [image_b64]}],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }).encode()

    req = urllib.request.Request(
        f"{endpoint}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - start

    eval_count = result.get("eval_count", 0)
    eval_duration_ns = result.get("eval_duration", 0)
    tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else None
    msg = result["message"]
    content = msg.get("content", "").strip()
    if not content:
        content = msg.get("thinking", "").strip()
    return {
        "response_text": content,
        "prompt_tokens": result.get("prompt_eval_count"),
        "generation_tokens": eval_count or None,
        "generation_tps": tps,
        "peak_memory_gb": None,
        "time_seconds": elapsed,
    }


def _run_transformers_inference(cfg, model, processor, image_path, question, max_tokens):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )

    # When mlx_vlm is imported in the same process, the processor may return
    # mlx arrays instead of torch tensors despite return_tensors="pt". Convert
    # everything explicitly so model.generate() receives proper tensors.
    import numpy as np
    safe_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            safe_inputs[k] = v.to(device)
        else:
            safe_inputs[k] = torch.from_numpy(np.asarray(v)).to(device)
    inputs = safe_inputs

    prompt_len = inputs["input_ids"].shape[1]
    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    elapsed = time.time() - start

    generated = output_ids[:, prompt_len:]
    generation_tokens = generated.shape[1]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]
    tps = generation_tokens / elapsed if elapsed > 0 else None
    return {
        "response_text": response.split("</think>")[-1].strip(),
        "prompt_tokens": prompt_len,
        "generation_tokens": generation_tokens,
        "generation_tps": tps,
        "peak_memory_gb": None,
        "time_seconds": elapsed,
    }


def _run_llamacpp_inference(cfg, image_path, question, max_tokens):
    llama_cli = shutil.which("llama-cli")
    if llama_cli is None:
        raise RuntimeError("llama-cli not found — run: brew install llama.cpp")

    model_gguf = cfg["model_gguf"]
    mmproj_gguf = cfg["mmproj_gguf"]
    for path in [model_gguf, mmproj_gguf]:
        if not os.path.exists(path):
            raise RuntimeError(f"Missing file: {path}")

    cmd = [
        llama_cli,
        "--model", model_gguf,
        "--mmproj", mmproj_gguf,
        "--image", image_path,
        "--prompt", question,
        "--n-predict", str(max_tokens),
        "--n-gpu-layers", "-1",
        "--no-display-prompt",
        "--log-disable",
        "--simple-io",
        "--single-turn",
    ]
    start = time.time()
    result = subprocess.run(cmd, text=True, capture_output=True, stdin=subprocess.DEVNULL)
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"llama-cli failed:\n{result.stderr[-500:]}")

    # Extract the actual model response from stdout.
    # llama-cli prints a banner and echoes the prompt as "> <question>" before
    # the response. The generation stats follow as "[ Prompt: X t/s | Generation: Y t/s ]".
    tps = None
    gen_tokens = None
    stdout = result.stdout
    # Remove spinner animation (char + backspace pairs)
    stdout = re.sub(r'.\x08', '', stdout)
    response_lines = []
    in_response = False
    for line in stdout.splitlines():
        if line.startswith('> '):
            in_response = True
            continue
        if in_response:
            m = re.search(r'Generation:\s*([\d.]+)\s*t/s', line)
            if m:
                tps = float(m.group(1))
                break
            if line.strip() in ('Exiting...', ''):
                continue
            # Strip leading spinner remnants (e.g. "|- ")
            clean = re.sub(r'^[|/\\-]+\s*', '', line)
            response_lines.append(clean)
    response_text = '\n'.join(response_lines).strip()

    # Also check old-style llama_print_timings from stderr as fallback
    if tps is None:
        for line in result.stderr.splitlines():
            m = re.search(
                r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second",
                line,
            )
            if m:
                gen_tokens = int(m.group(1))
                tps = float(m.group(2))
                break

    return {
        "response_text": response_text,
        "prompt_tokens": None,
        "generation_tokens": gen_tokens,
        "generation_tps": tps,
        "peak_memory_gb": None,
        "time_seconds": elapsed,
    }


def _run_lmstudio_inference(cfg, image_path, question, max_tokens):
    from openai import OpenAI

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    endpoint = cfg.get("endpoint", "http://localhost:1234")
    client = OpenAI(base_url=f"{endpoint}/v1", api_key="lm-studio")

    start = time.time()
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                {"type": "text", "text": question},
            ],
        }],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.time() - start

    usage = response.usage
    gen_tokens = usage.completion_tokens if usage else None
    prompt_tokens = usage.prompt_tokens if usage else None
    tps = gen_tokens / elapsed if gen_tokens and elapsed > 0 else None
    return {
        "response_text": response.choices[0].message.content.split("</think>")[-1].strip(),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": gen_tokens,
        "generation_tps": tps,
        "peak_memory_gb": None,
        "time_seconds": elapsed,
    }


# ── Backend lifecycle ─────────────────────────────────────────────────────────

def load_backend(cfg):
    t = cfg["type"]
    if t == "mlx_vlm":
        from mlx_vlm import load as mlx_load
        print(f"  Loading MLX-VLM model: {cfg['model']}")
        model, processor = mlx_load(cfg["model"])
        return {"model": model, "processor": processor}
    elif t == "transformers":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"  Loading Transformers model: {cfg['model']}")
        model = AutoModelForImageTextToText.from_pretrained(
            cfg["model"], torch_dtype=torch.bfloat16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(cfg["model"])
        return {"model": model, "processor": processor}
    else:
        return {}


def unload_backend(cfg, state):
    if cfg["type"] in ("mlx_vlm", "transformers") and state:
        del state["model"], state["processor"]
        gc.collect()


def _dispatch_inference(backend_type, state, cfg, image_path, question, max_tokens):
    if backend_type == "mlx_vlm":
        return _run_mlx_inference(cfg, state["model"], state["processor"], image_path, question, max_tokens)
    elif backend_type == "ollama":
        return _run_ollama_inference(cfg, image_path, question, max_tokens)
    elif backend_type == "transformers":
        return _run_transformers_inference(cfg, state["model"], state["processor"], image_path, question, max_tokens)
    elif backend_type == "llamacpp":
        return _run_llamacpp_inference(cfg, image_path, question, max_tokens)
    elif backend_type == "lmstudio":
        return _run_lmstudio_inference(cfg, image_path, question, max_tokens)
    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}")


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(backend_config, tests, max_tokens):
    backend_name = backend_config["name"]
    backend_type = backend_config["type"]

    print(f"\n{'=' * 70}")
    print(f"BACKEND BENCHMARK: {backend_name} ({backend_type})")
    print(f"{'=' * 70}")

    state = load_backend(backend_config)
    results = []
    total_start = time.time()
    total_input_tokens = 0
    total_output_tokens = 0

    try:
        for i, case in enumerate(tests, 1):
            question = case["question"]
            grounded = case.get("grounded_answer", "")
            image_path = case.get("image", "")
            if not image_path:
                print(f"\n  ❌ No image specified for test case {i}")
                continue
            if not os.path.exists(image_path):
                print(f"\n  ❌ Image not found: {image_path}")
                continue

            category = case.get("category", "")
            print(f"\n  Image: {image_path}")
            if category:
                print(f"  Category: {category}")

            try:
                inference = _dispatch_inference(backend_type, state, backend_config, image_path, question, max_tokens)
            except Exception as e:
                print(f"\n  ❌ Inference failed: {e}")
                results.append({
                    "question": question,
                    "model_response": f"ERROR: {e}",
                    "grounded_answer": grounded,
                    "image": image_path,
                    "category": category,
                    "accuracy_score": None,
                    "time_seconds": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tokens_per_second": None,
                    "peak_memory_gb": None,
                })
                continue

            response_text = inference["response_text"]
            input_tokens = inference["prompt_tokens"] or 0
            output_tokens = inference["generation_tokens"] or 0
            tps = inference["generation_tps"]
            elapsed = inference["time_seconds"]
            peak_memory = inference["peak_memory_gb"]

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            if grounded.strip():
                print(f"  🤖 Asking judge ({JUDGE_MODEL})...")
                accuracy_score = judge_response(question, grounded, response_text)
                if accuracy_score is not None:
                    if accuracy_score >= 7:
                        status = f"✅ {accuracy_score}/10"
                    elif accuracy_score >= 4:
                        status = f"⚠️  {accuracy_score}/10"
                    else:
                        status = f"❌ {accuracy_score}/10"
                else:
                    status = "📝 JUDGE FAILED"
            else:
                accuracy_score = None
                status = "📝 INFO (no grounded answer)"

            print(f"\n[{i}/{len(tests)}] {status}")
            print(f"  Q:       {question[:100]}...")
            print(f"  Answer:  {response_text[:400]}")
            if grounded.strip():
                print(f"  Truth:   {grounded}")
            if accuracy_score is not None:
                print(f"  Judge:   {accuracy_score}/10")
            tps_str = f"{tps:.1f}" if tps is not None else "N/A"
            mem_str = f"{peak_memory:.2f} GB" if peak_memory is not None else "N/A"
            in_str = str(input_tokens) if input_tokens else "N/A"
            out_str = str(output_tokens) if output_tokens else "N/A"
            print(f"  Time: {elapsed:.2f}s | In: {in_str} | Out: {out_str} | TPS: {tps_str} | Peak Mem: {mem_str}")

            results.append({
                "question": question,
                "model_response": response_text,
                "grounded_answer": grounded,
                "image": image_path,
                "category": category,
                "accuracy_score": accuracy_score,
                "time_seconds": round(elapsed, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": round(tps, 1) if tps is not None else None,
                "peak_memory_gb": round(peak_memory, 2) if peak_memory is not None else None,
            })

    finally:
        unload_backend(backend_config, state)

    total_elapsed = time.time() - total_start

    valid_results = [r for r in results if r["time_seconds"] > 0]
    tps_values = [r["tokens_per_second"] for r in valid_results if r["tokens_per_second"] is not None]
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else None
    avg_time = sum(r["time_seconds"] for r in valid_results) / len(valid_results) if valid_results else 0
    mem_values = [r["peak_memory_gb"] for r in valid_results if r["peak_memory_gb"] is not None]
    avg_peak_memory = sum(mem_values) / len(mem_values) if mem_values else None

    scored_results = [r for r in results if r["accuracy_score"] is not None]
    total_scored = len(scored_results)
    avg_score = sum(r["accuracy_score"] for r in scored_results) / total_scored if total_scored > 0 else None

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Backend:          {backend_name}")
    print(f"Test cases:       {len(results)}")
    if avg_score is not None:
        print(f"Avg judge score:  {avg_score:.1f}/10 ({total_scored} scored)")
    print(f"Total time:       {total_elapsed:.2f}s")
    print(f"Avg per case:     {avg_time:.2f}s")
    if avg_tps is not None:
        print(f"Avg TPS:          {avg_tps:.1f}")
    if avg_peak_memory is not None:
        print(f"Avg peak memory:  {avg_peak_memory:.2f} GB")
    print(f"Total tokens in:  {total_input_tokens}")
    print(f"Total tokens out: {total_output_tokens}")

    return {
        "model": backend_name,
        "backend_type": backend_type,
        "total_cases": len(results),
        "scored_cases": total_scored,
        "avg_judge_score": round(avg_score, 1) if avg_score is not None else None,
        "total_time_s": round(total_elapsed, 2),
        "avg_per_case_s": round(avg_time, 2),
        "avg_tokens_per_second": round(avg_tps, 1) if avg_tps is not None else None,
        "avg_peak_memory_gb": round(avg_peak_memory, 2) if avg_peak_memory is not None else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "results": results,
    }


# ── HTML report ───────────────────────────────────────────────────────────────

def generate_html_report(summaries, output_path):
    import html

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    comparison_rows = ""
    for s in summaries:
        avg_judge = s.get("avg_judge_score")
        if avg_judge is not None:
            if avg_judge >= 7:
                score_badge = f'<span style="color:#16a34a;font-weight:bold;">{avg_judge:.1f}/10</span>'
            elif avg_judge >= 4:
                score_badge = f'<span style="color:#d97706;font-weight:bold;">{avg_judge:.1f}/10</span>'
            else:
                score_badge = f'<span style="color:#dc2626;font-weight:bold;">{avg_judge:.1f}/10</span>'
        else:
            score_badge = '<span style="color:#999;">N/A</span>'

        tps = s.get("avg_tokens_per_second")
        tps_cell = f"{tps:.1f}" if tps is not None else "N/A"
        mem = s.get("avg_peak_memory_gb")
        mem_cell = f"{mem:.2f} GB" if mem is not None else "N/A"

        comparison_rows += f"""
        <tr>
          <td>{html.escape(s['model'])}</td>
          <td>{html.escape(s.get('backend_type', ''))}</td>
          <td>{tps_cell}</td>
          <td>{s['avg_per_case_s']:.2f}s</td>
          <td>{score_badge}</td>
          <td>{mem_cell}</td>
          <td>{s['total_input_tokens']}</td>
          <td>{s['total_output_tokens']}</td>
        </tr>"""

    max_cases = max((len(s.get("results", [])) for s in summaries), default=0)
    test_case_sections = ""
    for case_idx in range(max_cases):
        case_info = None
        for s in summaries:
            results = s.get("results", [])
            if case_idx < len(results):
                case_info = results[case_idx]
                break

        if not case_info:
            continue

        question = case_info.get("question", "")
        grounded = case_info.get("grounded_answer", "")
        img_path = case_info.get("image", "")
        category = case_info.get("category", "")

        image_html = ""
        if img_path and os.path.exists(img_path):
            img_b64 = encode_image_to_base64(img_path)
            image_html = f'<img src="{img_b64}" alt="{html.escape(img_path)}" style="max-height:300px;border-radius:8px;border:1px solid #e2e8f0;" />'

        comparison_rows_html = ""
        for s in summaries:
            results = s.get("results", [])
            if case_idx >= len(results):
                continue
            r = results[case_idx]

            score = r.get("accuracy_score")
            if score is not None:
                if score >= 7:
                    score_cell = f'<td style="color:#16a34a;font-weight:bold;">{score}/10</td>'
                elif score >= 4:
                    score_cell = f'<td style="color:#d97706;font-weight:bold;">{score}/10</td>'
                else:
                    score_cell = f'<td style="color:#dc2626;font-weight:bold;">{score}/10</td>'
            else:
                score_cell = '<td style="color:#999;">N/A</td>'

            tps = r.get("tokens_per_second")
            tps_cell = f"{tps:.1f}" if tps is not None else "N/A"
            mem = r.get("peak_memory_gb")
            mem_cell = f"{mem:.2f} GB" if mem is not None else "N/A"

            response_escaped = html.escape(r.get("model_response", ""))
            comparison_rows_html += f"""
            <tr>
              <td>{html.escape(s['model'])}</td>
              <td style="max-width:600px;white-space:pre-wrap;">{response_escaped}</td>
              {score_cell}
              <td>{r.get('time_seconds', 0):.2f}s</td>
              <td>{tps_cell}</td>
              <td>{mem_cell}</td>
            </tr>"""

        category_badge = (
            f'<span style="background:#e0e7ff;color:#3730a3;padding:0.2rem 0.6rem;border-radius:9999px;font-size:0.8rem;font-weight:600;">{html.escape(category)}</span>'
            if category else ""
        )
        test_case_sections += f"""
    <div class="test-case-section">
      <h2>Test Case {case_idx + 1} {category_badge}</h2>
      <div class="case-info">
        {image_html}
        <div class="case-details">
          <p><strong>Question:</strong> {html.escape(question)}</p>
          <p><strong>Expected Answer:</strong> {html.escape(grounded) if grounded else '<em style="color:#999;">none</em>'}</p>
        </div>
      </div>
      <table>
        <thead>
          <tr><th>Backend</th><th>Response</th><th>Score</th><th>Time</th><th>TPS</th><th>Peak Mem</th></tr>
        </thead>
        <tbody>{comparison_rows_html}
        </tbody>
      </table>
    </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Backend Benchmark Results - {timestamp}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #0f172a; }}
    .timestamp {{ color: #64748b; margin-bottom: 2rem; font-size: 0.95rem; }}
    .comparison {{ background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .comparison h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #334155; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th {{ background: #f1f5f9; padding: 0.75rem 1rem; text-align: left; font-weight: 600; color: #475569; border-bottom: 2px solid #e2e8f0; }}
    td {{ padding: 0.65rem 1rem; border-bottom: 1px solid #f1f5f9; vertical-align: top; }}
    tr:hover {{ background: #f8fafc; }}
    .test-case-section {{ background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .test-case-section h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
    .case-info {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; align-items: flex-start; flex-wrap: wrap; }}
    .case-details {{ flex: 1; min-width: 300px; }}
    .case-details p {{ margin-bottom: 0.5rem; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>⚡ Backend Benchmark Results</h1>
    <p class="timestamp">Generated: {timestamp} | Backends: {len(summaries)} | Judge: {html.escape(JUDGE_MODEL)}</p>

    <div class="comparison">
      <h2>Backend Comparison</h2>
      <table>
        <thead>
          <tr><th>Backend</th><th>Type</th><th>Avg TPS</th><th>Avg/Case</th><th>Avg Score</th><th>Avg Peak Mem</th><th>Tokens In</th><th>Tokens Out</th></tr>
        </thead>
        <tbody>{comparison_rows}
        </tbody>
      </table>
    </div>

    {test_case_sections}
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global API_KEY, JUDGE_MODEL

    parser = argparse.ArgumentParser(description="Backend Benchmark (Qwen3-VL-4B-Instruct across inference backends)")
    parser.add_argument("--backend", type=str, default=None,
                        help="Single backend name to run (must match 'name' in local_backends.json)")
    parser.add_argument("--all", action="store_true",
                        help=f"Benchmark all backends in {LOCAL_BACKENDS_FILE}")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens to generate (default: {MAX_TOKENS})")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenRouter API key for judge scoring (overrides env var)")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL,
                        help=f"Model to use for judge scoring (default: {JUDGE_MODEL})")
    args = parser.parse_args()

    if args.api_key:
        API_KEY = args.api_key
    JUDGE_MODEL = args.judge_model

    with open(TEST_CASES_FILE) as f:
        tests = json.load(f)

    for i, case in enumerate(tests, 1):
        img = case.get("image", "")
        if not img:
            print(f"ERROR: Test case {i} has no image specified")
            sys.exit(1)
        if not os.path.exists(img):
            print(f"ERROR: Image not found: {img} (test case {i})")
            sys.exit(1)

    with open(LOCAL_BACKENDS_FILE) as f:
        all_backends = json.load(f)

    if args.all:
        backends = all_backends
    elif args.backend:
        matches = [b for b in all_backends if b["name"] == args.backend]
        if not matches:
            available = ", ".join(b["name"] for b in all_backends)
            print(f"ERROR: Backend '{args.backend}' not found. Available: {available}")
            sys.exit(1)
        backends = matches
    else:
        print("ERROR: Specify --backend <name> or --all")
        sys.exit(1)

    print("=" * 70)
    print("  BACKEND BENCHMARK SUITE")
    print(f"  Backends:   {len(backends)}")
    print(f"  Test cases: {len(tests)}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Judge:      {JUDGE_MODEL} (via OpenRouter)")
    print("=" * 70)

    summaries = []
    failed = []

    for backend_config in backends:
        print(f"\n{'─' * 70}")
        print(f"  Backend: {backend_config['name']} ({backend_config['type']})")
        print(f"{'─' * 70}")
        try:
            summary = run_benchmark(backend_config, tests, args.max_tokens)
            summaries.append(summary)
        except Exception as e:
            print(f"\n  ❌ FAILED: {backend_config['name']}: {e}")
            failed.append(backend_config["name"])

    if failed:
        print(f"\n⚠️  {len(failed)} backend(s) failed:")
        for name in failed:
            print(f"   - {name}")

    if len(summaries) > 1:
        print("\n\n" + "=" * 70)
        print("  BACKEND COMPARISON")
        print("=" * 70)
        print(f"\n{'Backend':<35} {'Type':<15} {'Avg TPS':>10} {'Avg/Case':>10} {'Avg Score':>10} {'Peak Mem':>10}")
        print("-" * 92)
        for s in summaries:
            avg_judge = s.get("avg_judge_score")
            score_str = f"{avg_judge}/10" if avg_judge is not None else "N/A"
            tps = s.get("avg_tokens_per_second")
            tps_str = f"{tps:.1f}" if tps is not None else "N/A"
            mem = s.get("avg_peak_memory_gb")
            mem_str = f"{mem:.2f} GB" if mem is not None else "N/A"
            print(f"{s['model']:<35} {s.get('backend_type', ''):<15} {tps_str:>10} {s['avg_per_case_s']:>9.2f}s {score_str:>10} {mem_str:>10}")

    label = os.path.splitext(LOCAL_BACKENDS_FILE)[0] if args.all else sanitize_name(backends[0]["name"])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    json_output = os.path.join(output_dir, f"backend_results_{label}_{timestamp}.json")
    with open(json_output, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to: {json_output}")

    html_output = os.path.join(output_dir, f"backend_results_{label}_{timestamp}.html")
    generate_html_report(summaries, html_output)
    print(f"HTML report saved to: {html_output}")

    print("\n" + "=" * 70)
    print("  Backend benchmarks complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
