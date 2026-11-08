#!/usr/bin/env python3
"""
========================================================================================
vLLM Single-File Guide & Standalone Execution Script for nanoGentzen ClearCoT Studio
========================================================================================

This script demonstrates two standard ways to use vLLM in Python:
  Mode 1 (Recommended for ClearCoT Studio):
    - Launching & managing the vLLM OpenAI-compatible server from Python
    - Connecting with LocalLLMClient to run high-throughput concurrent batch requests
  Mode 2 (Offline Batch Inference):
    - Using the in-process `vllm.LLM` class directly for ultra-fast offline generation

Requirements:
  pip install vllm httpx
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import project client
from llm_backend.client import LocalLLMClient
from llm_backend.vllm_server import (
    is_vllm_installed,
    is_vllm_running,
    start_vllm_server,
    stop_vllm_server,
    wait_for_vllm_ready,
)


def run_server_mode(
    model: str = "Sagicc/Occam-2B-CCoT",
    port: int = 8000,
    gpu_memory_utilization: float = 0.70,
    quantization: str = "none",
    concurrency: int = 8
):
    """
    MODE 1: Python-managed vLLM Server + LocalLLMClient
    ---------------------------------------------------
    1. Checks if vLLM server is already running on http://localhost:8000/v1
    2. If not running, launches it as a background subprocess with desired parameters
    3. Waits for /health check to turn green
    4. Dispatches concurrent requests using LocalLLMClient
    """
    base_url = f"http://localhost:{port}/v1"
    print(f"\n=======================================================")
    print(f" Mode 1: vLLM Server Mode (OpenAI-Compatible API)")
    print(f" Target Endpoint: {base_url}")
    print(f" Model:           {model}")
    print(f" Quantization:    {quantization}")
    print(f" Concurrency:     {concurrency} workers")
    print(f"=======================================================\n")

    if not is_vllm_installed():
        print("❌ 'vllm' package is not installed in the current environment.")
        print("   Install it using:  pip install vllm\n")
        return

    # Check if server is already running
    if is_vllm_running(base_url):
        print(f"✅ vLLM server is already active at {base_url}.")
    else:
        print(f"🚀 Starting vLLM server process on port {port}...")
        pid = start_vllm_server(
            model=model,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048,
            quantization=quantization,
        )
        print(f"   Spawned PID: {pid}. Streaming logs to data/vllm_server.log (DTSTUDIO_DATA_DIR)...")
        print("   Waiting for model weights to load and server to report ready...")

        ready = wait_for_vllm_ready(base_url=base_url, timeout_seconds=120)
        if not ready:
            print("❌ Server failed to start within timeout. Check logs in data/vllm_server.log (DTSTUDIO_DATA_DIR).")
            return
        print(f"✅ Server is ONLINE and healthy at {base_url}!")

    # Test client connection & query models
    client = LocalLLMClient(base_url=base_url)
    available_models = client.list_models()
    print(f"\n📋 Live models registered in vLLM: {available_models}")

    active_model = available_models[0] if available_models else model

    # Test concurrent batch generation
    sample_prompts = [
        "Explain the relation between entropy and information theory in 2 sentences.",
        "State the Church-Turing thesis and its significance for computer science.",
        "What is Gentzen's sequent calculus LI and why is it cut-free?",
        "Define Kolmogorov complexity and how it relates to algorithmic probability.",
        "Describe how gradient descent optimizes non-convex neural network loss surfaces.",
        "Explain the difference between intuitionistic logic and classical logic.",
        "How does PagedAttention eliminate memory fragmentation in LLM inference?",
        "What is the mathematical definition of a Markov Decision Process (MDP)?"
    ]

    print(f"\n⚡ Dispatching {len(sample_prompts)} prompts concurrently (concurrency={concurrency})...")
    t0 = time.perf_counter()

    responses = client.generate_concurrent_batch(
        requests=[{"prompt": p, "model": active_model, "max_tokens": 128} for p in sample_prompts],
        concurrency=concurrency
    )

    elapsed = time.perf_counter() - t0
    print(f"🎉 Finished all {len(sample_prompts)} requests in {elapsed:.2f}s ({len(sample_prompts)/elapsed:.1f} req/s)!\n")

    for idx, (p, r) in enumerate(zip(sample_prompts[:3], responses[:3]), 1):
        print(f"--- [Sample {idx}] ---")
        print(f"Q: {p}")
        print(f"A: {r.strip()[:180]}...\n")


def run_offline_embedded_mode(
    model: str = "Sagicc/Occam-2B-CCoT",
    gpu_memory_utilization: float = 0.75,
    quantization: str = "none"
):
    """
    MODE 2: Offline Embedded vLLM (In-Process)
    -------------------------------------------
    Directly loads the LLM engine into Python memory using `vllm.LLM`.
    Best for standalone batch scripts outside of web servers.
    """
    print(f"\n=======================================================")
    print(f" Mode 2: In-Process Offline vLLM (vllm.LLM)")
    print(f" Model:        {model}")
    print(f" Quantization: {quantization}")
    print(f"=======================================================\n")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("❌ 'vllm' package is not installed. Run: pip install vllm")
        return

    print("⏳ Initializing in-process vLLM engine...")
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        quantization=quantization if quantization.lower() != "none" else None,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=256
    )

    prompts = [
        "What is Gentzen's Natural Deduction?",
        "Define continuous batching in high-throughput LLM serving."
    ]

    print(f"⚡ Generating outputs for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text.strip()[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Single-File Runner for ClearCoT Studio")
    parser.add_argument("--mode", choices=["server", "offline", "stop"], default="server",
                        help="'server': Launch/connect to vLLM OpenAI server; 'offline': in-process LLM; 'stop': stop server")
    parser.add_argument("--model", type=str, default="Sagicc/Occam-2B-CCoT",
                        help="HuggingFace model ID or local path (default: Sagicc/Occam-2B-CCoT)")
    parser.add_argument("--port", type=int, default=8000, help="Port for vLLM server (default: 8000)")
    parser.add_argument("--gpu-mem", type=float, default=0.55, help="GPU memory utilization (default: 0.55 for 8GB GPUs)")
    parser.add_argument("--quantization", type=str, default="none", choices=["none", "awq", "gptq", "bitsandbytes"],
                        help="Quantization type (default: none for 2B models on 8GB GPU)")
    parser.add_argument("--concurrency", type=int, default=8, help="Batch concurrency (default: 8)")

    args = parser.parse_args()

    if args.mode == "stop":
        stopped = stop_vllm_server(port=args.port)
        if stopped:
            print(f"⏹️ Successfully stopped vLLM server on port {args.port}.")
        else:
            print("ℹ️ No active vLLM process found to terminate.")
    elif args.mode == "offline":
        run_offline_embedded_mode(
            model=args.model,
            gpu_memory_utilization=args.gpu_mem,
            quantization=args.quantization
        )
    else:
        run_server_mode(
            model=args.model,
            port=args.port,
            gpu_memory_utilization=args.gpu_mem,
            quantization=args.quantization,
            concurrency=args.concurrency
        )
