#!/usr/bin/env python3
"""
vLLM perf runner (text-only, or text + image with ``--image``).

Builds a prompt of exactly ``--prompt-len`` tokens from a text file (repeating
the text if it is too short, truncating otherwise), replicates it
``--batch-size`` times, generates ``--gen-len`` tokens per request and reports
wall time plus TTFT / TPOT from vLLM's metrics.

With ``--image`` every request contains the image plus text wrapped in the
model's chat template. The text filler is sized so the *total* prompt
(template + image tokens + text) is ``--prompt-len`` tokens as counted by the
model's HF processor; ``--prompt`` supplies a literal text instead (then
``--prompt-len`` is only an upper bound).

``--profile`` enables vLLM's engine/worker PyTorch profiler
(``profiler_config`` + start/stop_profile). Traces are written to ``--trace-dir``.

Example:
    python run_llm_vllm_perf.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --prompt-len 128 --gen-len 32 --batch-size 4 --profile
    python run_llm_vllm_perf.py --model Qwen/Qwen3-VL-32B-Instruct --tp-size 4 \
        --image cat.jpg --prompt "Describe the content of the image"
"""

import argparse
import gc
import os
import time
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Histogram

RNG_SEED = 42
DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "sonnet_extended.txt"
DEFAULT_TRACE_DIR = os.environ.get("VLM_TORCH_TRACE_DIR", "torch_traces")


def build_prompt_token_ids(tokenizer, text: str, prompt_len: int) -> list[int]:
    """Tokenize ``text`` and repeat/truncate it to exactly ``prompt_len`` ids."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("Prompt file tokenized to zero tokens")
    bos = tokenizer.bos_token_id
    prefix = [bos] if bos is not None else []
    body_len = prompt_len - len(prefix)
    reps = -(-body_len // len(ids))  # ceil division
    return prefix + (ids * reps)[:body_len]


def print_latency_metrics(llm: LLM) -> None:
    names = {
        "vllm:time_to_first_token_seconds": "TTFT",
        "vllm:inter_token_latency_seconds": "TPOT",
    }
    for metric in llm.get_metrics():
        if isinstance(metric, Histogram) and metric.name in names:
            mean = metric.sum / metric.count if metric.count > 0 else 0.0
            print(
                f"Mean {names[metric.name]} = {mean:.4f}s "
                f"(count={metric.count}, sum={metric.sum:.4f}s)"
            )


def _render(processor, text: str) -> str:
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": text}],
    }]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _count(processor, prompt: str, image) -> int:
    out = processor(text=[prompt], images=[image], return_tensors="pt")
    return int(out["input_ids"].shape[-1])


def build_vlm_prompt(processor, tokenizer, image, text: str, prompt_len: int) -> tuple[str, int]:
    """Return (prompt string, actual token count) targeting ``prompt_len`` tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("Prompt file tokenized to zero tokens")

    def make(n: int) -> str:
        reps = -(-max(n, 1) // len(ids))
        return tokenizer.decode((ids * reps)[:n]) if n > 0 else ""

    overhead = _count(processor, _render(processor, ""), image)
    if overhead > prompt_len:
        raise SystemExit(
            f"Image + chat template already use {overhead} tokens > --prompt-len {prompt_len}; "
            "increase --prompt-len or use a smaller image."
        )
    n = prompt_len - overhead
    prompt = _render(processor, make(n))
    actual = _count(processor, prompt, image)
    # Decode/re-encode can drift by a few tokens; correct for it.
    for _ in range(5):
        if actual == prompt_len:
            break
        n = max(0, n + (prompt_len - actual))
        prompt = _render(processor, make(n))
        actual = _count(processor, prompt, image)
    if actual != prompt_len:
        print(f"WARNING: could not hit exactly {prompt_len} prompt tokens; got {actual}")
    return prompt, actual


def test_vllm_perf(
    model_name: str,
    prompt_len: int,
    gen_len: int,
    batch_size: int,
    tp_size: int = 1,
    prompt_file: Path = DEFAULT_PROMPT_FILE,
    enforce_eager: bool = True,
    enable_prefix_caching: bool = False,
    profile: bool = False,
    trace_dir: str = DEFAULT_TRACE_DIR,
    language_model_only: bool = False,
    max_num_batched_tokens: int | None = None,
    kv_cache_memory_bytes: int | None = None,
    image_path: Path | None = None,
    prompt: str | None = None,
):
    image = Image.open(image_path).convert("RGB") if image_path else None
    trace_dir = trace_dir if profile else None
    profiler_config = None
    if trace_dir:
        trace_dir = str(Path(trace_dir).resolve())
        os.makedirs(trace_dir, exist_ok=True)
        profiler_config = {"profiler": "torch", "torch_profiler_dir": trace_dir}

    gc.collect()
    llm = LLM(
        model=model_name,
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=prompt_len + gen_len,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=0.95,
        async_scheduling=False,
        trust_remote_code=True,
        disable_log_stats=False,
        profiler_config=profiler_config,
        language_model_only=language_model_only,
        max_num_batched_tokens=max_num_batched_tokens,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        limit_mm_per_prompt={"image": 1} if image else None,
    )

    if image:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        if prompt is not None:
            full_prompt = _render(processor, prompt)
            actual_len = _count(processor, full_prompt, image)
            if actual_len > prompt_len:
                raise SystemExit(f"Prompt is {actual_len} tokens > --prompt-len {prompt_len}")
        else:
            text = Path(prompt_file).read_text()
            full_prompt, actual_len = build_vlm_prompt(processor, llm.get_tokenizer(), image, text, prompt_len)
        prompts = [
            {"prompt": full_prompt, "multi_modal_data": {"image": image}} for _ in range(batch_size)
        ]
    else:
        if prompt is not None:
            prompt_ids = llm.get_tokenizer().encode(prompt)
            if len(prompt_ids) > prompt_len:
                raise SystemExit(f"Prompt is {len(prompt_ids)} tokens > --prompt-len {prompt_len}")
        else:
            text = Path(prompt_file).read_text()
            prompt_ids = build_prompt_token_ids(llm.get_tokenizer(), text, prompt_len)
        actual_len = len(prompt_ids)
        prompts = [TokensPrompt(prompt_token_ids=list(prompt_ids)) for _ in range(batch_size)]

    sampling_params = SamplingParams(
        seed=RNG_SEED,
        temperature=0.0,
        max_tokens=gen_len,
        ignore_eos=True,
    )

    if profiler_config:
        llm.start_profile()
    try:
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()
    finally:
        if profiler_config:
            llm.stop_profile()
            print(f"vLLM engine/worker profiler traces written to {trace_dir}")

    total_time = end_time - start_time
    num_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    print("=" * 60)
    print(f"Model: {model_name}  TP={tp_size}  eager={enforce_eager}")
    if image:
        print(f"Image: {image_path} {image.size[0]}x{image.size[1]}")
    print(f"Batch size: {batch_size}  prompt len: {actual_len} (requested {prompt_len})  gen len: {gen_len}")
    print(f"Total generate time: {total_time:.4f}s")
    print(f"Output tokens: {num_out}  throughput: {num_out / total_time:.2f} tok/s")
    print_latency_metrics(llm)
    print(f"Sample output[0]: {outputs[0].outputs[0].text!r}")
    print("=" * 60)
    return outputs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct", help="Model name or path")
    p.add_argument("--prompt-len", type=int, default=128, help="Prompt length in tokens")
    p.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate")
    p.add_argument("--batch-size", type=int, default=1, help="Number of identical prompts")
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE, help="Text file to build the prompt from")
    p.add_argument("--compile", action="store_true", help="Use torch.compile (enforce_eager=False)")
    p.add_argument("--enable-prefix-caching", action="store_true")
    p.add_argument("--profile", action="store_true", help="Enable the PyTorch profiler")
    p.add_argument("--language-model-only", action="store_true", help="Skip the vision tower of multimodal models")
    p.add_argument("--max-num-batched-tokens", type=int, default=None, help="Chunked-prefill token budget per step")
    p.add_argument("--kv-cache-memory-bytes", type=int, default=None, help="Fixed KV cache size per device; skips memory profiling")
    p.add_argument("--image", type=Path, default=None, help="Input image file (multimodal models)")
    p.add_argument("--prompt", default=None, help="Literal text prompt; skips filler padding, --prompt-len becomes an upper bound")
    p.add_argument("--trace-dir", default=DEFAULT_TRACE_DIR, help="Output dir for profiler traces")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.prompt_len < 1 or args.gen_len < 1 or args.batch_size < 1:
        raise SystemExit("--prompt-len, --gen-len and --batch-size must be >= 1")
    if args.image is not None and not args.image.is_file():
        raise SystemExit(f"--image not found: {args.image}")
    test_vllm_perf(
        model_name=args.model,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        batch_size=args.batch_size,
        tp_size=args.tp_size,
        prompt_file=args.prompt_file,
        enforce_eager=not args.compile,
        enable_prefix_caching=args.enable_prefix_caching,
        profile=args.profile,
        trace_dir=args.trace_dir,
        language_model_only=args.language_model_only,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        image_path=args.image,
        prompt=args.prompt,
    )
