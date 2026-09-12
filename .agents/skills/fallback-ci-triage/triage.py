#!/usr/bin/env python3
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# ruff: noqa: E501, UP031
# Triage a vllm-qaic Fallback CI sweep.
#
# Reads the log_1_<model>_tp<N>.log / parse_1_<model>_tp<N>.log pairs a sweep
# leaves in ci_scripts/ci_logs/<suite>/, works out each run's *root* failure,
# buckets it, and writes both a machine-readable CSV and a human FAILURE_REPORT.md.
#
# Usage (from anywhere):
#   triage.py ci_scripts/ci_logs/vlm --stdout     # print the summary only, write nothing
#   triage.py ci_scripts/ci_logs/llm              # writes both files into that dir
#   triage.py <dir> --force                       # required if those files already exist
#   triage.py <dir> --title "VLM"                 # override the report heading
#
# Writing refuses to clobber an existing FAILURE_REPORT.md / failure_categories.csv
# without --force: a delivered report normally carries hand-written sections this
# script cannot regenerate.
#
# The categorisation is deliberately conservative: anything it cannot place
# lands in F with the raw root error as the fix note, so unknown failures are
# visible rather than silently mislabelled. Grep the report for "F." after a
# new sweep and add rules for whatever shows up.
# ------------------------------------------------------------------
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import os
import regex as re
import sys

# A traceback's last line, e.g. "ValueError: ..." or a bare "Exception: ...".
# The (?:[A-Za-z_][\w.]*)? prefix MUST be optional -- a bare "Exception:" has no
# prefix and was silently dropped by an earlier version of this regex.
EXC = re.compile(
    r"^((?:[A-Za-z_][\w.]*)?(?:Error|Exception|NotFound|Interrupt))(?:\(.*?\))?: (.*)$"
)

# vLLM re-raises worker failures through several layers. These wrappers carry no
# diagnostic value, so the root cause is the first exception that is NOT one of
# them. Without this filter every multiproc failure reads as "Engine core
# initialization failed".
WRAPPERS = (
    "Engine core initialization failed",
    "Worker failed with error",
    "Engine core proc",
    "WorkerProc initialization failed",
    "cancelled",
)

LOG_PREFIX = re.compile(
    r"^\([^)]*\)\s*(ERROR|WARNING|INFO|DEBUG)?\s*\d\d-\d\d \d\d:\d\d:\d\d \[[^\]]+\]\s?"
)
STAMP = re.compile(r"^[A-Z][a-z]{2} [A-Z][a-z]{2} +\d+ \d\d:\d\d:\d\d \d{4}$")

OWNER = {
    "A": "vllm-qaic code",
    "B": "QAIC backend / hardware",
    "C": "CI config / credentials",
    "D": "CI config",
    "E": "environment",
    "F": "upstream HF / model repo",
    "G": "vllm-qaic or model code",
    "H": "environment",
}
CATEGORY = {
    "A": "A. vllm-qaic platform gap",
    "B": "B. QAIC device / runtime error",
    "C": "C. Model access / download",
    "D": "D. Config / harness / TP",
    "E": "E. Environment / deps",
    "F": "F. Processor / tokenizer / config mismatch",
    "G": "G. Shape / weight-loading / device-placement bug",
    "H": "H. Hexagon SDK install broken",
}
BLURB = {
    "A": "Missing capability in the `vllm_qaic` plugin, or a kernel/quantization scheme it never registers.",
    "B": "Real backend/hardware failures — the only group needing device-level debugging. Check the frame "
    "column: repeated frames mean one root cause across many models.",
    "C": "Nothing was ever downloaded. Note that HF returns **404, not 403**, for a repo the caller is not "
    "authorised to see, so a 404 means *either* a wrong/renamed `repo_id` *or* a private/gated repo this "
    "token cannot reach. Verify before pruning the model list.",
    "D": "Fixable in the CI model list / harness args.",
    "E": "Fix with `pip install`. No code changes needed.",
    "F": "Model remote code or repo metadata vs. the installed transformers/tokenizers. Mostly not yours to fix.",
    "G": "Genuine tensor-shape, weight-name or device-placement bugs needing investigation.",
    "H": "Missing native library. The worker retries for minutes, then is SIGKILLed and surfaces only as a "
    "generic cancellation.",
}


def clean(line: str) -> str:
    """Strip the '(Worker_TP2 pid=123) ERROR 08-26 11:56:30 [file.py:987] ' prefix."""
    m = LOG_PREFIX.match(line)
    return (line[m.end() :] if m else re.sub(r"^\([^)]*\)\s*", "", line)).strip()


def level(line: str) -> str:
    """The log level of a prefixed line, or '' for a bare (unprefixed) traceback line."""
    m = LOG_PREFIX.match(line)
    return (m.group(1) or "") if m else ""


def tp_advice(n_heads: int, tp: int) -> str:
    """Largest workable tp values below the one that failed."""
    cands = [c for c in (8, 4, 2, 1) if c < tp and n_heads % c == 0]
    return (
        " or ".join("tp=%d" % c for c in cands)
        if cands
        else "no smaller tp divides n_heads"
    )


def frame_before(cleaned: list[str], anchor: re.Pattern) -> tuple[str, str]:
    """Deepest interesting frame before `anchor`, plus the startup stage it sits in.

    'Interesting' means vllm_qaic / models / layers -- the innermost vllm plumbing
    frames (linear.py, attention/attention.py __init__) are the same for every
    model and tell you nothing.
    """
    i = next((j for j, line in enumerate(cleaned) if anchor.match(line)), 0)
    window = cleaned[max(0, i - 120) : i]
    frames = [line for line in window if line.startswith('File "')]
    interesting = [
        f for f in frames if "/vllm_qaic/" in f or "/models/" in f or "/layers/" in f
    ]
    pick = interesting[-1] if interesting else (frames[-1] if frames else "")
    m = re.search(r'([^/]+\.py)", line (\d+), in (\S+)', pick)
    frame = "{}:{} {}()".format(*m.groups()) if m else "?"

    back = "\n".join(window)
    load = any(
        k in back
        for k in ("in set_default_quant_scales", "in load_model", "load_weights")
    )
    return frame, ("model load" if load else "profile_run")


def root_error(cleaned: list[str], levels: list[str] | None = None) -> str:
    """First non-wrapper exception, with pydantic's real message pulled forward.

    Exceptions logged at WARNING level are *caught* ones the run survived -- e.g.
    vLLM's Mamba2 SSD kernel warm-up logs a full traceback via logger.warning and
    then dies several frames later on something else entirely. Those must never
    outrank the fatal error, so fatal-level lines are searched first and WARNING
    lines are consulted only if nothing else carries an exception.
    """

    def pick(subset: list[str]) -> str:
        excs = [
            (m.group(1), m.group(2)) for m in (EXC.match(line) for line in subset) if m
        ]
        if not excs:
            return ""
        kind, msg = next(
            (e for e in excs if not any(w in e[1] for w in WRAPPERS)), excs[0]
        )
        if "ValidationError" in kind:
            # pydantic prints "1 validation error for VllmConfig" and puts the real
            # complaint on the NEXT line. Without this the message is useless.
            i = next(
                (j for j, line in enumerate(subset) if "validation error for" in line),
                None,
            )
            if i is not None and i + 1 < len(subset):
                real = re.sub(r"^Value error, ", "", subset[i + 1]).split(" [type=")[0]
                return "ValidationError: " + real
        return "{}: {}".format(kind, msg)

    if levels is not None:
        fatal = [
            line
            for line, lv in zip(cleaned, levels, strict=False)
            if lv not in ("WARNING", "INFO", "DEBUG")
        ]
        found = pick(fatal)
        if found:
            return found
    return pick(cleaned)


def classify(
    model: str, tp: int, txt: str, cleaned: list[str], rc: str
) -> tuple[str, str, str]:
    """-> (category letter, short reason, terse fix note)."""

    def heads() -> tuple[int, int]:
        m = re.search(
            r"attention heads \((\d+)\) must be divisible by tensor parallel size \((\d+)\)",
            rc,
        )
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"(\d+) is not divisible by (\d+)", rc)
        return (int(m.group(1)), int(m.group(2))) if m else (0, tp)

    # --- H: checked before A, because the SDK failure masquerades as a cancellation
    if "libqhmath.a was not found" in txt:
        return (
            "H",
            "missing libqhmath.a",
            (
                "install Hexagon SDK lib: `libqhmath.a` missing from "
                "`/opt/qti-qic/dev/hexagon_sdk/libs/qhl/prebuilt/hexagon_toolv87_v68`"
            ),
        )

    # --- A: vllm-qaic platform gaps
    if "num_compute_units is not implemented" in rc:
        return (
            "A",
            "num_compute_units() not implemented",
            (
                "add a `num_compute_units()` classmethod to `QaicPlatform` (`vllm_qaic/platform_base.py`), "
                "delegating to `get_num_cores()`; guard the `is_aot` branch, which returns `None`"
            ),
        )
    if "QAic doesn't support MLA" in rc:
        return (
            "A",
            "MLA attention unsupported",
            "QAIC backend has no MLA attention support",
        )
    if "mrotary_embedding_eager_fallback" in rc:
        return (
            "A",
            "interleaved mRoPE unsupported",
            "QAIC has no interleaved mRoPE (`is_neox_style=False`) support",
        )
    if "QAicTorchAttentionBackend" in rc:
        return (
            "A",
            "attention backend unsupported",
            "`QAicTorchAttentionBackend` unsupported for this model",
        )
    if "BFloat16 datatype" in rc:
        return "A", "bfloat16 unsupported", "device rejects bf16 — force fp16 dtype"
    if "TORCH_SDPA backend now" in rc:
        return (
            "A",
            "non-SDPA backend required",
            "model rejects `TORCH_SDPA`; QAIC offers no alternative backend",
        )
    if "PlatformEnum.OOT" in rc:
        return (
            "A",
            "no scaled_mm kernel for OOT platform",
            (
                "no scaled_mm kernel registered for `PlatformEnum.OOT` "
                "(`vllm/model_executor/kernels/linear/__init__.py:495`) — register a QAIC W8A16-FP8 kernel"
            ),
        )
    m = re.search(r"'_C' object has no attribute '(\w+)'", rc)
    if m:
        return (
            "A",
            "missing _C op: {}".format(m.group(1)),
            ("missing kernel `_C::{}` — build/register the op".format(m.group(1))),
        )
    if "flash_attn_supports_sinks" in rc:
        return (
            "A",
            "attention sinks backend unavailable",
            (
                "model needs FLASH_ATTN_DIFFKV / attention sinks; `vllm_flash_attn` absent → "
                "`flash_attn_supports_sinks` undefined"
            ),
        )
    if "use_alibi_sqrt is not supported" in rc:
        return (
            "A",
            "ALiBi-sqrt unsupported",
            "`use_alibi_sqrt` unsupported by the QAIC CUSTOM attention backend",
        )
    m = re.search(r"(\S+) quantization is currently not supported", rc)
    if m:
        return (
            "A",
            "quant scheme unsupported: {}".format(m.group(1)),
            ("quantization `{}` not supported in qaic".format(m.group(1))),
        )
    if "Hybrid KV cache manager is disabled" in rc:
        return (
            "A",
            "hybrid KV cache unsupported",
            (
                "hybrid / sliding-window KV cache unsupported — QAIC cannot unify the KV cache specs"
            ),
        )

    # --- B: device / native runtime failures
    if re.search(r"malloc\(\): mismatching|corrupted size vs\. prev_size", txt):
        return (
            "B",
            "glibc heap corruption",
            (
                "glibc heap corruption (`malloc(): mismatching next->prev_size`) in the workers during model load"
            ),
        )
    # torch_qaic registers an allocator that is not a c10::DeviceAllocator, so any
    # torch.accelerator memory call hard-asserts. Must precede the Triton rules: the
    # Mamba2 models hit the caught SSD-warmup TypeError first, but die here.
    if "is not a DeviceAllocator" in rc:
        frame, stage = frame_before(cleaned, re.compile(r"is not a DeviceAllocator"))
        return (
            "B",
            "qaic allocator is not a c10::DeviceAllocator",
            (
                "`torch.accelerator.empty_cache()` asserts — torch_qaic's qaic allocator is not a "
                "`c10::DeviceAllocator`; register one (or make empty_cache a no-op for qaic). "
                "Raised from `{}` during **{}**".format(frame, stage)
            ),
        )
    if re.search(r"Error code: (500|9)\b", rc):
        frame, stage = frame_before(
            cleaned, re.compile(r"RuntimeError: Device id: \d+ QID")
        )
        code = "error 9 (send data to device)" if "Error code: 9" in rc else "error 500"
        return (
            "B",
            "{}, during {}".format(code, stage),
            ("QAIC device {} during **{}** at `{}`".format(code, stage, frame)),
        )

    # --- C: model access
    # A dropped connection yields no status code at all, so this must be checked
    # before the 403/404 rules -- the repo's real access state is simply unknown.
    if (
        "RemoteProtocolError" in rc
        or "Server disconnected without sending a response" in rc
    ):
        what = "processor" if "get_hf_processor" in txt else "config"
        return (
            "C",
            "transient HF hub disconnect",
            (
                "transient HF hub disconnect while fetching the {} (hf_hub_download already retried "
                "via http_backoff) — re-run; no status code was returned, so access state is unknown".format(
                    what
                )
            ),
        )
    if "403 Forbidden" in rc or "gated repo" in rc:
        return (
            "C",
            "gated repo (403)",
            (
                "gated repo — token is authenticated but not in the authorized list; request access on the model page"
            ),
        )
    if (
        "404 Not Found" in rc
        or "is not a local folder" in rc
        or "Invalid repository ID" in rc
    ):
        return (
            "C",
            "repo 404",
            (
                "HF 404 — verify the repo_id (renamed/removed?); if private or gated, the current token lacks permission"
            ),
        )
    if (
        "does not appear to have a file named" in rc
        or "Cannot find any model weights" in rc
    ):
        return (
            "C",
            "required file missing from repo",
            "required file missing from the repo — drop or pin a revision",
        )

    # --- D: config / harness / TP
    m = re.search(r"model type `(\w+)` but Transformers does not recognize", rc)
    if m or "Transformers does not recognize this architecture" in txt:
        mt = m.group(1) if m else "?"
        return (
            "D",
            "transformers too old (model_type={})".format(mt),
            (
                "transformers does not know `model_type={}` — needs a newer transformers".format(
                    mt
                )
            ),
        )
    if "must be divisible by tensor parallel size" in rc or "is not divisible by" in rc:
        n, t = heads()
        return (
            "D",
            "heads not divisible by tp",
            ("issue with tp_size, n_heads=%d (use %s)" % (n, tp_advice(n, t))),
        )
    if "only supported for generative models" in rc:
        return (
            "D",
            "runner resolved to pooling",
            (
                "vLLM auto-resolved `--runner auto` → `pooling`; harness must pass `--runner generate`"
            ),
        )
    if "No available memory for the cache blocks" in rc:
        return (
            "D",
            "no KV-cache memory",
            (
                "no KV-cache memory left — raise `gpu_memory_utilization` or lower `max_model_len`"
            ),
        )
    if "is greater than the derived max_model_len" in txt:
        return (
            "D",
            "max_model_len too large",
            "harness max_model_len exceeds the model max — lower it for this model",
        )
    if "Unexpected keyword argument" in txt:
        return (
            "D",
            "unsupported harness kwarg",
            "harness passes a `limit_per_prompt` key this vLLM rejects",
        )
    if "No model architectures are specified" in txt:
        return (
            "D",
            "no architectures in config.json",
            "config.json has no `architectures` field — pass `--hf-overrides`",
        )
    if "does not support tensor parallel" in rc:
        return "D", "no TP support", "no tensor-parallel support in vLLM — run tp=1"
    # The three rules below only fire once the engine actually reaches inference, so
    # they stay hidden behind any startup crash (e.g. num_compute_units) in the same run.
    if "Failed to apply prompt replacement" in rc:
        return (
            "D",
            "wrong image placeholder in harness prompt",
            (
                "harness image placeholder is not the token this processor expands — set a per-model "
                "`prompt` in `eager/model_configs_vlm.py` (defaults at `run_vlms.py:119`)"
            ),
        )
    if re.search(r"At most 0 image\(s\) may be provided", rc):
        return (
            "D",
            "audio-only model fed an image",
            (
                "audio-only model fed an image — harness always attaches `Cloud_AI_100.jpeg` "
                "(`run_vlms.py:115`); send an audio asset and set the audio limit instead"
            ),
        )
    m = re.search(
        r"decoder prompt \(length (\d+)\) is longer than the maximum model length of (\d+)",
        rc,
    )
    if m:
        return (
            "D",
            "image tokens exceed max_model_len",
            (
                "image expands the prompt to {} tokens > max_model_len={} — raise `max_model_len` "
                "in `eager/model_configs_vlm.py` (or shrink the 160x160 image limit)".format(
                    m.group(1), m.group(2)
                )
            ),
        )

    # --- E: environment
    # vLLM replaces the triton module with TritonPlaceholder whenever it finds no
    # active Triton driver. The placeholder's `jit` is an identity decorator, so
    # `kernel[grid](...)` raises "'function' object is not subscriptable", and it
    # omits next_power_of_2 entirely. On a QAIC host this means the stock PyPI triton
    # wheel has replaced the Qualcomm build that ships
    # triton.backends.qcom_hexagon_backend -- check for that module before blaming
    # the model. Only Mamba/SSM models reach a Triton kernel, so this hides easily.
    if (
        "no attribute 'next_power_of_2'" in rc
        or "'function' object is not subscriptable" in rc
    ):
        return (
            "E",
            "Triton disabled, placeholder in use",
            (
                "vLLM fell back to `TritonPlaceholder` (0 active Triton drivers), breaking every "
                "Triton kernel — reinstall the Qualcomm Triton that provides "
                "`triton.backends.qcom_hexagon_backend`; the stock PyPI `triton` wheel overwrites it"
            ),
        )
    if "not supported in your version of timm" in rc or "Unknown model (" in rc:
        return (
            "E",
            "timm too old",
            "timm too old for `mobilenetv5_300m_enc` — `pip install -U timm`",
        )
    if "were not found in your environment" in rc or "No module named" in rc:
        m = re.search(r"environment: ([\w, ]+)", rc) or re.search(
            r"No module named '([\w]+)'", rc
        )
        pkg = m.group(1) if m else "?"
        return "E", "missing package: {}".format(pkg), "missing package {}".format(pkg)

    # --- G: shape / weight loading / device placement
    if "Expected all tensors to be on the same device" in rc:
        frame, _ = frame_before(
            cleaned, re.compile(r"RuntimeError: Expected all tensors")
        )
        return (
            "G",
            "cpu/qaic tensor device mismatch",
            (
                "cpu/qaic device mismatch at `{}` — a tensor built on CPU meets a qaic tensor; "
                "it needs `.to(x.device)`".format(frame)
            ),
        )
    if re.search(r"is invalid for input of size|view size is not compatible", rc):
        return (
            "G",
            "tensor shape mismatch",
            "tensor shape mismatch — {}".format(rc[:110]),
        )
    if "no module or parameter named" in rc:
        m = re.search(r"named '(\w+)' in (\w+)", rc)
        return (
            "G",
            "checkpoint param names mismatch",
            (
                "%s param-name mismatch: checkpoint provides `%s.*` — vLLM weight-loader mismatch"
                % ((m.group(2), m.group(1)) if m else ("model", "?"))
            ),
        )

    # --- F: tokenizer / processor / remote-code vs installed transformers
    if any(
        k in rc
        for k in (
            "piece must not include null",
            "prepend_scheme",
            "add_prefix_space",
            "Couldn't instantiate the backend tokenizer",
        )
    ):
        return (
            "F",
            "tokenizer will not load",
            (
                "tokenizer will not load with the installed tokenizers/transformers — {}".format(
                    rc[:90]
                )
            ),
        )
    if "Can not find <image>, please add" in rc:
        return (
            "F",
            "special tokens missing from tokenizer_config",
            (
                "processor: `<image>` and friends missing from `additional_special_tokens` in tokenizer_config.json"
            ),
        )
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", rc)
    if m:
        return (
            "F",
            "removed transformers API: {}".format(m.group(2)),
            (
                "remote code calls `{}.{}` — removed from the installed transformers".format(
                    m.group(1), m.group(2)
                )
            ),
        )
    m = re.search(
        r"'(\w*Config)' object has no attribute '(\w+)'(?:\. Did you mean: '(\w+)')?",
        rc,
    )
    if m:
        renamed = ", renamed to `{}`".format(m.group(3)) if m.group(3) else ""
        return (
            "F",
            "renamed config field: {}".format(m.group(2)),
            ("remote code uses `{}.{}`{}".format(m.group(1), m.group(2), renamed)),
        )
    m = re.search(r"'(\w*Processor)' object has no attribute '(\w+)'", rc)
    if m:
        return (
            "F",
            "processor missing attribute: {}".format(m.group(2)),
            ("`{}` has no `{}` attribute".format(m.group(1), m.group(2))),
        )
    m = re.search(r"output must include `(\w+)`", rc)
    if m:
        return (
            "F",
            "processor output missing key: {}".format(m.group(1)),
            ("processor output missing required key `{}`".format(m.group(1))),
        )
    if "Invalid type of HuggingFace processor" in rc:
        m = re.search(r"found type: <class '([\w.]+)'", rc)
        return (
            "F",
            "AutoProcessor returns wrong type",
            (
                "AutoProcessor returns %s, not `ProcessorMixin`"
                % (
                    "`{}`".format(m.group(1).rsplit(".", 1)[-1])
                    if m
                    else "the wrong type"
                )
            ),
        )
    if "you instantiated the text-only version of this model" in rc:
        return (
            "F",
            "text-only arch selected for a vision model",
            ("pass `--hf-overrides` to select the vision architecture"),
        )
    return (
        "F",
        "unclassified",
        rc[:130] or "no exception captured — inspect the log by hand",
    )


def load(directory: str) -> tuple[list[dict], list[datetime.datetime]]:
    rows, stamps = [], []
    for path in sorted(glob.glob(os.path.join(directory, "log_1_*.log"))):
        model = os.path.basename(path)[len("log_1_") : -len(".log")]
        with open(path, errors="replace") as log_file:
            txt = log_file.read()
        lines = txt.splitlines()
        cleaned = [clean(line) for line in lines]
        levels = [level(line) for line in lines]

        for cand in (
            lines[1] if len(lines) > 1 else "",
            lines[-2] if len(lines) > 2 else "",
        ):
            if STAMP.match(cand.strip()):
                stamps.append(
                    datetime.datetime.strptime(cand.strip(), "%a %b %d %H:%M:%S %Y")
                )

        # The logged --tp-size is only the sweep default; run_{llm,vlm}s.py re-resolves
        # the pinned value via model_configs_*.get_tp_size(). The FILENAME carries the
        # effective tp -- verified against runtime tensor_parallel_size=N.
        tp = int(model.rsplit("_tp", 1)[1])
        hf = (
            (re.search(r"--model-name\s+(\S+)", lines[0]) or [None, model])[1]
            if lines
            else model
        )

        parse = os.path.join(directory, "parse_1_{}.log".format(model))
        ops = []
        if os.path.exists(parse):
            with open(parse, errors="replace") as parse_file:
                ptxt = parse_file.read()
            # An empty defaultdict prints as ", {})". Do NOT try to regex the
            # <lambda at 0x...> address -- that spelling varies.
            if ", {})" not in ptxt:
                ops = re.findall(r"'(aten::[^']+)': \{'tc': (\d+)", ptxt)

        ok = "python exited with status code" not in txt
        rc = "" if ok else root_error(cleaned, levels)
        if ok:
            letter, reason, fixnote = (
                "0",
                "ran to completion, generated output",
                "n/a — passed",
            )
        else:
            letter, reason, fixnote = classify(model, tp, txt, cleaned, rc)

        rows.append(
            dict(
                model=model,
                hf=hf,
                tp=tp,
                letter=letter,
                reason=reason,
                fix=fixnote,
                root_error=rc[:220],
                completed="yes" if ok else "no",
                fallback_ops=", ".join("{} ({})".format(o, c) for o, c in ops),
            )
        )
    return rows, stamps


def write_csv(directory: str, rows: list[dict]) -> str:
    out = os.path.join(directory, "failure_categories.csv")
    cols = [
        "category",
        "reason",
        "model",
        "hf",
        "tp",
        "fallback_ops",
        "completed",
        "fix",
        "root_error",
    ]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(
            rows, key=lambda r: (r["letter"], r["reason"], r["hf"].lower())
        ):
            w.writerow(
                {
                    **{c: r.get(c, "") for c in cols},
                    "category": CATEGORY.get(r["letter"], "0. SUCCESS"),
                }
            )
    return out


def write_md(
    directory: str, rows: list[dict], stamps: list[datetime.datetime], title: str
) -> str:
    fails = [r for r in rows if r["completed"] == "no"]
    succ = [r for r in rows if r["completed"] == "yes"]
    withops = [r for r in rows if r["fallback_ops"]]
    counts = collections.Counter(r["letter"] for r in fails)
    L: list[str] = []
    A = L.append

    A("# vLLM-QAIC {} Fallback CI — Failure Report".format(title))
    A("")
    if stamps:
        A(
            "Run window: {} → {}  ({})".format(
                min(stamps).strftime("%Y-%m-%d %H:%M"),
                max(stamps).strftime("%Y-%m-%d %H:%M"),
                max(stamps) - min(stamps),
            )
        )
        A("")
    A(
        "**%d model runs — %d completed, %d exited with status code 1.**"
        % (len(rows), len(succ), len(fails))
    )
    A("")
    A(
        "**%d of %d runs logged CPU fallback-op data** (the actual point of this sweep)."
        % (len(withops), len(rows))
    )
    A("")
    if withops:
        A("| Run | Ops logged | Outcome |")
        A("|---|---|---|")
        for r in sorted(withops, key=lambda r: r["hf"].lower()):
            A(
                "| `{}` | {} | {} |".format(
                    r["hf"],
                    r["fallback_ops"],
                    "passed"
                    if r["completed"] == "yes"
                    else "failed — {}".format(r["reason"]),
                )
            )
        A("")

    A("## Summary")
    A("")
    A("| Category | Runs | Fix owner |")
    A("|---|---:|---|")
    for letter in sorted(counts):
        A("| %s | %d | %s |" % (CATEGORY[letter][3:], counts[letter], OWNER[letter]))
    A("| SUCCESS | %d | — |" % len(succ))
    A("| **Total** | **%d** | |" % len(rows))
    A("")
    A("## Biggest single wins")
    A("")
    grouped = collections.Counter((r["letter"], r["reason"]) for r in fails)
    for (letter, reason), n in grouped.most_common(6):
        A("- **%d runs** — %s *(%s)*" % (n, reason, CATEGORY[letter][3:]))
    A("")
    A(
        "> A crash during memory profiling or model load happens *before* real inference, so clearing it "
        "does not guarantee those models pass — expect a second wave of per-model failures."
    )
    A("")
    A("---")
    A("")

    for letter in sorted(counts):
        group = [r for r in fails if r["letter"] == letter]
        A(
            "## %s — %d run%s"
            % (CATEGORY[letter], len(group), "" if len(group) == 1 else "s")
        )
        A("")
        A(BLURB[letter])
        A("")
        by_fix = collections.Counter(r["fix"] for r in group)
        # One fix shared by a dozen+ models: a bullet list under a single heading
        # reads far better than a dozen identical table rows.
        bulk = [f for f, n in by_fix.items() if n >= 12]
        for f in bulk:
            members = sorted(
                [r for r in group if r["fix"] == f], key=lambda r: r["hf"].lower()
            )
            A("### %d runs — one shared fix" % len(members))
            A("")
            A("**Fix:** {}".format(f))
            A("")
            for r in members:
                A("- `%s` (tp=%d)" % (r["hf"], r["tp"]))
            A("")
        rest = sorted(
            [r for r in group if r["fix"] not in bulk], key=lambda r: r["hf"].lower()
        )
        if rest:
            if bulk:
                A("### Remaining %d" % len(rest))
                A("")
            A("| Model | Fix |")
            A("|---|---|")
            for r in rest:
                A("| `%s` (tp=%d) | %s |" % (r["hf"], r["tp"], r["fix"]))
            A("")

    A("## 0. SUCCESS — %d run%s" % (len(succ), "" if len(succ) == 1 else "s"))
    A("")
    A("Ran to completion and generated output.")
    A("")
    if succ:
        A("<details><summary>%d models</summary>" % len(succ))
        A("")
        for r in sorted(succ, key=lambda r: r["hf"].lower()):
            A(
                "- `%s` (tp=%d)%s"
                % (
                    r["hf"],
                    r["tp"],
                    " — ops: {}".format(r["fallback_ops"]) if r["fallback_ops"] else "",
                )
            )
        A("")
        A("</details>")
        A("")

    pkgs = sorted(
        {
            r["fix"].replace("missing package ", "")
            for r in fails
            if r["reason"].startswith("missing package")
        }
    )
    if pkgs:
        A("## Environment fix-list")
        A("")
        A("```bash")
        A("# NB: import name != pip name for several of these (kaldi_native_fbank ->")
        A("# kaldi-native-fbank, open_clip -> open_clip_torch). Check before pasting.")
        A("pip install " + " ".join(pkgs))
        A("```")
        A("")

    fourohfour = [r for r in fails if r["reason"] == "repo 404"]
    if fourohfour:
        A("## Models returning HF 404 — verify before pruning")
        A("")
        A(
            "A 404 does **not** prove the repo is gone: HF returns 404 for repos the caller is not authorised "
            "to see. Check well-known names by hand rather than deleting them from the model list."
        )
        A("")
        for r in sorted(fourohfour, key=lambda r: r["hf"].lower()):
            A("- `{}`".format(r["hf"]))
        A("")

    A("---")
    A("")
    A(
        "Generated from %d `log_1_*.log` / `parse_1_*.log` pairs in `%s`. "
        "Machine-readable version: `failure_categories.csv`." % (len(rows), directory)
    )

    out = os.path.join(directory, "FAILURE_REPORT.md")
    with open(out, "w") as report_file:
        report_file.write("\n".join(L) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Triage a vllm-qaic Fallback CI sweep.")
    ap.add_argument(
        "logdir", help="directory holding log_1_*.log / parse_1_*.log pairs"
    )
    ap.add_argument(
        "--title",
        default=None,
        help="report heading, e.g. LLM or VLM (default: dir name upper)",
    )
    ap.add_argument(
        "--stdout", action="store_true", help="print the summary only; write nothing"
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing FAILURE_REPORT.md / failure_categories.csv",
    )
    args = ap.parse_args()

    directory = os.path.abspath(args.logdir)
    if not os.path.isdir(directory):
        print("error: {} is not a directory".format(directory), file=sys.stderr)
        return 1

    rows, stamps = load(directory)
    if not rows:
        print("error: no log_1_*.log files in {}".format(directory), file=sys.stderr)
        return 1

    fails = [r for r in rows if r["completed"] == "no"]
    print(
        "Runs: %d | completed: %d | logged fallback ops: %d\n"
        % (len(rows), len(rows) - len(fails), sum(1 for r in rows if r["fallback_ops"]))
    )
    counts = collections.Counter(r["letter"] for r in fails)
    sub = collections.Counter((r["letter"], r["reason"]) for r in fails)
    for letter in sorted(counts):
        print("%-46s %3d" % (CATEGORY[letter], counts[letter]))
        for (lt, reason), n in sorted(sub.items(), key=lambda kv: -kv[1]):
            if lt == letter:
                print("      %3d  %s" % (n, reason))
    unknown = sum(1 for r in fails if r["reason"] == "unclassified")
    if unknown:
        print(
            "\n!! %d unclassified -- add rules for these before trusting the report"
            % unknown
        )

    if not args.stdout:
        # A delivered report usually carries hand-written sections this script cannot
        # regenerate (fix ordering, environment notes, cross-sweep caveats). Clobbering
        # one silently has already destroyed a finished report once, so refuse by default.
        existing = [
            p
            for p in ("FAILURE_REPORT.md", "failure_categories.csv")
            if os.path.exists(os.path.join(directory, p))
        ]
        if existing and not args.force:
            print(
                "\nrefusing to overwrite {} in {}".format(
                    " and ".join(existing), directory
                ),
                file=sys.stderr,
            )
            print(
                "re-run with --stdout to only print the summary, or --force to overwrite.",
                file=sys.stderr,
            )
            return 1
        title = args.title or os.path.basename(directory).upper()
        print("\nCSV      -> {}".format(write_csv(directory, rows)))
        print("Report   -> {}".format(write_md(directory, rows, stamps, title)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
