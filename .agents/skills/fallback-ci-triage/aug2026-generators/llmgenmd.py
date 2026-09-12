# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# ruff: noqa
import csv
import os
import regex as re
import collections

D = "/local/mnt5/workspace/agokhale/fallback-ci/vllm-qaic/ci_scripts/ci_logs/llm"
rows = [r for r in csv.DictReader(open(os.path.join(D, "failure_categories.csv")))]


def clean(l):
    m = re.match(
        r"^\([^)]*\)\s*(?:ERROR|WARNING|INFO|DEBUG)?\s*\d\d-\d\d \d\d:\d\d:\d\d \[[^\]]+\]\s?",
        l,
    )
    return (l[m.end() :] if m else re.sub(r"^\([^)]*\)\s*", "", l)).strip()


def hf_and_tp(model):
    # tp comes from the FILENAME: the logged --tp-size is only the sweep default (4);
    # run_llms.py re-resolves the pinned value via model_configs_llm.get_tp_size().
    p = os.path.join(D, "log_1_{}.log".format(model))
    l1 = open(p, errors="replace").readline()
    m = re.search(r"--model-name\s+(\S+)", l1)
    return (m.group(1) if m else model), model.rsplit("_tp", 1)[1]


def frame(model):
    """Deepest interesting frame before the device-error line."""
    cl = [
        clean(l)
        for l in open(os.path.join(D, "log_1_{}.log".format(model)), errors="replace")
        .read()
        .splitlines()
    ]
    i = next(
        (
            j
            for j, l in enumerate(cl)
            if re.match(r"RuntimeError: Device id: \d+ QID", l)
        ),
        0,
    )
    fr = [l for l in cl[max(0, i - 120) : i] if l.startswith('File "')]
    interesting = [
        f for f in fr if "/vllm_qaic/" in f or "/models/" in f or "/layers/" in f
    ]
    pick = interesting[-1] if interesting else (fr[-1] if fr else "")
    m = re.search(r'([^/]+\.py)", line (\d+), in (\S+)', pick)
    return "{}:{} {}()".format(*m.groups()) if m else "?"


HEADS = {
    "allenai_Olmo-Hybrid-7B_tp4": (30, "use tp=2 or tp=1"),
    "inceptionai_Jais-2-8B-Chat_tp4": (26, "use tp=2 or tp=1"),
    "tiiuae_Falcon-H1-34B-Base_tp8": (20, "use tp=4, tp=2 or tp=1"),
}
TOK = {
    "internlm_internlm-7b_tp4": "tokenizer fails to load: sentencepiece piece contains a null character — needs matching tokenizers/transformers",
    "xverse_XVERSE-7B-Chat_tp4": "tokenizer.json `add_prefix_space` vs declared `prepend_scheme` mismatch (line 78) — needs matching tokenizers version",
}


def fix(r):
    m, s = r["model"], r["reason"]
    if s.startswith("MLA"):
        return "QAIC backend has no MLA attention support"
    if s.startswith("bfloat16"):
        return "device rejects bf16 — force fp16 dtype"
    if s.startswith("no scaled_mm"):
        return "no scaled_mm kernel registered for `PlatformEnum.OOT` (`vllm/model_executor/kernels/linear/__init__.py:495`) — register a QAIC W8A16-FP8 kernel"
    if s.startswith("Mamba/SSM"):
        return (
            "missing kernel `_C::selective_scan_fwd` — build/register the Mamba SSM op"
        )
    if s.startswith("FP8 kernel"):
        return "missing kernel `_C::static_scaled_fp8_quant` — build/register the FP8 quant op"
    if s.startswith("FLASH_ATTN_DIFFKV"):
        return "model needs FLASH_ATTN_DIFFKV / attention sinks; `vllm_flash_attn` absent → `flash_attn_supports_sinks` undefined"
    if s.startswith("hybrid"):
        return "hybrid / sliding-window KV cache unsupported — QAIC cannot unify the KV cache specs"
    if s.startswith("quantization scheme"):
        return "quantization `{}` not supported in qaic".format(s.split(": ")[1])
    if s.startswith("ALiBi"):
        return "`use_alibi_sqrt` unsupported by the QAIC CUSTOM attention backend"
    if s.startswith("glibc"):
        return "glibc heap corruption (`malloc(): mismatching next->prev_size`) in all 4 workers during model load"
    if s.startswith("error 500"):
        st = "**profile_run**" if "profiling" in s else "**model load**"
        return "QAIC device error 500 during {} at `{}`".format(st, frame(m))
    if s.startswith("error 9"):
        return "QAIC device error 9 (send data to device) during **model load** at `{}`".format(
            frame(m)
        )
    if s.startswith("gated"):
        return "gated repo — token is authenticated but not in the authorized list; request access on the model page"
    if s.startswith("repo not found"):
        return "HF 404 — verify the repo_id (renamed/removed?); if private or gated, the current token lacks permission"
    if s.startswith("attention heads"):
        n, adv = HEADS[m]
        return "issue with tp_size, n_heads=%d (%s)" % (n, adv)
    if s.startswith("embedding/pooling"):
        return "vLLM auto-resolved `--runner auto` → `pooling`; harness must pass `--runner generate`"
    if s.startswith("no KV-cache"):
        return "no KV-cache memory left — raise `gpu_memory_utilization` or lower `max_model_len`"
    if s.startswith("transformers too old"):
        return "transformers 5.5.4 does not know `model_type=axk1` — needs newer transformers"
    if s.startswith("timm"):
        return "timm 1.0.14 too old for `mobilenetv5_300m_enc` — `pip install -U timm`"
    if s.startswith("tokenizer will not load"):
        return TOK[m]
    if s.startswith("checkpoint param names"):
        return "`Mamba2ForCausalLM` expects `backbone.*` params, checkpoint provides `model.*` — vLLM weight-loader mismatch"
    if s.startswith("ran to completion"):
        return "n/a — passed"
    return r["root_error"][:110]


for r in rows:
    r["hf"], r["tp"] = hf_and_tp(r["model"])
    r["fix"] = fix(r)

fails = [r for r in rows if r["completed"] == "no"]
succ = [r for r in rows if r["completed"] == "yes"]
cc = collections.Counter(r["category"] for r in fails)

OWNER = {
    "A. vllm-qaic platform gap": "vllm-qaic code",
    "B. QAIC device / runtime error": "QAIC backend / hardware",
    "C. Model access / download": "CI config / credentials",
    "D. Config / harness / TP": "CI config",
    "E. Environment / deps": "environment",
    "F. Tokenizer / config mismatch": "upstream HF / model repo",
    "G. Weight-loading bug": "vLLM model code",
}
TITLE = {
    "A. vllm-qaic platform gap": "A. vllm-qaic platform gap",
    "B. QAIC device / runtime error": "B. QAIC device / runtime error",
    "C. Model access / download": "C. Model access / download",
    "D. Config / harness / TP": "D. Config / harness / TP",
    "E. Environment / deps": "E. Environment / deps",
    "F. Tokenizer / config mismatch": "F. Tokenizer / config mismatch",
    "G. Weight-loading bug": "G. Weight-loading bug",
}
BLURB = {
    "A. vllm-qaic platform gap": "Missing capability in the `vllm_qaic` plugin or in its kernel/quantization registration. No single dominant gap here — unlike the VLM sweep, these are 10 distinct issues.",
    "B. QAIC device / runtime error": "Real backend/hardware failures — the group needing device-level debugging. 21 of 25 die during model load; the failing frame is almost always a plain tensor allocation on the device (quant-scale defaults, ALiBi slopes, layernorm weights), and 14 of them at the same frame. Mostly very large models (70B–675B), consistent with device memory exhaustion surfacing as a stream-sync error.",
    "C. Model access / download": "Nothing was ever downloaded. A working HF token **is** present: the 3 gated repos fail with `403 ... you are not in the authorized list`, i.e. authenticated but not approved. The 8 `404`s are ambiguous by design \u2014 HF returns 404 rather than 403 for a repo the caller may not know exists, so a 404 means *either* a wrong/renamed `repo_id` *or* a private/gated repo this token cannot see.",
    "D. Config / harness / TP": "Fixable in the CI model list / harness args.",
    "E. Environment / deps": "Fix with `pip install`. No code changes needed.",
    "F. Tokenizer / config mismatch": "Model repo tokenizer files vs. installed `tokenizers`/`transformers`. Not yours to fix.",
    "G. Weight-loading bug": "Genuine weight-name mismatch needing investigation.",
}

L = []
A = L.append
A("# vLLM-QAIC LLM Fallback CI — Failure Report")
A("")
A(
    "Run date: 2026-08-25 18:04 → 2026-08-26 05:15  |  vLLM 0.23.0  |  `vllm_qaic` 0.23.0.dev25+gf9ec83794  |  transformers 5.5.4  |  `QAIC_VISIBLE_DEVICES=32..39`"
)
A("")
A(
    "**148 model runs — 77 completed, 71 exited with status code 1.** The pass rate is far better than the VLM sweep (77/148 vs 1/116): none of the LLM runs hit the `num_compute_units` gap that killed 45 VLM runs."
)
A("")
A(
    "**But the sweep still collected almost no fallback-op data: only 2 of 148 runs logged a single `CPU fallback op`, and both of those runs crashed.** All 77 successful runs recorded zero CPU fallbacks."
)
A("")
A("| Run | Ops logged | Outcome |")
A("|---|---|---|")
A(
    "| `mistralai/Ministral-3-3B-Instruct-2512` | `aten::mul.out` (4 calls) | failed — missing `_C::static_scaled_fp8_quant` |"
)
A(
    "| `rednote-hilab/dots.ocr` | `aten::repeat_interleave.Tensor`, `aten::index_select`, `aten::bmm.out` | failed — QAIC device error 500 |"
)
A("")
A(
    "Both are atypical models (an FP8 checkpoint and a vision-language model). The 77 plain-decoder LLMs that ran clean produced no fallbacks at all — so for this model class the sweep is confirming full on-device coverage rather than finding gaps."
)
A("")
A("## Summary")
A("")
A("| Category | Runs | Fix owner |")
A("|---|---:|---|")
for c, n in sorted(cc.items()):
    A("| %s | %d | %s |" % (c[3:], n, OWNER[c]))
A("| SUCCESS | %d | — |" % len(succ))
A("| **Total** | **%d** | |" % len(rows))
A("")
A("## Fix order")
A("")
A(
    "1. **Triage the 25 device errors** — 14 share one frame (`attention.py:97 set_default_quant_scales()`) and 3 more share `ssd_chunk_state.py:379`. Two root causes may clear 17 runs."
)
A(
    "2. **Register a scaled_mm kernel for `PlatformEnum.OOT`** → 2 runs, and unblocks every compressed-tensors FP8 checkpoint. (Work already in flight: `vllm_qaic/quantization/qaic_fp8_scaled_mm.py`, `qaic_fp8_block_scaled_mm.py`, `patch/patch_fp8_scaled_mm.py`.)"
)
A("3. **Fix tp_size for 3 models + pass `--runner generate` for 2 more** → 5 runs.")
A(
    "4. **Request access for the 3 gated repos; verify the 8 `404` names** → 11 runs stop consuming slots."
)
A(
    "5. **Build the missing `_C` ops** (`selective_scan_fwd`, `static_scaled_fp8_quant`) → 3 runs, and is the only path to Mamba/SSM coverage."
)
A(
    "6. Remaining: 8 MLA models (needs an MLA implementation), 3 bf16, 3 KV-cache-memory, 2 attention-sinks, 2 unsupported quant schemes, 2 tokenizer, 1 hybrid KV, 1 ALiBi-sqrt, 1 weight-loader, 1 transformers version."
)
A("")
A(
    "> Note: the `error 500` failures are **not** cross-sweep device contention. The LLM sweep ran on `QAIC_VISIBLE_DEVICES=32..39` and the concurrent VLM sweep on `40..47`, so the two never shared a device."
)
A("")
A("---")
A("")
for c in sorted(cc):
    A("## %s — %d run%s" % (TITLE[c], cc[c], "" if cc[c] == 1 else "s"))
    A("")
    A(BLURB[c])
    A("")
    A("| Model | Fix |")
    A("|---|---|")
    for r in sorted(
        [x for x in fails if x["category"] == c], key=lambda x: x["hf"].lower()
    ):
        A("| `{}` (tp={}) | {} |".format(r["hf"], r["tp"], r["fix"]))
    A("")
A("## 0. SUCCESS — %d runs" % len(succ))
A("")
A("Ran to completion and generated output. **None logged a CPU fallback op.**")
A("")
A("<details><summary>%d models</summary>" % len(succ))
A("")
for r in sorted(succ, key=lambda x: x["hf"].lower()):
    A("- `{}` (tp={})".format(r["hf"], r["tp"]))
A("")
A("</details>")
A("")
A("---")
A("")
A("## Environment fix-list")
A("")
A("```bash")
A("pip install -U timm            # 1.0.14 lacks mobilenetv5_300m_enc (gemma-3n)")
A("pip install -U transformers    # 5.5.4 lacks model_type=axk1 (skt/A.X-K1)")
A("# no missing packages beyond these two; the HF token is already valid,")
A("# but 3 gated repos still need access approval on their model pages.")
A("```")
A("")
A(
    "Everything else in this sweep is a code or config fix, not a package install — a much shorter environment list than the VLM sweep."
)
A("")
A("## Models returning HF 404 — verify before pruning")
A("")
A(
    "A 404 here does **not** prove the repo is gone: HF returns 404 for repos the caller is not authorized to see. `databricks/dbrx-base` is a gated repo, and `databricks/dolly-v2-12b` / `mosaicml/mpt-7b` were long-standing public repos \u2014 check those three by hand rather than deleting them from the model list."
)
A("")
for r in sorted(
    [x for x in fails if x["reason"].startswith("repo not found")],
    key=lambda x: x["hf"].lower(),
):
    A("- `{}`".format(r["hf"]))
A("")
A("## Models needing `--runner generate`")
A("")
for r in sorted(
    [x for x in fails if x["reason"].startswith("embedding/pooling")],
    key=lambda x: x["hf"].lower(),
):
    A(
        "- `{}` — `--runner auto` resolves to `pooling`, so `LLM.generate()` refuses to run".format(
            r["hf"]
        )
    )
A("")
A("---")
A("")
A(
    "Generated from 148 `log_1_*.log` / `parse_1_*.log` pairs in `ci_scripts/ci_logs/llm/`. Machine-readable version: `failure_categories.csv`."
)

out = os.path.join(D, "FAILURE_REPORT.md")
open(out, "w").write("\n".join(L) + "\n")
print("wrote %s (%d lines)" % (out, len(L)))
