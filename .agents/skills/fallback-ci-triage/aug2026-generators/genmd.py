# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# ruff: noqa
import csv
import os
import collections

D = "/local/mnt5/workspace/agokhale/fallback-ci/vllm-qaic/ci_scripts/ci_logs/vlm"
rows = list(csv.DictReader(open(os.path.join(D, "failure_categories.csv"))))

import regex as re


def invocation(model):
    first = open(
        os.path.join(D, "log_1_{}.log".format(model)), errors="replace"
    ).readline()
    mn = re.search(r"--model-name (\S+)", first)
    # NOTE: the logged "--tp-size" is only the sweep default; run_vlms.py:156 re-resolves
    # the pinned value via model_configs_vlm.get_tp_size(). The filename carries the
    # effective tp (verified: log_..._tp1 <-> runtime tensor_parallel_size=1).
    return (mn.group(1) if mn else model), model.rsplit("_tp", 1)[1]


for r in rows:
    r["hf_name"], r["tp"] = invocation(r["model"])

PKG = {
    "allenai_Molmo-7B-D-0924_tp4": "tensorflow",
    "allendou_FireRedASR2-LLM-vllm_tp4": "kaldi_native_fbank",
    "baidu_ERNIE-4.5-VL-28B-A3B-PT_tp4": "decord",
    "nvidia_Llama-3.1-Nemotron-Nano-VL-8B-V1_tp4": "open_clip",
    "PatchyTisa_FireRedLID-vllm_tp4": "kaldi_native_fbank",
    "Qwen_Qwen-VL_tp4": "matplotlib",
}
HEADS = {
    "deepseek-ai_deepseek-vl2-tiny_tp4": 10,
    "llava-hf_llava-onevision-qwen2-0.5b-ov-hf_tp4": 14,
    "Qwen_Qwen3-ASR-0.6B_tp4": 14,
}
MTYPE = {
    "LGAI-EXAONE_EXAONE-4.5-33B_tp4": "exaone4_5",
    "google_gemma-4-12B-it_tp4": "gemma4_unified",
    "ibm-granite_granite-speech-4.1-2b-plus_tp4": "granite_speech_plus",
}
BSTAGE = {
    "MiniMaxAI_MiniMax-VL-01_tp4": (
        "model load",
        "minimax_linear_attn.py:229 _build_slope_tensor()",
    ),
    "Qwen_QVQ-72B-Preview_tp4": (
        "model load",
        "attention.py:97 set_default_quant_scales()",
    ),
    "Qwen_Qwen3-ASR-1.7B_tp4": (
        "profile_run",
        "qwen3_omni_moe_thinker.py:492 forward()",
    ),
    "ai9stars_Cheers_tp4": ("profile_run", "F.group_norm()"),
    "allendou_Fun-ASR-Nano-2512-vllm_tp4": (
        "profile_run",
        "funasr.py:200 forward_fsmn()",
    ),
    "ibm-granite_granite-speech-3.3-8b_tp4": ("profile_run", "c10d all_reduce()"),
    "internlm_Intern-S1-Pro_tp4": (
        "model load",
        "attention.py:97 set_default_quant_scales()",
    ),
    "internlm_Intern-S1_tp4": (
        "model load",
        "attention.py:97 set_default_quant_scales()",
    ),
    "kakaocorp_kanana-1.5-v-3b-instruct_tp4": ("profile_run", "F.conv2d()"),
    "nvidia_NVLM-D-72B_tp4": (
        "model load",
        "attention.py:97 set_default_quant_scales()",
    ),
    "stepfun-ai_Step-3.7-Flash_tp4": (
        "model load",
        "attention.py:97 set_default_quant_scales()",
    ),
    "stepfun-ai_step3_tp4": (
        "model load",
        "attention.py:97 set_default_quant_scales()",
    ),
    "zai-org_GLM-4.5V_tp4": (
        "model load",
        "attention.py:97 set_default_quant_scales()",
    ),
}
FNOTE = {
    "AIDC-AI_Ovis2.5-9B_tp8": "processor: `<image>` missing from `additional_special_tokens` in tokenizer_config.json",
    "AIDC-AI_Ovis2.6-30B-A3B_tp4": "processor: `<image>` missing from `additional_special_tokens` in tokenizer_config.json",
    "microsoft_Phi-4-reasoning-vision-15B_tp1": "remote code calls `image_processing_siglip2.filter_out_non_signature_kwargs` — removed in transformers 5.5.4",
    "moondream_moondream3-preview_tp4": "tokenizer will not instantiate (needs matching tokenizers/transformers)",
    "nvidia_music-flamingo-2601-hf_tp4": "processor output missing required key `rote_timestamps`",
    "openbmb_MiniCPM-V-2_tp4": "AutoProcessor returns TokenizersBackend, not ProcessorMixin",
    "OpenGVLab_InternVideo2_5_Chat_8B_tp4": "remote code uses `InternVLChatConfig.llm_config`, renamed to `sub_configs`",
    "xlangai_OpenCUA-7B_tp4": "`OpenCUAProcessor` has no `image_processor` attribute",
    "zai-org_glm-4v-9b_tp4": 'add `--hf-overrides \'{"architectures": ["GLM4VForCausalLM"]}\'`',
}
GNOTE = {
    "OpenGVLab_InternVL3-9B_tp4": "reshape to `[8192, 2, 10, 128]` (=20.9M) from input of 41,943,040 elems (2x too big)",
    "mispeech_midashenglm-7b_tp4": "reshape to `[163, 250, 3, 4, 320]` (=156.5M) from input of 39,120,000 elems",
    "zai-org_GLM-ASR-Nano-2512_tp4": "non-contiguous `.view()` — needs `.reshape()`",
}
CNOTE = {
    "PerceptronAI_Isaac-0.1_tp4": "no model weights present in repo",
    "rhymes-ai_Aria_tp4": "repo is missing `vision_processor.py` referenced by its remote code",
}


def fix(r):
    m, sub = r["model"], r["reason"]
    if r["category"].startswith("0."):
        return "n/a — passed"
    if "num_compute_units" in sub:
        return "implement `num_compute_units()` in `vllm_qaic/platform_base.py` (delegate to existing `get_num_cores()`)"
    if "MLA attention" in sub:
        return "QAIC backend has no MLA attention support"
    if "mrope interleaved" in sub:
        return "QAIC has no interleaved mRoPE (`is_neox_style=False`) support"
    if "bfloat16" in sub:
        return "device rejects bf16 — force fp16 dtype"
    if "non-SDPA" in sub:
        return "model rejects `TORCH_SDPA`; QAIC offers no alternative backend"
    if "attention backend unsupported" in sub:
        return "`QAicTorchAttentionBackend` unsupported for this model"
    if r["category"].startswith("H."):
        return "install Hexagon SDK lib: `libqhmath.a` missing from `/opt/qti-qic/dev/hexagon_sdk/libs/qhl/prebuilt/hexagon_toolv87_v68`"
    if r["category"].startswith("B."):
        st, fr = BSTAGE[m]
        return "QAIC device error 500 during **{}** at `{}`".format(st, fr)
    if "gated repo" in sub:
        return "gated repo — set `HF_TOKEN` and accept the license"
    if "404" in sub:
        return "repo not public / does not exist — fix name or drop from model list"
    if "file missing from repo" in sub:
        return CNOTE[m]
    if "transformers too old" in sub:
        return "transformers 5.5.4 does not know `model_type={}` — needs newer transformers".format(
            MTYPE[m]
        )
    if "divisible by tp" in sub:
        return "issue with tp_size, n_heads=%d (use tp=2 or tp=1)" % HEADS[m]
    if "max_model_len" in sub:
        return "harness sets max_model_len=4096, model max is 448 — lower it for this model"
    if "limit_per_prompt" in sub:
        return (
            "harness passes `limit_per_prompt={'fps': 2}`; vLLM 0.23 rejects `fps` key"
        )
    if "no architectures" in sub:
        return "config.json has no `architectures` field — pass `--hf-overrides`"
    if "no TP support" in sub:
        return "no tensor-parallel support in vLLM — run tp=1"
    if "timm too old" in sub:
        return "timm 1.0.14 too old for `mobilenetv5_300m_enc` — `pip install -U timm`"
    if "missing python package" in sub:
        return "missing package {}".format(PKG[m])
    if r["category"].startswith("F."):
        return FNOTE[m]
    if r["category"].startswith("G."):
        return GNOTE[m]
    return sub


ORDER = [
    "A. vllm-qaic platform gap",
    "H. Hexagon SDK install broken",
    "E. Environment / deps",
    "D. Config / harness / TP",
    "C. Model access / download",
    "B. QAIC device runtime error 500",
    "F. HF processor / config mismatch",
    "G. Shape / tensor bug",
    "0. SUCCESS",
]
OWNER = {
    "A. vllm-qaic platform gap": "vllm-qaic code",
    "H. Hexagon SDK install broken": "environment",
    "E. Environment / deps": "environment",
    "D. Config / harness / TP": "CI config",
    "C. Model access / download": "CI config / credentials",
    "B. QAIC device runtime error 500": "QAIC backend / hardware",
    "F. HF processor / config mismatch": "upstream HF / model repo",
    "G. Shape / tensor bug": "vllm-qaic or model code",
    "0. SUCCESS": "—",
}
BLURB = {
    "A. vllm-qaic platform gap": "Missing capability in the `vllm_qaic` plugin. 45 of these 53 are the *same* one-method gap.",
    "H. Hexagon SDK install broken": "Missing native library. The worker retries for ~8 min, then is SIGKILLed and surfaces only as `RuntimeError: cancelled`.",
    "E. Environment / deps": "Fix with `pip install`. No code changes needed.",
    "D. Config / harness / TP": "Fixable in the CI model list / harness args.",
    "C. Model access / download": "Nothing was ever downloaded; fix credentials or prune the model list.",
    "B. QAIC device runtime error 500": "Real backend/hardware failures — the only group needing device-level debugging. Spread across the whole 19-hour run, so not one wedged device.",
    "F. HF processor / config mismatch": "Model remote code or repo metadata vs. installed transformers 5.5.4. Mostly not yours to fix.",
    "G. Shape / tensor bug": "Genuine tensor-shape mismatches needing investigation.",
    "0. SUCCESS": "Ran to completion and generated output.",
}

by = collections.defaultdict(list)
for r in rows:
    by[r["category"]].append(r)

L = []
L.append("# vLLM-QAIC VLM Fallback CI — Failure Report\n")
L.append(
    "Run date: 2026-08-25 18:39 → 2026-08-26 14:03  |  vLLM 0.23.0  |  "
    "`vllm_qaic` 0.23.0.dev25+gf9ec83794  |  transformers 5.5.4\n"
)
L.append(
    "**116 model runs — 1 completed, 115 exited with status code 1.** "
    "28 runs logged some CPU fallback ops before dying; 88 produced an empty result, "
    "so the sweep collected almost no usable fallback-op data.\n"
)
L.append("## Summary\n")
L.append("| Category | Runs | Fix owner |")
L.append("|---|---:|---|")
for c in ORDER:
    if c in by:
        L.append("| %s | %d | %s |" % (c[3:], len(by[c]), OWNER[c]))
L.append("| **Total** | **%d** | |\n" % len(rows))

L.append("## Fix order\n")
L.append(
    "1. **Implement `num_compute_units()`** in `vllm_qaic/platform_base.py` → unblocks 45 runs (39%)."
)
L.append("2. **Install `libqhmath.a`** → 4 runs, plus ~30 min of wall-clock hang.")
L.append("3. **`pip install`** the 6 missing packages + upgrade `timm` → 7 runs.")
L.append(
    "4. **Set `HF_TOKEN`; prune the 7 dead repos** → 14 runs stop consuming slots."
)
L.append("5. **Fix tp_size for 3 models + 2 harness arg bugs** → 5 runs.")
L.append(
    "6. Remaining: 13 QAIC device errors, 9 upstream HF issues, 3 shape bugs, 8 other platform gaps.\n"
)
L.append(
    "> Note: the `num_compute_units` crash happens in memory profiling, *before* real inference. "
    "Clearing it does not guarantee those 45 models pass — expect a second wave of per-model failures.\n"
)
L.append("---\n")

for c in ORDER:
    if c not in by:
        continue
    L.append("## %s — %d run%s\n" % (c, len(by[c]), "" if len(by[c]) == 1 else "s"))
    L.append("{}\n".format(BLURB[c]))
    grp = [r for r in by[c] if "num_compute_units" in r["reason"]]
    rest = [r for r in by[c] if r not in grp]
    if grp:
        L.append("### A1. `num_compute_units()` not implemented — %d runs\n" % len(grp))
        L.append(
            "All %d fail identically in memory profiling. **One fix clears all of them:** "
            "add a `num_compute_units()` classmethod to `QaicPlatform` in "
            "`vllm_qaic/platform_base.py`, delegating to the existing `get_num_cores()` "
            "(line 124). Guard the `is_aot` branch, which currently returns `None`.\n"
            % len(grp)
        )
        for r in sorted(grp, key=lambda r: r["hf_name"].lower()):
            L.append("- `{}` (tp={})".format(r["hf_name"], r["tp"]))
        L.append("")
        L.append("### A2. Other platform gaps — %d runs\n" % len(rest))
    L.append("| Model | Fix |")
    L.append("|---|---|")
    for r in sorted(rest if grp else by[c], key=lambda r: r["hf_name"].lower()):
        L.append("| `{}` (tp={}) | {} |".format(r["hf_name"], r["tp"], fix(r)))
    L.append("")

L.append("---\n")
L.append("## Environment fix-list\n")
L.append("```bash")
L.append("pip install tensorflow decord open_clip_torch matplotlib kaldi_native_fbank")
L.append(
    "pip install -U timm            # 1.0.14 lacks mobilenetv5_300m_enc (gemma-3n)"
)
L.append(
    "pip install -U transformers    # 5.5.4 lacks exaone4_5, gemma4_unified, granite_speech_plus"
)
L.append("export HF_TOKEN=<token>        # 7 gated repos")
L.append("# Hexagon SDK: ensure libqhmath.a exists in")
L.append("#   /opt/qti-qic/dev/hexagon_sdk/libs/qhl/prebuilt/hexagon_toolv87_v68")
L.append("```\n")
L.append("## Models to drop or rename (do not exist publicly)\n")
for r in sorted(by["C. Model access / download"], key=lambda r: r["model"].lower()):
    if "404" in r["reason"]:
        L.append("- `{}`".format(r["hf_name"]))
L.append("")
L.append("---\n")
L.append(
    "Generated from 116 `log_1_*.log` / `parse_1_*.log` pairs in `ci_scripts/ci_logs/vlm/`. "
    "Machine-readable version: `failure_categories.csv`."
)

out = os.path.join(D, "FAILURE_REPORT.md")
open(out, "w").write("\n".join(L) + "\n")
print("wrote %s (%d lines)" % (out, len(L)))
