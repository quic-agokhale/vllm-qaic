# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# ruff: noqa
import regex as re
import os
import glob
import collections
import csv

D = "/local/mnt5/workspace/agokhale/fallback-ci/vllm-qaic/ci_scripts/ci_logs/llm"
EXC = re.compile(
    r"^((?:[A-Za-z_][\w.]*)?(?:Error|Exception|NotFound|Interrupt))(?:\(.*?\))?: (.*)$"
)
WRAP = (
    "Engine core initialization failed",
    "Worker failed with error",
    "Engine core proc",
    "cancelled",
    "WorkerProc initialization failed",
)


def clean(l):
    m = re.match(
        r"^\([^)]*\)\s*(?:ERROR|WARNING|INFO|DEBUG)?\s*\d\d-\d\d \d\d:\d\d:\d\d \[[^\]]+\]\s?",
        l,
    )
    return (l[m.end() :] if m else re.sub(r"^\([^)]*\)\s*", "", l)).strip()


rows = []
for f in sorted(glob.glob(os.path.join(D, "log_*.log"))):
    model = os.path.basename(f)[6:-4]
    txt = open(f, errors="replace").read()
    lines = txt.splitlines()
    cl = [clean(l) for l in lines]
    excs = [(m.group(1), m.group(2)) for m in (EXC.match(l) for l in cl) if m]
    root = next(
        ((t, m) for t, m in excs if not any(w in m for w in WRAP)),
        excs[0] if excs else ("None", ""),
    )
    rc = "{}: {}".format(*root)
    # pydantic wraps the real message on the NEXT line
    if "ValidationError" in root[0]:
        i = next((j for j, l in enumerate(cl) if "validation error for" in l), None)
        if i is not None:
            rc = (
                "ValidationError: "
                + re.sub(r"^Value error, ", "", cl[i + 1]).split(" [type=")[0]
            )
    pf = os.path.join(D, "parse_1_{}.log".format(model))
    ops = os.path.exists(pf) and ", {})" not in open(pf, errors="replace").read()
    ok = "python exited with status code" not in txt

    if ok:
        cat, sub = ("0. SUCCESS", "ran to completion, generated output")
    elif re.search(r"malloc\(\): mismatching|corrupted size vs\. prev_size", txt):
        cat, sub = (
            "B. QAIC device / runtime error",
            "glibc heap corruption in all workers during model load",
        )
    elif "QAic doesn't support MLA" in rc:
        cat, sub = ("A. vllm-qaic platform gap", "MLA attention unsupported")
    elif "BFloat16 datatype" in rc:
        cat, sub = ("A. vllm-qaic platform gap", "bfloat16 unsupported on device")
    elif "PlatformEnum.OOT" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "no scaled_mm linear kernel registered for OOT platform (compressed-tensors W8A16-FP8)",
        )
    elif "selective_scan_fwd" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "Mamba/SSM kernel _C::selective_scan_fwd not built",
        )
    elif "static_scaled_fp8_quant" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "FP8 kernel _C::static_scaled_fp8_quant not built",
        )
    elif "flash_attn_supports_sinks" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "FLASH_ATTN_DIFFKV backend selected; vllm_flash_attn absent (attention sinks)",
        )
    elif "use_alibi_sqrt is not supported" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "ALiBi-sqrt unsupported by QAIC CUSTOM attention backend",
        )
    elif "quantization is currently not supported in qaic" in rc:
        q = re.search(r"(\S+) quantization is currently not supported", rc)
        cat, sub = (
            "A. vllm-qaic platform gap",
            "quantization scheme unsupported: %s" % (q.group(1) if q else "?"),
        )
    elif "Hybrid KV cache manager is disabled" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "hybrid/sliding-window KV cache unsupported -> cannot unify KV specs",
        )
    elif re.search(r"Error code: 500", rc) or re.search(r"Error code: 9$", rc):
        i = next(
            (
                j
                for j, l in enumerate(cl)
                if re.match(r"RuntimeError: Device id: \d+ QID", l)
            ),
            0,
        )
        pre = "\n".join(cl[max(0, i - 90) : i])
        st = (
            "during model load / weight+scale init"
            if (
                "in set_default_quant_scales" in pre
                or "in load_model" in pre
                or "load_weights" in pre
            )
            else "during memory profiling (profile_run)"
        )
        code = "error 9 (send data to device)" if "Error code: 9" in rc else "error 500"
        cat, sub = ("B. QAIC device / runtime error", "{}, {}".format(code, st))
    elif "403 Forbidden" in rc or "gated repo" in rc:
        cat, sub = ("C. Model access / download", "gated repo (403) - needs HF token")
    elif (
        "404 Not Found" in rc
        or "is not a local folder" in rc
        or "Invalid repository ID" in rc
    ):
        cat, sub = (
            "C. Model access / download",
            "repo not found (404) - model does not exist publicly",
        )
    elif "Transformers does not recognize this architecture" in rc:
        cat, sub = (
            "D. Config / harness / TP",
            "transformers too old for this model_type",
        )
    elif "must be divisible by tensor parallel size" in rc:
        cat, sub = (
            "D. Config / harness / TP",
            "attention heads not divisible by tp -> wrong tp in model config",
        )
    elif "only supported for generative models" in rc:
        cat, sub = (
            "D. Config / harness / TP",
            "embedding/pooling model - harness must pass --runner generate or drop it",
        )
    elif "No available memory for the cache blocks" in rc:
        cat, sub = (
            "D. Config / harness / TP",
            "no KV-cache memory left - raise gpu_memory_utilization or use more devices",
        )
    elif "not supported in your version of timm" in rc or "Unknown model (" in rc:
        cat, sub = ("E. Environment / deps", "timm too old (mobilenetv5_300m_enc)")
    elif "were not found in your environment" in rc or "No module named" in rc:
        pkg = re.search(r"environment: ([\w, ]+)", rc) or re.search(
            r"No module named '([\w]+)'", rc
        )
        cat, sub = (
            "E. Environment / deps",
            "missing python package: %s" % (pkg.group(1) if pkg else "?"),
        )
    elif (
        "piece must not include null" in rc
        or "prepend_scheme" in rc
        or "add_prefix_space" in rc
    ):
        cat, sub = (
            "F. Tokenizer / config mismatch",
            "tokenizer will not load with installed tokenizers/transformers",
        )
    elif re.search(r"is invalid for input of size|view size is not compatible", rc):
        cat, sub = ("G. Shape / tensor bug", "tensor shape mismatch")
    elif "no module or parameter named" in rc:
        cat, sub = (
            "G. Weight-loading bug",
            "checkpoint param names do not match vLLM model class",
        )
    else:
        cat, sub = ("F. Tokenizer / config mismatch", rc[:80])
    rows.append(
        dict(
            model=model,
            category=cat,
            reason=sub,
            root_error=rc[:220],
            fallback_ops="yes" if ops else "no",
            completed="yes" if ok else "no",
        )
    )

out = os.path.join(D, "failure_categories.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(
        fh,
        fieldnames=[
            "category",
            "reason",
            "model",
            "fallback_ops",
            "completed",
            "root_error",
        ],
    )
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["category"], r["reason"], r["model"])):
        w.writerow(r)

print(
    "Runs: %d | completed: %d | produced fallback-op data: %d\n"
    % (
        len(rows),
        sum(r["completed"] == "yes" for r in rows),
        sum(r["fallback_ops"] == "yes" for r in rows),
    )
)
cc = collections.Counter(r["category"] for r in rows)
sc = collections.Counter((r["category"], r["reason"]) for r in rows)
for c, n in sorted(cc.items()):
    print("%-40s %3d" % (c, n))
    for (cat, s), k in sorted(sc.items(), key=lambda kv: -kv[1]):
        if cat == c and cat != "0. SUCCESS":
            print("      %3d  %s" % (k, s))
print("\nCSV -> {}".format(out))
