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

D = "/local/mnt5/workspace/agokhale/fallback-ci/vllm-qaic/ci_scripts/ci_logs/vlm"
EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|NotFound))(?:\(.*?\))?: (.*)$")
WRAP = (
    "Engine core initialization failed",
    "Worker failed with error",
    "Engine core proc",
    "cancelled",
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
    excs = [(m.group(1), m.group(2)) for m in (EXC.match(clean(l)) for l in lines) if m]
    root = next(
        ((t, m) for t, m in excs if not any(w in m for w in WRAP)),
        excs[0] if excs else ("None", ""),
    )
    rc = "{}: {}".format(*root)
    pf = os.path.join(D, "parse_1_{}.log".format(model))
    ops = os.path.exists(pf) and ", {})" not in open(pf, errors="replace").read()
    ok = "python exited with status code" not in txt

    if ok:
        cat, sub = ("0. SUCCESS", "ran to completion, generated output")
    elif "libqhmath.a was not found" in txt:
        cat, sub = (
            "H. Hexagon SDK install broken",
            "missing libqhmath.a -> worker hangs ~8min then killed",
        )
    elif "num_compute_units is not implemented" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "num_compute_units() not implemented (Triton top-k/top-p sampler)",
        )
    elif "QAic doesn't support MLA" in rc:
        cat, sub = ("A. vllm-qaic platform gap", "MLA attention unsupported")
    elif "mrotary_embedding_eager_fallback" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "mrope interleaved (is_neox_style=False) unsupported",
        )
    elif "QAicTorchAttentionBackend" in rc:
        cat, sub = ("A. vllm-qaic platform gap", "attention backend unsupported")
    elif "BFloat16 datatype" in rc:
        cat, sub = ("A. vllm-qaic platform gap", "bfloat16 unsupported on device")
    elif "TORCH_SDPA backend now" in rc:
        cat, sub = (
            "A. vllm-qaic platform gap",
            "model requires non-SDPA attention backend",
        )
    elif re.search(r"Error code: 500", rc):
        cl = [clean(l) for l in lines]
        i = next(
            (
                j
                for j, l in enumerate(cl)
                if re.match(r"RuntimeError: Device id: \d+ QID", l)
            ),
            0,
        )
        pre = "\n".join(cl[max(0, i - 70) : i])
        st = (
            "during model load / weight+scale init"
            if ("in set_default_quant_scales" in pre or "in load_model" in pre)
            else "during memory profiling (profile_run)"
        )
        cat, sub = ("B. QAIC device runtime error 500", st)
    elif "403 Forbidden" in rc or "gated repo" in rc:
        cat, sub = ("C. Model access / download", "gated repo (403) - needs HF token")
    elif "404 Not Found" in rc or "is not a local folder" in rc:
        cat, sub = (
            "C. Model access / download",
            "repo not found (404) - model does not exist publicly",
        )
    elif (
        "does not appear to have a file named" in rc
        or "Cannot find any model weights" in rc
    ):
        cat, sub = ("C. Model access / download", "required file missing from repo")
    elif "Transformers does not recognize this architecture" in txt:
        cat, sub = (
            "D. Config / harness / TP",
            "transformers too old for this architecture",
        )
    elif (
        "must be divisible by tensor parallel size" in txt
        or "is not divisible by" in rc
    ):
        cat, sub = (
            "D. Config / harness / TP",
            "attention heads not divisible by tp -> wrong tp in model config",
        )
    elif "is greater than the derived max_model_len" in txt:
        cat, sub = (
            "D. Config / harness / TP",
            "CI harness max_model_len=4096 exceeds model max",
        )
    elif "Unexpected keyword argument" in txt:
        cat, sub = (
            "D. Config / harness / TP",
            "CI harness passes limit_per_prompt fps= unsupported by vLLM 0.23",
        )
    elif "No model architectures are specified" in txt:
        cat, sub = (
            "D. Config / harness / TP",
            "config.json has no architectures field",
        )
    elif "does not support tensor parallel" in rc:
        cat, sub = ("D. Config / harness / TP", "model has no TP support in vLLM")
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
    elif re.search(r"is invalid for input of size|view size is not compatible", rc):
        cat, sub = (
            "G. Shape / tensor bug",
            "tensor shape mismatch in model or QAic kernel",
        )
    else:
        cat, sub = ("F. HF processor / config mismatch", rc[:75])
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
    print("%-36s %3d" % (c, n))
    for (cat, s), k in sorted(sc.items(), key=lambda kv: -kv[1]):
        if cat == c:
            print("      %3d  %s" % (k, s))
print("\nCSV -> {}".format(out))
