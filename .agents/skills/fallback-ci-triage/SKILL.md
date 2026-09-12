---
name: fallback-ci-triage
description: Triage a vllm-qaic Fallback CI sweep. Parses log and parse pairs in ci_scripts/ci_logs, finds each run's true root cause, buckets it by fix owner, and writes FAILURE_REPORT.md plus failure_categories.csv. Use when asked why fallback CI model runs are failing, to categorize or parse errors in ci_logs, to build or refresh a fallback CI failure report, to diff two sweeps, to find which models need a given fix, or to check how many runs produced CPU fallback-op data.
---

# Fallback CI triage

A sweep runs ~120–150 HuggingFace models through vLLM on QAIC to collect **CPU fallback ops**.
Most runs die during startup instead. This skill turns a log directory into a ranked, actionable
failure report.

## Run it

```bash
python ~/.agents/skills/fallback-ci-triage/triage.py <dir> --stdout   # summary only, writes nothing
python ~/.agents/skills/fallback-ci-triage/triage.py ci_scripts/ci_logs/llm
python ~/.agents/skills/fallback-ci-triage/triage.py ci_scripts/ci_logs/vlm --title VLM
```

Writes `FAILURE_REPORT.md` and `failure_categories.csv` into the log directory. Reproduces the
Aug-2026 sweeps exactly: LLM 148 runs / 77 pass, VLM 116 runs / 1 pass.

**Always read the `--stdout` summary first.** If it ends with `!! N unclassified`, those runs have
no rule yet — diagnose them by hand and add rules (see below) before trusting the report.

**Never regenerate over a delivered report.** A finished report carries hand-written sections the
script cannot reproduce — fix ordering, environment notes, cross-sweep caveats, the HF 404-vs-403
wording. Writing therefore aborts unless `--force` is given; do not reach for `--force` on a
directory whose report someone has already edited, and prefer editing that report in place. The
Aug-2026 reports were destroyed exactly this way; the scripts that rebuilt them are archived in
`aug2026-generators/`, and its README lists the later hand-edits they do *not* contain.

## Input layout

One pair per run, in `ci_scripts/ci_logs/<suite>/`:

- `log_1_<model_with_underscores>_tp<N>.log` — full stdout/stderr; line 1 is the invocation
- `parse_1_<model>_tp<N>.log` — `fallback_parser.py` output; a `defaultdict(..., {})` means no ops

Sweeps are driven by `ci_scripts/ci_fallback_ops.sh` (VLM) and `ci_fallback_ops_llm.sh` (LLM).

## Pitfalls that have already cost time

These are all handled in `triage.py`. Re-read them before hand-checking anything or writing an
ad-hoc grep, because each one produced a wrong answer first.

1. **Never classify on a whole-log grep.** Strings like `Hybrid KV cache manager is disabled` and
   `max_position_embeddings` appear as *warnings* in runs that passed. Classify on the root error
   only. A whole-log grep for "hybrid KV" matched 8 files when exactly 1 failed for that reason.

2. **The root cause is not the last exception.** vLLM wraps worker failures in
   `Engine core initialization failed`, `Worker failed with error`, `WorkerProc initialization
   failed`, and `cancelled`. Skip those and take the first real exception.

3. **pydantic hides the message on the next line.** `ValidationError: 1 validation error for
   VllmConfig` is useless; the actual complaint (`Total number of attention heads (30) must be
   divisible by...`) is the following line.

4. **A bare `Exception:` has no module prefix.** A regex requiring `[\w.]+` before
   `Error|Exception` silently drops it.

5. **tp comes from the filename, not the log.** The harness always logs the sweep default
   (`--tp-size 4`); `run_{llm,vlm}s.py` then re-resolves the pinned value via
   `model_configs_*.get_tp_size()`. Verified against runtime `tensor_parallel_size=N`.

6. **Detect empty fallback-op output with the literal `, {})`.** Don't regex the
   `<lambda at 0x...>` address — that spelling varies.

7. **HF 404 ≠ "repo does not exist."** HF returns 404 rather than 403 for a repo the caller isn't
   authorised to see, so a 404 covers wrong names *and* inaccessible private/gated repos. A 403
   saying *"you are not in the authorized list"* means a valid token **is** present and only needs
   access approval — don't tell the user to set `HF_TOKEN`. Verify well-known names before pruning.

8. **`--runner auto` resolving to `pooling`** is usually vLLM incorrectly detecting a generative model, not
   proof the model is embedding-only. Fix is `--runner generate`, not deletion.

9. **For device errors, report the frame, not the message.** Every one says
   `Failed to synchronize stream. Error code: 500`. The useful signal is the deepest
   `vllm_qaic`/`models`/`layers` frame before it, plus whether it sits in model load or
   `profile_run`. Shared frames mean one root cause across many models — 14 LLM runs died at
   `attention.py:97 set_default_quant_scales()`.

10. **Before blaming device contention, check `QAIC_VISIBLE_DEVICES`** in both sweep scripts. The
    Aug-2026 sweeps overlapped in wall-clock time but used disjoint sets (LLM `32..39`, VLM
    `40..47`), so contention was ruled out.

11. **The Hexagon SDK failure masquerades as a cancellation.** A missing `libqhmath.a` makes the
    worker retry for ~8 min, get SIGKILLed, and surface only as `RuntimeError: cancelled`. Check
    for the library name before classifying a cancellation.

12. **The shell here is tcsh.** No `$((...))` arithmetic — use `awk -v n=$n`. Working directory
    resets between tool calls, so use absolute paths or re-`cd`.

13. **A caught exception logged at WARNING can precede the fatal one.** `mamba_mixer2.py`
    wraps its SSD warm-up in `try/except` and dumps a full traceback via `logger.warning`, then
    dies several frames later at `empty_cache()`. Grabbing the first exception in file order gives
    the wrong root cause, so `root_error()` searches non-WARNING lines first and only falls back to
    WARNING lines if nothing else has an exception.

14. **`'function' object is not subscriptable` means Triton is disabled, not a model bug.** vLLM
    swaps in `TritonPlaceholder` whenever it finds ≠1 active Triton driver; the placeholder's `jit`
    is an identity decorator, so `kernel[grid](...)` fails that way and `triton.next_power_of_2`
    goes missing. On a QAIC host the cause is the stock PyPI `triton` wheel overwriting Qualcomm's
    build — check `python -c "import triton.backends.qcom_hexagon_backend"` and compare how many
    logs contain `Triton is installed but 0 active driver(s)` between sweeps. Only Mamba/SSM models
    reach a Triton kernel, so this hides in plain sight.

15. **A changed failure reason is often a regression, not progress.** When a model's error moves,
    check whether the *old* error is actually gone. Two Mamba runs stopped reporting
    `_C::selective_scan_fwd` only because Triton now fails earlier — `hasattr(torch.ops._C,
    "selective_scan_fwd")` was still `False`. Say so, or the report reads as a fix.

16. **`--delete-hf-checkpoint` can poison the next sweep.** Compare the `[cleanup] deleted
    checkpoint ... (freed N)` sizes across sweeps: 32.5G on the first run and 7.7M on the re-run
    means the weights were never re-downloaded and the model failed with `Cannot find any model
    weights` seconds after launch.

## Every report must stand alone

A report describes **one** sweep. Someone reading it will not have the other sweep's logs, its
report, or the conversation in which the comparison was made, so nothing in the report may depend on
them. No "unchanged since 08-25", no "77 → 76 completed", no "restores parity with the previous
sweep", no "What changed since…" section, and no `(re-run)` in the title — `re-run` alone implies a
run the reader cannot see.

Findings discovered *by* comparing sweeps are still valid; restate them from this sweep's own
evidence. Practically:

- environment regression → cite the `.dist-info` timestamp against this sweep's start time, and the
  count of *this* sweep's logs containing the symptom
- a stale-cache failure → cite this run's `freed N` size against the model's actual weight size, not
  the previous run's `freed` size
- "fixing X only restores the old failure" → say which layer the fix exposes next, and back it with a
  direct check (`hasattr(torch.ops._C, ...)`) rather than the other sweep's error

Cross-sweep deltas belong in the chat reply, not in the file. `triage.py` never emits such text; it
creeps in during hand-augmentation, so grep the finished report for `08-\d\d|previous|re-run|parity|
unchanged since|since the` before delivering.

## Comparing two sweeps

Most of the value in a re-run is the delta, and it is usually small — 6 of 148 runs changed between
the Aug-25 and Aug-26 LLM sweeps. Load both and diff per model rather than eyeballing two reports:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / '.agents/skills/fallback-ci-triage'))
import triage
old = {r['model']: r for r in triage.load('ci_logs/llm')[0]}
new = {r['model']: r for r in triage.load('ci_logs/llm_0826')[0]}
```

Report pass→fail, fail→pass, changed reason, and whether fallback-op yield moved — **in your reply,
not in either report** (see above). Run python with `-B` so importing `triage` leaves no
`__pycache__` in the skill directory.

Also check what changed in the *environment*, not just the logs: `ls -la` the `.dist-info`
directories in site-packages and compare their timestamps against the sweep window. That is how the
Triton regression above was found.

## Lead with fallback-op yield, not the pass rate

The sweep exists to find CPU fallbacks, so report that first. A high pass rate can still mean zero
data collected: in the Aug-2026 LLM sweep 77 of 148 runs passed but **only 2 runs logged any
fallback op, and both of those crashed**. Plain decoder LLMs ran fully on device; fallbacks showed
up only in an FP8 checkpoint and a VLM. State that plainly instead of implying the sweep succeeded.

## Categories

| | Bucket | Owner |
|---|---|---|
| A | vllm-qaic platform gap (missing method, kernel, quant scheme, attention mode) | vllm-qaic code |
| B | QAIC device / runtime error (code 500/9, native crash) | QAIC backend / hardware |
| C | Model access / download (403, 404, missing file) | CI config / credentials |
| D | Config / harness / TP (head divisibility, model_type, runner, KV memory) | CI config |
| E | Environment / deps (missing or stale package) | environment |
| F | Processor / tokenizer / remote code vs installed transformers | upstream HF / model repo |
| G | Shape or weight-name mismatch | vllm-qaic or model code |
| H | Hexagon SDK install broken (`libqhmath.a`) | environment |

## Report style

Fix notes are **terse and directly actionable** — `missing package decord`, `issue with tp_size,
n_heads=30 (use tp=2 or tp=1)`. No prose, no root-cause essays; the CSV carries the raw error.

When ≥12 models share one fix, `triage.py` emits a bullet list under a single **Fix:** heading
instead of a dozen identical table rows (the VLM report had 45 such rows before this).

Rank by runs-unblocked-per-fix, and say so when a fix is not a guarantee: a crash in memory
profiling or model load happens *before* real inference, so clearing it usually exposes a second
wave of per-model failures.

## Adding rules

Rules live in `classify()` in `triage.py` as an ordered if/elif ladder returning
`(category_letter, short_reason, terse_fix_note)`. Order matters — H is checked before A, and
`'_C' object has no attribute` (an A-class missing kernel) must precede the generic F-class
attribute rules. Anything unmatched lands in F as `unclassified` with the raw error, so new failure
modes are loud rather than mislabelled.

After editing, re-run both suites with `--stdout` and confirm the other one didn't shift.
