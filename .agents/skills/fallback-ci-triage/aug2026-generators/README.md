# Aug-2026 one-off generators (archived)

These are the throwaway scripts that produced the **delivered** Aug-2026 reports, before the
logic was generalised into `../triage.py`. They lived in `/tmp` and were nearly lost; they are
kept here only because they are the sole reproduction path for the hand-tuned prose in those
two reports.

| Script | Produces |
|---|---|
| `final.py` | `ci_logs/vlm/failure_categories.csv` |
| `genmd.py` | `ci_logs/vlm/FAILURE_REPORT.md` (reads the CSV) |
| `llmfinal.py` | `ci_logs/llm/failure_categories.csv` |
| `llmgenmd.py` | `ci_logs/llm/FAILURE_REPORT.md` (reads the CSV) |

Run the CSV script before its matching MD script.

## They are NOT a faithful copy of the delivered reports

Corrections applied by hand **after** these scripts last ran are absent here. Re-running
`genmd.py` therefore *regresses* the VLM report — it silently drops:

- the `### Reinstall required for the num_compute_units fix` section
- `kaldi_native_fbank` → `kaldi-native-fbank` (import name vs pip name)
- the HF 404-vs-403 rewording in section C, the environment fix-list, and the
  "verify before pruning" list — including the `omni-search` → `omni-research` org typo

If you must regenerate, re-apply those afterwards. Prefer editing the report in place.

`../triage.py` supersedes these for any **new** sweep, and refuses to overwrite an existing
report unless given `--force`.
