# TinyTruthfulQA (TinyBenchmarks) surface audit

Script: `scripts/run_tinybenchmarks_truthfulqa_surface_audit.py`

Loads `tinyBenchmarks/tinyTruthfulQA` (`validation`, 100 rows), maps each HF row to a local `TruthfulQA.csv` row via `mc1_targets` (normalized question + correct answer + incorrect matching one of the distractors), then runs the same grouped CV logistic audit as `audit_subset_evaluator.evaluate_subset_grouped_cv`.

## Dependency

```bash
pip install datasets
```

(Also listed in `requirements-paper-full.txt`.)

## Run

```bash
python3 scripts/run_tinybenchmarks_truthfulqa_surface_audit.py --root .
```

## Outputs (`results/tinybenchmarks_audit/`)

- `tinybenchmarks_surface_audit.csv` — full reference, tau=0.53 audited subset, IRT anchors, 10× random 100-pair controls, aggregate mean±std
- `anchor_match_report.csv` — per-HF-row match status
- `anchor_pair_ids.json` — matched local `pair_id` list
- `config.json` — seeds, counts, random baseline statistics

## Matching caveat

Some HF anchor questions may not appear in the pinned local `TruthfulQA.csv`; those rows are reported as `unmatched` in `anchor_match_report.csv`. The audit runs on **matched unique** `pair_id`s only.
