# v8 Step 4 — running the 4 heavy families

Only needed if you want Qwen2.5-0.5B, SmolLM2-1.7B, Qwen2.5-3B, Phi-3.5-mini
in the v8 forced-choice table. 3 light families (surface_lr, BGE-large,
ModernBERT-base) are already scored — see `v8_forced_choice_results.json`.

## Prerequisites

Any Python env with:
- torch, transformers (HF Hub offline OK — caches are in `artifacts/hf_cache/hub/`)
- scikit-learn, joblib, numpy

The existing CHPC env used for commits 6c609ae / 3bc4044 / b7d8d33 is sufficient.

## Commands

```bash
cd /Users/foadnamjoo/PROJECT/LLM_Dataset_creation/truthfulqa_audit

# one at a time (3B models take ~10-20 min each on MPS / ~5 min on A100):
python scripts/run_v8_heavy_family.py qwen
python scripts/run_v8_heavy_family.py smollm2
python scripts/run_v8_heavy_family.py qwen3b
python scripts/run_v8_heavy_family.py phi35

# or all four in sequence:
python scripts/run_v8_heavy_family.py all
```

## What it does

For each family, rewires the v3 scorer's A-side input from
`stage0_singleton_v7a_v3_generations.json` (length-matched 22-26 words)
to `stage0_singleton_v7a_generations.json` (FALSE-class cues, 5-15
words) — that is the v8 A-side. B-side (§5.1) is unchanged.

Outputs:
- `stage0_v8_inverted_bilateral/v8_<family>_scores.json`  (v8 forced-choice schema)
- `stage0_v8_inverted_bilateral/v8_<family>_raw.json`     (full v3-style unilateral + bilateral payload, for reference)

## After running

```bash
python scripts/score_v8_forced_choice.py
```

re-runs the aggregator; 7-family table will populate automatically.
