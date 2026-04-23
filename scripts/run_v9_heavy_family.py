#!/usr/bin/env python3
"""
v9 same-question heavy-family runner.

Scores each of the 4 heavy classifier families (Qwen2.5-0.5B, SmolLM2-1.7B,
Qwen2.5-3B, Phi-3.5-mini) on the v4 paired same-question TRUE/FALSE pairs
(stage0_paired_tqa/stage0_paired_generations_v4.json). Reuses the existing
v7a-v3 encoder functions and pickles.

Output per family:
    stage0_v9_same_question/v9_<fam>_scores.json

Schema matches the v8 heavy-family schema consumed by the forced-choice
aggregator:
    [{pair_id, P_A_truthful_full, P_A_truthful_cleaned,
               P_B_truthful_full, P_B_truthful_cleaned}, ...]

Usage:
    python scripts/run_v9_heavy_family.py qwen
    python scripts/run_v9_heavy_family.py smollm2
    python scripts/run_v9_heavy_family.py qwen3b
    python scripts/run_v9_heavy_family.py phi35
    python scripts/run_v9_heavy_family.py all
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

V4_GEN_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations_v4.json"
V9_DIR = REPO_ROOT / "stage0_v9_same_question"

FAMILIES = {
    "qwen":    {
        "module":      "scripts.score_singleton_v7a_v3_qwen",
        "encoder":     "_encode_qwen",
        "pkl_full":    "QWEN_PKL_FULL",
        "pkl_clean":   "QWEN_PKL_CLEAN",
        "v9_out_file": V9_DIR / "v9_qwen_scores.json",
    },
    "qwen15b": {
        "module":      "scripts.score_singleton_v7a_v3_qwen15b",
        "encoder":     "_encode_qwen15b",
        "pkl_full":    "PKL_FULL",
        "pkl_clean":   "PKL_CLEAN",
        "v9_out_file": V9_DIR / "v9_qwen15b_scores.json",
    },
    "smollm2": {
        "module":      "scripts.score_singleton_v7a_v3_smollm2",
        "encoder":     "_encode_smollm2",
        "pkl_full":    "SMOLLM2_PKL_FULL",
        "pkl_clean":   "SMOLLM2_PKL_CLEAN",
        "v9_out_file": V9_DIR / "v9_smollm2_scores.json",
    },
    "qwen3b":  {
        "module":      "scripts.score_singleton_v7a_v3_qwen3b",
        "encoder":     "_encode_qwen3b",
        "pkl_full":    "PKL_FULL",
        "pkl_clean":   "PKL_CLEAN",
        "v9_out_file": V9_DIR / "v9_qwen3b_scores.json",
    },
    "phi35":   {
        "module":      "scripts.score_singleton_v7a_v3_phi35",
        "encoder":     "_encode_phi35",
        "pkl_full":    "PHI_PKL_FULL",
        "pkl_clean":   "PHI_PKL_CLEAN",
        "v9_out_file": V9_DIR / "v9_phi35_scores.json",
    },
    "llama32_3b": {
        "module":      "scripts.score_singleton_v7a_v3_llama32_3b",
        "encoder":     "_encode_llama32_3b",
        "pkl_full":    "PKL_FULL",
        "pkl_clean":   "PKL_CLEAN",
        "v9_out_file": V9_DIR / "v9_llama32_3b_scores.json",
    },
}


def _load_v4_pairs(gen_path: Path = V4_GEN_JSON):
    with open(gen_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = sorted(data, key=lambda r: int(r["pair_id"]))
    pair_ids = [int(r["pair_id"]) for r in data]
    a_texts = [str(r["a_side"]) for r in data]
    b_texts = [str(r["b_side"]) for r in data]
    return pair_ids, a_texts, b_texts


def _resolve_encoder(mod, encoder_name: str):
    """Resolve encoder attr; fall back to the first _encode_* function."""
    if hasattr(mod, encoder_name):
        return getattr(mod, encoder_name)
    for name in dir(mod):
        if name.startswith("_encode_") and callable(getattr(mod, name)):
            return getattr(mod, name)
    raise RuntimeError(f"{mod.__name__}: no encoder function found")


def _resolve_pickle(mod, candidates: list[str]):
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(
        f"{mod.__name__}: no pickle attr among {candidates}")


def run_family(short: str, gen_path: Path = V4_GEN_JSON,
               out_dir: Path = V9_DIR, out_suffix: str = "") -> int:
    import joblib
    import numpy as np

    if short not in FAMILIES:
        print(f"Unknown family: {short}. Valid: {list(FAMILIES)}",
              file=sys.stderr)
        return 2
    cfg = FAMILIES[short]
    print("=" * 72)
    print(f"v9 heavy-family run: {short}")
    print(f"  gen_path: {gen_path}")
    print(f"  out_dir:  {out_dir}")
    print("=" * 72)

    mod = importlib.import_module(cfg["module"])
    encode = _resolve_encoder(mod, cfg["encoder"])
    pkl_full = _resolve_pickle(mod, [cfg["pkl_full"]])
    pkl_clean = _resolve_pickle(mod, [cfg["pkl_clean"]])
    print(f"  encoder: {encode.__name__}")
    print(f"  pkl_full:  {pkl_full}")
    print(f"  pkl_clean: {pkl_clean}")

    pair_ids, a_texts, b_texts = _load_v4_pairs(gen_path)
    n = len(pair_ids)
    print(f"  v4 pairs: {n}  (pair_ids={pair_ids})")

    print(f"\nEncoding {n} A-side texts ...")
    Xa = encode(a_texts)
    print(f"  Xa shape = {Xa.shape}")
    print(f"\nEncoding {n} B-side texts ...")
    Xb = encode(b_texts)
    print(f"  Xb shape = {Xb.shape}")

    print("\nLoading pickles ...")
    art_full = joblib.load(pkl_full)
    art_clean = joblib.load(pkl_clean)
    print(f"  full    CV AUC: {art_full.get('cv_auc_group5'):.4f}")
    print(f"  cleaned CV AUC: {art_clean.get('cv_auc_group5'):.4f}")

    proba = mod._proba_truthful
    PA_full  = proba(art_full["pipeline"],  Xa)
    PA_clean = proba(art_clean["pipeline"], Xa)
    PB_full  = proba(art_full["pipeline"],  Xb)
    PB_clean = proba(art_clean["pipeline"], Xb)

    rows = []
    for i, pid in enumerate(pair_ids):
        rows.append({
            "pair_id": int(pid),
            "P_A_truthful_full":    float(PA_full[i]),
            "P_A_truthful_cleaned": float(PA_clean[i]),
            "P_B_truthful_full":    float(PB_full[i]),
            "P_B_truthful_cleaned": float(PB_clean[i]),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    if out_suffix:
        out = out_dir / f"v9{out_suffix}_{short}_scores.json"
    else:
        out = cfg["v9_out_file"]
        if out.parent != out_dir:
            out = out_dir / out.name
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote v9 scores: {out.relative_to(REPO_ROOT)}  ({len(rows)} pairs)")

    # Quick summary
    correct_full  = sum(1 for r in rows if r["P_A_truthful_full"]    > r["P_B_truthful_full"])
    correct_clean = sum(1 for r in rows if r["P_A_truthful_cleaned"] > r["P_B_truthful_cleaned"])
    print(f"  pair_accuracy  full   = {correct_full}/{n} = {correct_full/n:.3f}")
    print(f"  pair_accuracy  cleaned= {correct_clean}/{n} = {correct_clean/n:.3f}")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES) + ["all"])
    ap.add_argument("--gen-json", type=str, default=str(V4_GEN_JSON),
                    help="Paired same-question generations JSON "
                         "(each row: pair_id, a_side, b_side).")
    ap.add_argument("--out-dir", type=str, default=str(V9_DIR))
    ap.add_argument("--out-suffix", type=str, default="",
                    help='e.g. "_scaled" -> v9_scaled_<fam>_scores.json')
    args = ap.parse_args()
    gen = Path(args.gen_json).resolve()
    odir = Path(args.out_dir).resolve()

    if args.family == "all":
        for fam in FAMILIES:
            rc = run_family(fam, gen, odir, args.out_suffix)
            if rc != 0:
                return rc
        return 0
    return run_family(args.family, gen, odir, args.out_suffix)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
