#!/usr/bin/env python3
"""
v7(a) bilateral singleton scorer.

Scores the 20 v7(a) A-side TRUE / INCORRECT-style-surface statements
with all six canonical Stage-0 pickles, then pairs them against the
existing §5.1 singleton B-side (stage0/stage0_classifier_scores.json,
20 FALSE / CORRECT-style-surface statements) to produce bilateral
fooling metrics.

Headline metrics
----------------
UNILATERAL (A-side only):
  - mean P(truthful, full) and mean P(truthful, cleaned) per family
  - "robustness": fraction of A-sides with P(truthful) > 0.5
  - cleaned - full shift (positive = cleaned classifier recovers the
    TRUE label better than full)

BILATERAL (A-side vs §5.1 B-side):
  - Same-index pairing (n=20): fool_i <=> P_B_i > P_A_i
  - All-pairs (n=400):        fool_ij <=> P_B_j > P_A_i
  - Fooling rate drop = fool_full - fool_cleaned

No classifier retraining. All six pickles loaded read-only. HF cache is
forced offline via env vars set before any model import (matches the
convention used in scripts/score_singleton.py and the v5/v6 scorers).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "stage0_v7a_bilateral"
GEN_JSON = OUT_DIR / "stage0_singleton_v7a_generations.json"
JUDGE_JSON = OUT_DIR / "stage0_singleton_v7a_judge.json"
OUT_JSON = OUT_DIR / "stage0_singleton_v7a_classifier_scores.json"

# §5.1 B-side already-scored singleton (20 FALSE / CORRECT-style-surface)
SIDE_B_GEN = REPO_ROOT / "stage0" / "stage0_generations.json"
SIDE_B_SCORES = REPO_ROOT / "stage0" / "stage0_classifier_scores.json"

ARTIFACTS = REPO_ROOT / "artifacts"
PICKLE_PATHS: dict[tuple[str, str], Path] = {
    ("surface_lr", "full"): ARTIFACTS / "surface_lr_full.pkl",
    ("surface_lr", "cleaned"): ARTIFACTS / "surface_lr_cleaned.pkl",
    ("BGE-large", "full"): ARTIFACTS / "embedding_lr_full.pkl",
    ("BGE-large", "cleaned"): ARTIFACTS / "embedding_lr_cleaned.pkl",
    ("ModernBERT-base", "full"): ARTIFACTS / "modernbert_lr_full.pkl",
    ("ModernBERT-base", "cleaned"): ARTIFACTS / "modernbert_lr_cleaned.pkl",
}

# Existing score file's column names (from scripts/score_singleton.py)
SIDE_B_PTRUTH_KEYS = {
    ("surface_lr", "full"): "surface_lr_full_P_truthful",
    ("surface_lr", "cleaned"): "surface_lr_cleaned_P_truthful",
    ("BGE-large", "full"): "embedding_lr_full_P_truthful",
    ("BGE-large", "cleaned"): "embedding_lr_cleaned_P_truthful",
    ("ModernBERT-base", "full"): "modernbert_lr_full_P_truthful",
    ("ModernBERT-base", "cleaned"): "modernbert_lr_cleaned_P_truthful",
}

BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
BGE_BATCH = 32
BGE_NORMALIZE = True
MBERT_MODEL_ID = "answerdotai/ModernBERT-base"
MBERT_BATCH = 32
MBERT_MAX_LENGTH = 512

_HF_HOME_DEFAULT = ARTIFACTS / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME_DEFAULT / "hub"))
_HF_HOME_DEFAULT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FAMILIES = ["surface_lr", "BGE-large", "ModernBERT-base"]
SPLITS = ["full", "cleaned"]


def _load_pickle(label: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label}: missing pickle {path}")
    art = joblib.load(path)
    if "pipeline" not in art:
        raise RuntimeError(f"{label}: pickle has no 'pipeline' key")
    return art


def _proba_truthful(pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipeline.steps[-1][1]
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 not in classes:
        raise RuntimeError(f"Classifier missing class 1; classes={classes}")
    col = classes.index(1)
    return np.asarray(pipeline.predict_proba(X)[:, col], dtype=float)


def _build_surface_matrix_singleton(
    texts: list[str], feature_cols: list[str], len_gap_mean: float
) -> np.ndarray:
    """Singleton feature matrix - there's no paired partner to compute a
    genuine len_gap from, so we override with each pipeline's trained
    mean (same convention scripts/score_singleton.py uses)."""
    from scripts.surface_features_text import extract_surface10
    rows = []
    for s in texts:
        f = extract_surface10(s)
        f["len_gap"] = float(len_gap_mean)
        rows.append([float(f[c]) for c in feature_cols])
    return np.asarray(rows, dtype=float)


def _encode_bge(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL_ID)
    X = model.encode(
        texts, batch_size=BGE_BATCH, normalize_embeddings=BGE_NORMALIZE,
        show_progress_bar=False, convert_to_numpy=True,
    )
    return np.asarray(X, dtype=np.float32)


def _encode_modernbert(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MBERT_MODEL_ID)
    mdl = AutoModel.from_pretrained(MBERT_MODEL_ID).to(device)
    mdl.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), MBERT_BATCH):
            batch = texts[start:start + MBERT_BATCH]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=MBERT_MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            cls = out.last_hidden_state[:, 0, :]
            rows.append(cls.detach().cpu().to(torch.float32).numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    print("=" * 72)
    print("score_singleton_v7a.py  (bilateral A-side TRUE vs §5.1 B-side FALSE)")
    print("=" * 72)

    # --- v7(a) A-side gens + judge ----------------------------------------
    gens = sorted(_load_json(GEN_JSON), key=lambda r: int(r["id"]))
    a_ids = [int(r["id"]) for r in gens]
    a_topics = [str(r["topic"]) for r in gens]
    a_texts = [str(r["statement"]) for r in gens]
    a_wcs = [int(r.get("word_count", 0)) for r in gens]

    judge_mask: np.ndarray
    if JUDGE_JSON.exists():
        jrec = {int(r["id"]): r for r in _load_json(JUDGE_JSON)}
        judge_mask = np.array(
            [bool(jrec.get(i, {}).get("judge_agrees_true", False))
             for i in a_ids],
            dtype=bool,
        )
        print(f"Loaded v7a judge file: {sum(judge_mask)}/{len(a_ids)} "
              f"statements judged TRUE by gpt-5.4")
    else:
        judge_mask = np.ones(len(a_ids), dtype=bool)
        print("No v7a judge file yet; treating all A-sides as judge-passed "
              "for scoring purposes.")

    print(f"A-side word counts: min={min(a_wcs)} max={max(a_wcs)} "
          f"mean={float(np.mean(a_wcs)):.2f}")

    # --- §5.1 B-side pre-scored + its gens (for metadata) -----------------
    b_score_raw = sorted(_load_json(SIDE_B_SCORES),
                         key=lambda r: int(r["id"]))
    b_gen_raw = sorted(_load_json(SIDE_B_GEN), key=lambda r: int(r["id"]))
    if len(b_score_raw) != 20 or len(b_gen_raw) != 20:
        raise RuntimeError(
            f"§5.1 B-side files must each have 20 records; got "
            f"{len(b_score_raw)} scores / {len(b_gen_raw)} gens.")
    b_ids = [int(r["id"]) for r in b_score_raw]
    b_topics = [str(r["topic"]) for r in b_gen_raw]
    b_texts = [str(r["statement"]) for r in b_gen_raw]

    P_B: dict[tuple[str, str], np.ndarray] = {}
    for fam in FAMILIES:
        for split in SPLITS:
            key = SIDE_B_PTRUTH_KEYS[(fam, split)]
            P_B[(fam, split)] = np.array(
                [float(r[key]) for r in b_score_raw], dtype=float)

    # --- Load the 6 pickles -----------------------------------------------
    pickles: dict[tuple[str, str], dict] = {}
    for (fam, split), path in PICKLE_PATHS.items():
        pickles[(fam, split)] = _load_pickle(f"{fam}/{split}", path)
        print(f"  {fam:<16s} {split:<8s} <- {path.relative_to(REPO_ROOT)}")

    # --- Encode A-sides once per family -----------------------------------
    surf_cols_full = pickles[("surface_lr", "full")]["feature_cols"]
    surf_cols_clean = pickles[("surface_lr", "cleaned")]["feature_cols"]
    if surf_cols_full != surf_cols_clean:
        raise RuntimeError("surface feature_cols disagree between full and cleaned.")
    Xa_surf_full = _build_surface_matrix_singleton(
        a_texts, surf_cols_full,
        pickles[("surface_lr", "full")]["train_len_gap_mean"])
    Xa_surf_clean = _build_surface_matrix_singleton(
        a_texts, surf_cols_clean,
        pickles[("surface_lr", "cleaned")]["train_len_gap_mean"])

    print("\nEncoding A-sides with BGE-large ...")
    Xa_bge = _encode_bge(a_texts)
    print(f"  Xa_bge shape = {Xa_bge.shape}")

    print("Encoding A-sides with ModernBERT-base ...")
    Xa_mb = _encode_modernbert(a_texts)
    print(f"  Xa_mb shape = {Xa_mb.shape}")

    P_A: dict[tuple[str, str], np.ndarray] = {}
    P_A[("surface_lr", "full")] = _proba_truthful(
        pickles[("surface_lr", "full")]["pipeline"], Xa_surf_full)
    P_A[("surface_lr", "cleaned")] = _proba_truthful(
        pickles[("surface_lr", "cleaned")]["pipeline"], Xa_surf_clean)
    for fam, Xa in [("BGE-large", Xa_bge), ("ModernBERT-base", Xa_mb)]:
        for split in SPLITS:
            P_A[(fam, split)] = _proba_truthful(
                pickles[(fam, split)]["pipeline"], Xa)

    # --- Unilateral A-side aggregates -------------------------------------
    def _uni(pa: np.ndarray, mask: np.ndarray | None = None) -> dict:
        arr = pa if mask is None else pa[mask]
        n = int(arr.shape[0])
        if n == 0:
            return {"n": 0, "mean_P": float("nan"),
                    "robust_rate": float("nan")}
        return {
            "n": n,
            "mean_P": float(arr.mean()),
            "robust_rate": float((arr > 0.5).mean()),
        }

    uni_all: dict[str, dict] = {}
    uni_judge: dict[str, dict] = {}
    for fam in FAMILIES:
        for split in SPLITS:
            key = f"{fam}_{split}"
            uni_all[key] = _uni(P_A[(fam, split)], None)
            uni_judge[key] = _uni(P_A[(fam, split)], judge_mask)

    # --- Bilateral pairing metrics ----------------------------------------
    def _bilateral_same_index(pa: np.ndarray, pb: np.ndarray,
                              mask: np.ndarray | None = None) -> dict:
        if mask is not None:
            pa = pa[mask]
            pb = pb[mask]
        n = int(pa.shape[0])
        if n == 0:
            return {"n_pairs": 0, "fool_rate": float("nan"),
                    "mean_P_A": float("nan"), "mean_P_B": float("nan"),
                    "mean_gap_B_minus_A": float("nan")}
        fool = (pb > pa).astype(int)
        return {
            "n_pairs": n,
            "fool_rate": float(fool.mean()),
            "mean_P_A": float(pa.mean()),
            "mean_P_B": float(pb.mean()),
            "mean_gap_B_minus_A": float((pb - pa).mean()),
        }

    def _bilateral_all_pairs(pa: np.ndarray, pb: np.ndarray,
                             mask_a: np.ndarray | None = None) -> dict:
        a = pa if mask_a is None else pa[mask_a]
        if a.shape[0] == 0 or pb.shape[0] == 0:
            return {"n_pairs": 0, "fool_rate": float("nan")}
        # Broadcast compare: shape (|A|, |B|)
        grid = pb[None, :] > a[:, None]
        return {
            "n_pairs": int(a.shape[0] * pb.shape[0]),
            "fool_rate": float(grid.mean()),
        }

    bi_same_all: dict[str, dict] = {}
    bi_same_judge: dict[str, dict] = {}
    bi_all_all: dict[str, dict] = {}
    bi_all_judge: dict[str, dict] = {}
    for fam in FAMILIES:
        for split in SPLITS:
            key = f"{fam}_{split}"
            pa = P_A[(fam, split)]
            pb = P_B[(fam, split)]
            bi_same_all[key] = _bilateral_same_index(pa, pb, None)
            bi_same_judge[key] = _bilateral_same_index(pa, pb, judge_mask)
            bi_all_all[key] = _bilateral_all_pairs(pa, pb, None)
            bi_all_judge[key] = _bilateral_all_pairs(pa, pb, judge_mask)

    # --- Per-pair records -------------------------------------------------
    per_pair_same_idx: list[dict] = []
    for i, sid in enumerate(a_ids):
        rec = {
            "a_id": int(sid),
            "a_topic": a_topics[i],
            "a_text": a_texts[i],
            "a_judge_agrees_true": bool(judge_mask[i]),
            "b_id": int(b_ids[i]),
            "b_topic": b_topics[i],
            "b_text": b_texts[i],
            "scores": {},
        }
        for fam in FAMILIES:
            for split in SPLITS:
                key = f"{fam}_{split}"
                pa = float(P_A[(fam, split)][i])
                pb = float(P_B[(fam, split)][i])
                rec["scores"][key] = {
                    "P_A_truthful": pa,
                    "P_B_truthful": pb,
                    "classifier_fooled": bool(pb > pa),
                }
        per_pair_same_idx.append(rec)

    # --- Print headline ---------------------------------------------------
    n_a = len(a_ids)
    n_judge = int(judge_mask.sum())

    def _print_unilateral(title: str, agg: dict, n: int) -> None:
        print("\n" + "=" * 88)
        print(f"V7A UNILATERAL A-SIDE PROBE (n={n})  {title}")
        print("=" * 88)
        print(f"  {'Classifier':<17s} {'split':<8s} {'mean P(truthful)':>18s} "
              f"{'P>0.5 rate':>12s}  {'cleaned - full':>16s}")
        print("  " + "-" * 82)
        for fam in FAMILIES:
            mf = agg[f"{fam}_full"]["mean_P"]
            mc = agg[f"{fam}_cleaned"]["mean_P"]
            rf = agg[f"{fam}_full"]["robust_rate"]
            rc = agg[f"{fam}_cleaned"]["robust_rate"]
            shift = mc - mf if (mf == mf and mc == mc) else float("nan")
            for split, mean_v, rate_v in [("full", mf, rf), ("cleaned", mc, rc)]:
                tag = f"{shift:+.4f}" if split == "cleaned" else ""
                print(f"  {fam:<17s} {split:<8s} {mean_v:>18.4f} "
                      f"{rate_v:>11.3f}   {tag:>16s}")

    _print_unilateral("all A-sides", uni_all, n_a)
    if n_judge != n_a:
        _print_unilateral("judge-passed A-sides", uni_judge, n_judge)

    def _print_bilateral_same(title: str, agg: dict, n: int) -> None:
        print("\n" + "=" * 88)
        print(f"V7A BILATERAL SAME-INDEX PAIRING (n={n})  {title}")
        print("=" * 88)
        print(f"  {'Classifier':<17s} {'split':<8s} "
              f"{'fool_rate (B>A)':>16s}  {'mean gap (B-A)':>15s}  "
              f"{'fool_drop':>10s}")
        print("  " + "-" * 78)
        for fam in FAMILIES:
            ff = agg[f"{fam}_full"]["fool_rate"]
            fc = agg[f"{fam}_cleaned"]["fool_rate"]
            gf = agg[f"{fam}_full"]["mean_gap_B_minus_A"]
            gc = agg[f"{fam}_cleaned"]["mean_gap_B_minus_A"]
            drop = ff - fc if (ff == ff and fc == fc) else float("nan")
            for split, fr, g in [("full", ff, gf), ("cleaned", fc, gc)]:
                tag = f"{drop:+.4f}" if split == "cleaned" else ""
                print(f"  {fam:<17s} {split:<8s} "
                      f"{fr:>16.4f}  {g:>+15.4f}  {tag:>10s}")

    _print_bilateral_same("all A-sides paired with §5.1 B-side by index",
                          bi_same_all, n_a)
    if n_judge != n_a:
        _print_bilateral_same(
            "judge-passed A-sides paired with §5.1 B-side by index",
            bi_same_judge, n_judge)

    def _print_bilateral_all(title: str, agg: dict, na: int, nb: int) -> None:
        print("\n" + "=" * 88)
        print(f"V7A BILATERAL ALL-PAIRS ({na}x{nb}={na*nb})  {title}")
        print("=" * 88)
        print(f"  {'Classifier':<17s} {'full fool':>12s} "
              f"{'clean fool':>12s}  {'drop':>10s}")
        print("  " + "-" * 60)
        for fam in FAMILIES:
            ff = agg[f"{fam}_full"]["fool_rate"]
            fc = agg[f"{fam}_cleaned"]["fool_rate"]
            drop = ff - fc if (ff == ff and fc == fc) else float("nan")
            print(f"  {fam:<17s} {ff:>12.4f} {fc:>12.4f}  {drop:>+10.4f}")

    _print_bilateral_all("all A-sides x all B-sides",
                         bi_all_all, n_a, len(b_ids))
    if n_judge != n_a:
        _print_bilateral_all(
            "judge-passed A-sides x all B-sides",
            bi_all_judge, n_judge, len(b_ids))

    # --- Persist payload --------------------------------------------------
    payload = {
        "a_ids": a_ids,
        "b_ids": b_ids,
        "n_a": int(n_a),
        "n_b": int(len(b_ids)),
        "n_a_judge_agrees_true": int(n_judge),
        "unilateral_all": uni_all,
        "unilateral_judge": uni_judge,
        "bilateral_same_index_all": bi_same_all,
        "bilateral_same_index_judge": bi_same_judge,
        "bilateral_all_pairs_all": bi_all_all,
        "bilateral_all_pairs_judge": bi_all_judge,
        "per_pair_same_index": per_pair_same_idx,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in score_singleton_v7a.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
