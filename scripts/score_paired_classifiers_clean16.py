#!/usr/bin/env python3
"""
Stage 0 paired-probe Step 2: score the 16 classifier-clean pairs through
all six existing Stage 0 classifiers and report pair-accuracy.

The 16 pair_ids included here are the subset of the 20 v3 paired
rewrites in stage0_paired_tqa/stage0_paired_generations.json that the
human truth-class review judged faithful AND that the paper's actual
surface_lr lexicon judged genuinely surface-flipped on both A and B.
The 4 excluded pair_ids (206, 222, 482, 688) leak the lexical token
"no" or "nothing" on the A-side and would dirty the surface-flip
contrast.

Per-pair scoring (no retraining; pickles loaded as-is):

  surface_lr  - extract_surface10(text) is called on each rewrite.
                len_gap is a pair-level feature; we inject the actual
                rel_len_gap(a_text, b_text) for the pair (identical
                value for both rows in the same pair, so it does not
                affect picked_side; it does affect P magnitudes).

  BGE         - bge-large-en-v1.5, normalize_embeddings=True (matches
                scripts/build_bge_embeddings.py).

  ModernBERT  - answerdotai/ModernBERT-base, CLS-token pooling, no
                normalization, max_length=512 (matches
                scripts/build_modernbert_embeddings.py).

For every (pair, classifier) cell we compute:
  P_A = P_truthful(a_side_text), P_B = P_truthful(b_side_text)
  picked = "A" if P_A > P_B else "B"
  correct_pick = (picked == "A")    # A is the true-content side

Aggregates per classifier: pair_accuracy, mean_P_A, mean_P_B, mean gap.

If any full-trained classifier scores pair_accuracy > 0.80 it is
flagged - that would mean the surface-flip did not fool the classifier
and is worth surfacing.

Outputs:
  stage0_paired_tqa/stage0_paired_classifier_scores_clean16.json
"""
from __future__ import annotations

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
GEN_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations.json"
OUT_JSON = (REPO_ROOT / "stage0_paired_tqa"
            / "stage0_paired_classifier_scores_clean16.json")

ARTIFACTS = REPO_ROOT / "artifacts"
PICKLE_PATHS: dict[tuple[str, str], Path] = {
    ("surface_lr",      "full"):    ARTIFACTS / "surface_lr_full.pkl",
    ("surface_lr",      "cleaned"): ARTIFACTS / "surface_lr_cleaned.pkl",
    ("BGE-large",       "full"):    ARTIFACTS / "embedding_lr_full.pkl",
    ("BGE-large",       "cleaned"): ARTIFACTS / "embedding_lr_cleaned.pkl",
    ("ModernBERT-base", "full"):    ARTIFACTS / "modernbert_lr_full.pkl",
    ("ModernBERT-base", "cleaned"): ARTIFACTS / "modernbert_lr_cleaned.pkl",
}

# Clean-16 pair IDs (per task spec).
CLEAN16_PAIR_IDS: list[int] = [
    31, 37, 41, 98, 101, 115, 136, 153,
    202, 216, 245, 357, 474, 521, 635, 690,
]

# BGE config
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
BGE_BATCH = 32
BGE_NORMALIZE = True

# ModernBERT config
MBERT_MODEL_ID = "answerdotai/ModernBERT-base"
MBERT_BATCH = 32
MBERT_MAX_LENGTH = 512

# HF cache / offline setup (matches scripts/score_paired_tqa.py)
_HF_HOME_DEFAULT = ARTIFACTS / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME_DEFAULT / "hub"))
_HF_HOME_DEFAULT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FULL_AUC_FLAG_THRESHOLD = 0.80


# --- Loaders --------------------------------------------------------------
def _load_pickle(label: str, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label}: missing pickle {path}")
    art = joblib.load(path)
    if "pipeline" not in art:
        raise RuntimeError(f"{label}: pickle has no 'pipeline' key")
    return art


def _load_pairs() -> list[dict]:
    with open(GEN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {int(r["pair_id"]): r for r in data}
    missing = [pid for pid in CLEAN16_PAIR_IDS if pid not in by_id]
    if missing:
        raise RuntimeError(f"Missing pair_ids in generation JSON: {missing}")
    return [by_id[pid] for pid in CLEAN16_PAIR_IDS]


# --- P_truthful from a fitted Pipeline ------------------------------------
def _proba_truthful(pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipeline.steps[-1][1]
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 not in classes:
        raise RuntimeError(f"Classifier has no class 1; classes_={classes}")
    col = classes.index(1)
    return np.asarray(pipeline.predict_proba(X)[:, col], dtype=float)


# --- Surface10 featurizer (paired len_gap injection) ----------------------
def _build_surface10(a_texts: list[str], b_texts: list[str],
                     feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from scripts.surface_features_text import extract_surface10, rel_len_gap
    Xa, Xb = [], []
    for a, b in zip(a_texts, b_texts):
        gap = float(rel_len_gap(a, b))
        fa = extract_surface10(a)
        fb = extract_surface10(b)
        fa["len_gap"] = gap
        fb["len_gap"] = gap
        Xa.append([float(fa[c]) for c in feature_cols])
        Xb.append([float(fb[c]) for c in feature_cols])
    return (np.asarray(Xa, dtype=float),
            np.asarray(Xb, dtype=float))


# --- Embedders ------------------------------------------------------------
def _encode_bge(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"  Loading BGE {BGE_MODEL_ID} ...")
    model = SentenceTransformer(BGE_MODEL_ID)
    X = model.encode(
        texts, batch_size=BGE_BATCH,
        normalize_embeddings=BGE_NORMALIZE,
        show_progress_bar=False, convert_to_numpy=True,
    )
    return np.asarray(X, dtype=np.float32)


def _encode_modernbert(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"  Loading ModernBERT {MBERT_MODEL_ID} (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MBERT_MODEL_ID)
    model = AutoModel.from_pretrained(MBERT_MODEL_ID)
    model.eval()
    model.to(device)
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), MBERT_BATCH):
            batch = texts[start:start + MBERT_BATCH]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=MBERT_MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]
            rows.append(cls.detach().cpu().to(torch.float32).numpy())
    X = np.concatenate(rows, axis=0).astype(np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("ModernBERT embeddings contain NaN/Inf")
    return X


# --- Main -----------------------------------------------------------------
FAMILIES = ["surface_lr", "BGE-large", "ModernBERT-base"]
SPLITS = ["full", "cleaned"]


def main() -> int:
    print("=" * 72)
    print("score_paired_classifiers_clean16.py")
    print("=" * 72)

    pairs = _load_pairs()
    n = len(pairs)
    print(f"Loaded {n} clean-16 pairs from {GEN_JSON.relative_to(REPO_ROOT)}")
    pids = [int(p["pair_id"]) for p in pairs]
    a_texts = [p["a_side_rewritten"] for p in pairs]
    b_texts = [p["b_side_rewritten"] for p in pairs]
    print(f"  pair_ids: {pids}")
    if pids != CLEAN16_PAIR_IDS:
        raise RuntimeError("pair_id ordering drifted; aborting")

    # Load all six pickles and print provenance
    print("\nLoading six classifier pickles ...")
    pickles: dict[tuple[str, str], dict] = {}
    for (fam, split), path in PICKLE_PATHS.items():
        art = _load_pickle(f"{fam}/{split}", path)
        pickles[(fam, split)] = art
        cv = art.get("cv_auc_group5", float("nan"))
        n_rows = art.get("n_rows", "?")
        n_pairs = art.get("n_pairs", "?")
        print(f"  {fam:<16s} {split:<8s} <- "
              f"{path.relative_to(REPO_ROOT)}  "
              f"cv_auc_group5={cv:.4f}  n_rows={n_rows}  n_pairs={n_pairs}")

    surf_full = pickles[("surface_lr", "full")]
    surf_clean = pickles[("surface_lr", "cleaned")]
    if surf_full["feature_cols"] != surf_clean["feature_cols"]:
        raise RuntimeError("surface_lr feature_cols disagree across splits")
    feature_cols = surf_full["feature_cols"]
    print(f"  surface_lr feature_cols = {feature_cols}")

    # --- Build per-classifier feature matrices ---------------------------
    print("\nBuilding feature matrices ...")
    Xa_surf, Xb_surf = _build_surface10(a_texts, b_texts, feature_cols)
    print(f"  surface10  Xa {Xa_surf.shape}  Xb {Xb_surf.shape}")

    print("\nEncoding 32 texts (16 A + 16 B) with BGE-large ...")
    X_bge = _encode_bge(a_texts + b_texts)
    Xa_bge, Xb_bge = X_bge[:n], X_bge[n:]
    print(f"  BGE  Xa {Xa_bge.shape}  Xb {Xb_bge.shape}")

    print("\nEncoding 32 texts (16 A + 16 B) with ModernBERT-base ...")
    X_mb = _encode_modernbert(a_texts + b_texts)
    Xa_mb, Xb_mb = X_mb[:n], X_mb[n:]
    print(f"  ModernBERT  Xa {Xa_mb.shape}  Xb {Xb_mb.shape}")

    # --- Score every (pair, classifier) cell -----------------------------
    feat_by_family = {
        "surface_lr":      (Xa_surf, Xb_surf),
        "BGE-large":       (Xa_bge, Xb_bge),
        "ModernBERT-base": (Xa_mb, Xb_mb),
    }
    P: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for fam in FAMILIES:
        Xa, Xb = feat_by_family[fam]
        for split in SPLITS:
            pipe = pickles[(fam, split)]["pipeline"]
            P[(fam, split)] = {
                "A": _proba_truthful(pipe, Xa),
                "B": _proba_truthful(pipe, Xb),
            }

    # --- Per-pair records ------------------------------------------------
    per_pair: list[dict] = []
    for i, pid in enumerate(pids):
        rec: dict = {
            "pair_id":     int(pid),
            "a_side_text": a_texts[i],
            "b_side_text": b_texts[i],
            "scores": {},
        }
        for fam in FAMILIES:
            for split in SPLITS:
                key = f"{fam}_{split}"
                pa = float(P[(fam, split)]["A"][i])
                pb = float(P[(fam, split)]["B"][i])
                picked = "A" if pa > pb else "B"
                rec["scores"][key] = {
                    "P_A": pa,
                    "P_B": pb,
                    "picked": picked,
                    "correct_pick": bool(picked == "A"),
                }
        per_pair.append(rec)

    # --- Aggregates ------------------------------------------------------
    aggregates: dict[str, dict] = {}
    for fam in FAMILIES:
        for split in SPLITS:
            key = f"{fam}_{split}"
            pa = P[(fam, split)]["A"]
            pb = P[(fam, split)]["B"]
            picked_A = (pa > pb).astype(int)
            aggregates[key] = {
                "pair_accuracy": float(picked_A.mean()),
                "n_correct":     int(picked_A.sum()),
                "n_total":       int(n),
                "mean_P_A":      float(pa.mean()),
                "mean_P_B":      float(pb.mean()),
                "mean_gap":      float((pa - pb).mean()),
            }

    cleaning_effect: dict[str, dict] = {}
    for fam in FAMILIES:
        full_acc = aggregates[f"{fam}_full"]["pair_accuracy"]
        clean_acc = aggregates[f"{fam}_cleaned"]["pair_accuracy"]
        cleaning_effect[fam] = {
            "full_acc":    full_acc,
            "cleaned_acc": clean_acc,
            "delta":       clean_acc - full_acc,
        }

    # --- Print headline tables ------------------------------------------
    print("\n" + "=" * 88)
    print(f"PAIRED PROBE CLASSIFIER RESULTS  (n={n} classifier-clean pairs)")
    print("=" * 88)
    hdr = (f"  {'Classifier':<17s} {'TrainedOn':<10s} "
           f"{'Pair acc':>10s}  {'mean P(A)':>10s}  "
           f"{'mean P(B)':>10s}  {'mean gap':>10s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for fam in FAMILIES:
        for split in SPLITS:
            a = aggregates[f"{fam}_{split}"]
            print(f"  {fam:<17s} {split:<10s} "
                  f"{a['n_correct']:>3d}/{a['n_total']:<3d} "
                  f"({a['pair_accuracy']:.3f})  "
                  f"{a['mean_P_A']:>10.4f}  "
                  f"{a['mean_P_B']:>10.4f}  "
                  f"{a['mean_gap']:>+10.4f}")

    print("\n" + "=" * 64)
    print("CLEANING EFFECT ON PAIR ACCURACY")
    print("=" * 64)
    print(f"  {'Family':<17s}  {'Full acc':>10s}  "
          f"{'Cleaned acc':>11s}  {'Delta':>10s}")
    print("  " + "-" * 56)
    for fam in FAMILIES:
        ce = cleaning_effect[fam]
        print(f"  {fam:<17s}  {ce['full_acc']:>10.4f}  "
              f"{ce['cleaned_acc']:>11.4f}  "
              f"{ce['delta']:>+10.4f}")
    print("  Interpretation: delta > 0  =>  cleaning helped robustness "
          "to the surface-flip.")

    # --- Per-pair compact table -----------------------------------------
    print("\n" + "=" * 88)
    print("PER-PAIR PICKED SIDE AND P_truthful OF PICKED SIDE")
    print("=" * 88)
    cols = [
        ("surface_lr", "full"),     ("surface_lr", "cleaned"),
        ("BGE-large", "full"),      ("BGE-large", "cleaned"),
        ("ModernBERT-base", "full"),("ModernBERT-base", "cleaned"),
    ]
    short = {
        ("surface_lr", "full"):       "surfLR_full",
        ("surface_lr", "cleaned"):    "surfLR_cln",
        ("BGE-large", "full"):        "BGE_full",
        ("BGE-large", "cleaned"):     "BGE_cln",
        ("ModernBERT-base", "full"):  "MBERT_full",
        ("ModernBERT-base", "cleaned"): "MBERT_cln",
    }
    h = "  pid  " + "  ".join(f"{short[c]:>11s}" for c in cols)
    print(h)
    print("  " + "-" * (len(h) - 2))
    for rec in per_pair:
        cells = []
        for c in cols:
            s = rec["scores"][f"{c[0]}_{c[1]}"]
            picked = s["picked"]
            p = s["P_A"] if picked == "A" else s["P_B"]
            cells.append(f"{picked}/{p:.3f}".rjust(11))
        print(f"  {rec['pair_id']:>3d}  " + "  ".join(cells))

    # --- Flags ----------------------------------------------------------
    flags: list[str] = []
    for fam in FAMILIES:
        full_acc = aggregates[f"{fam}_full"]["pair_accuracy"]
        if full_acc > FULL_AUC_FLAG_THRESHOLD:
            flags.append(
                f"FLAG: {fam} full-trained pair_accuracy = {full_acc:.4f} "
                f"> {FULL_AUC_FLAG_THRESHOLD}; "
                f"the surface-flip did not fool this classifier."
            )
    if flags:
        print("\n" + "!" * 72)
        for line in flags:
            print(line)
        print("!" * 72)

    # --- Save JSON ------------------------------------------------------
    out_payload = {
        "clean16_pair_ids":            CLEAN16_PAIR_IDS,
        "n_pairs":                     int(n),
        "generation_source":           str(GEN_JSON.relative_to(REPO_ROOT)),
        "classifier_pickles": {
            f"{fam}_{split}": str(p.relative_to(REPO_ROOT))
            for (fam, split), p in PICKLE_PATHS.items()
        },
        "surface10_len_gap_injection": (
            "rel_len_gap(a_text, b_text), per pair, identical for both "
            "rows; does not affect picked_side"
        ),
        "per_pair":         per_pair,
        "aggregates":       aggregates,
        "cleaning_effect":  cleaning_effect,
        "flags":            flags,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in score_paired_classifiers_clean16.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
