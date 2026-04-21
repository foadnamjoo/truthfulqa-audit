#!/usr/bin/env python3
"""
Step 6 + 7 (paired-pair audit probe) - score every pair's a_side_rewritten
and b_side_rewritten with all SIX classifier pickles trained in Stage 0:

    surface_lr_full.pkl            surface_lr_cleaned.pkl
    embedding_lr_full.pkl   (BGE)  embedding_lr_cleaned.pkl   (BGE)
    modernbert_lr_full.pkl         modernbert_lr_cleaned.pkl

For each pair and each classifier:

    P_a = predict_proba(a_side_rewritten)[:, 1]   # P(truthful-looking)
    P_b = predict_proba(b_side_rewritten)[:, 1]
    picked_side = "a" if P_a > P_b else "b"
    correct = (picked_side == "a")          # A always carries the TRUE label

Per-classifier aggregates over the 20 pairs:
    pair_accuracy = mean(correct)
    mean_margin   = mean(P_a - P_b)         # positive => leans toward A

The 3-family comparison table compares pair_acc_full vs pair_acc_cleaned
for each of {surface_lr, embedding_lr (BGE), modernbert_lr}.

Outputs:
    stage0_paired_tqa/stage0_paired_classifier_scores.json
        list of 20 records, one per pair_id, with per-classifier
        P(a), P(b), picked_side, correct_<family>_<train_set> flags.
        Also includes per-family aggregates as a summary tail.
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
GEN_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations.json"
JUDGE_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_judge.json"
OUT_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_classifier_scores.json"

ARTIFACTS = REPO_ROOT / "artifacts"
SURFACE_FULL_PKL = ARTIFACTS / "surface_lr_full.pkl"
SURFACE_CLEAN_PKL = ARTIFACTS / "surface_lr_cleaned.pkl"
EMBED_FULL_PKL = ARTIFACTS / "embedding_lr_full.pkl"
EMBED_CLEAN_PKL = ARTIFACTS / "embedding_lr_cleaned.pkl"
MBERT_FULL_PKL = ARTIFACTS / "modernbert_lr_full.pkl"
MBERT_CLEAN_PKL = ARTIFACTS / "modernbert_lr_cleaned.pkl"

# BGE config (mirrors scripts/build_bge_embeddings.py and score_singleton.py)
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
BGE_BATCH = 20
BGE_NORMALIZE = True

# ModernBERT config (mirrors scripts/build_modernbert_embeddings.py and
# score_singleton.py - CLS pooling, NO normalization, max_length=512)
MBERT_MODEL_ID = "answerdotai/ModernBERT-base"
MBERT_BATCH = 20
MBERT_MAX_LENGTH = 512


# --- HF cache / offline setup (identical to score_singleton.py) ------------
_HF_HOME_DEFAULT = ARTIFACTS / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME_DEFAULT / "hub"))
_HF_HOME_DEFAULT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --- IO helpers ------------------------------------------------------------
def _load_pairs() -> list[dict]:
    if not GEN_JSON.exists():
        raise RuntimeError(f"Missing {GEN_JSON}")
    with open(GEN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data, key=lambda r: int(r["pair_id"]))


def _load_judge() -> dict[int, dict]:
    if not JUDGE_JSON.exists():
        return {}
    with open(JUDGE_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {int(r["pair_id"]): r for r in d}


def _load_surface_pkl(path: Path, label: str) -> dict:
    art = joblib.load(path)
    for k in ("pipeline", "feature_cols", "train_len_gap_mean"):
        if k not in art:
            raise RuntimeError(f"{label}: missing key {k!r}")
    return art


def _load_embed_pkl(path: Path, label: str) -> dict:
    art = joblib.load(path)
    if "pipeline" not in art:
        raise RuntimeError(f"{label}: missing key 'pipeline'")
    return art


def _proba_class1(pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipeline.steps[-1][1]
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 not in classes:
        raise RuntimeError(f"classifier missing class 1; classes_={classes}")
    col = classes.index(1)
    return np.asarray(pipeline.predict_proba(X)[:, col], dtype=float)


# --- Featurizers -----------------------------------------------------------
def _build_surface_matrix(statements: list[str], feature_cols: list[str],
                          len_gap_mean: float) -> np.ndarray:
    from scripts.surface_features_text import extract_surface10
    rows = []
    for s in statements:
        feats = extract_surface10(s)
        feats["len_gap"] = float(len_gap_mean)
        rows.append([float(feats[c]) for c in feature_cols])
    return np.asarray(rows, dtype=float)


def _encode_bge(statements: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"Loading BGE {BGE_MODEL_ID} ...")
    model = SentenceTransformer(BGE_MODEL_ID)
    X = model.encode(
        statements, batch_size=BGE_BATCH,
        normalize_embeddings=BGE_NORMALIZE,
        show_progress_bar=False, convert_to_numpy=True,
    )
    return np.asarray(X, dtype=np.float32)


def _encode_modernbert(statements: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Loading ModernBERT {MBERT_MODEL_ID} (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MBERT_MODEL_ID)
    model = AutoModel.from_pretrained(MBERT_MODEL_ID)
    model.eval()
    model.to(device)
    out_rows = []
    n = len(statements)
    with torch.no_grad():
        for start in range(0, n, MBERT_BATCH):
            batch = statements[start:start + MBERT_BATCH]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=MBERT_MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]
            out_rows.append(cls.detach().cpu().to(torch.float32).numpy())
    X = np.concatenate(out_rows, axis=0).astype(np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("ModernBERT embedding contained NaN/Inf")
    return X


# --- Per-family scoring ----------------------------------------------------
FAMILIES = ["surface_lr", "embedding_lr", "modernbert_lr"]
SPLITS = ["full", "cleaned"]


def main() -> int:
    print("=" * 72)
    print("Step 6/7 - score_paired_tqa.py")
    print("=" * 72)

    pairs = _load_pairs()
    n = len(pairs)
    pair_ids = [int(p["pair_id"]) for p in pairs]
    a_texts = [p["a_side_rewritten"] for p in pairs]
    b_texts = [p["b_side_rewritten"] for p in pairs]
    print(f"Loaded {n} pairs from {GEN_JSON}")
    print(f"  pair_ids: {pair_ids}")

    judge_by_id = _load_judge()
    if judge_by_id:
        print(f"  judge file present: {len(judge_by_id)} records")

    # --- Load all six pickles ------------------------------------------------
    print("\nLoading six classifier pickles ...")
    classifiers: dict[str, dict[str, dict]] = {fam: {} for fam in FAMILIES}
    classifiers["surface_lr"]["full"] = _load_surface_pkl(
        SURFACE_FULL_PKL, "surface_lr_full")
    classifiers["surface_lr"]["cleaned"] = _load_surface_pkl(
        SURFACE_CLEAN_PKL, "surface_lr_cleaned")
    classifiers["embedding_lr"]["full"] = _load_embed_pkl(
        EMBED_FULL_PKL, "embedding_lr_full")
    classifiers["embedding_lr"]["cleaned"] = _load_embed_pkl(
        EMBED_CLEAN_PKL, "embedding_lr_cleaned")
    classifiers["modernbert_lr"]["full"] = _load_embed_pkl(
        MBERT_FULL_PKL, "modernbert_lr_full")
    classifiers["modernbert_lr"]["cleaned"] = _load_embed_pkl(
        MBERT_CLEAN_PKL, "modernbert_lr_cleaned")
    for fam in FAMILIES:
        for split in SPLITS:
            cv = float(classifiers[fam][split].get("cv_auc_group5", float("nan")))
            print(f"  {fam:<14s} {split:<8s} cv_auc={cv:.4f}")

    if (classifiers["surface_lr"]["full"]["feature_cols"]
            != classifiers["surface_lr"]["cleaned"]["feature_cols"]):
        raise RuntimeError("surface feature_cols disagree across pickles")

    # --- Surface matrices (each split has its own len_gap mean) -------------
    print("\nBuilding surface matrices ...")
    fcols = classifiers["surface_lr"]["full"]["feature_cols"]
    print(f"  feature_cols = {fcols}")
    Xa_surf_full = _build_surface_matrix(
        a_texts, fcols,
        classifiers["surface_lr"]["full"]["train_len_gap_mean"])
    Xb_surf_full = _build_surface_matrix(
        b_texts, fcols,
        classifiers["surface_lr"]["full"]["train_len_gap_mean"])
    Xa_surf_clean = _build_surface_matrix(
        a_texts, fcols,
        classifiers["surface_lr"]["cleaned"]["train_len_gap_mean"])
    Xb_surf_clean = _build_surface_matrix(
        b_texts, fcols,
        classifiers["surface_lr"]["cleaned"]["train_len_gap_mean"])

    # --- BGE -----------------------------------------------------------------
    print("\nEncoding A and B with BGE-large-en-v1.5 ...")
    all_texts_bge = a_texts + b_texts
    X_bge_all = _encode_bge(all_texts_bge)
    Xa_bge = X_bge_all[:n]
    Xb_bge = X_bge_all[n:]
    print(f"  Xa_bge {Xa_bge.shape}  Xb_bge {Xb_bge.shape}")

    # --- ModernBERT ----------------------------------------------------------
    print("\nEncoding A and B with ModernBERT (CLS, no norm) ...")
    X_mbert_all = _encode_modernbert(all_texts_bge)
    Xa_mb = X_mbert_all[:n]
    Xb_mb = X_mbert_all[n:]
    print(f"  Xa_mbert {Xa_mb.shape}  Xb_mbert {Xb_mb.shape}")

    # --- Score every cell ---------------------------------------------------
    P: dict[str, dict[str, dict[str, np.ndarray]]] = {
        fam: {split: {} for split in SPLITS} for fam in FAMILIES
    }
    P["surface_lr"]["full"]["a"] = _proba_class1(
        classifiers["surface_lr"]["full"]["pipeline"], Xa_surf_full)
    P["surface_lr"]["full"]["b"] = _proba_class1(
        classifiers["surface_lr"]["full"]["pipeline"], Xb_surf_full)
    P["surface_lr"]["cleaned"]["a"] = _proba_class1(
        classifiers["surface_lr"]["cleaned"]["pipeline"], Xa_surf_clean)
    P["surface_lr"]["cleaned"]["b"] = _proba_class1(
        classifiers["surface_lr"]["cleaned"]["pipeline"], Xb_surf_clean)
    P["embedding_lr"]["full"]["a"] = _proba_class1(
        classifiers["embedding_lr"]["full"]["pipeline"], Xa_bge)
    P["embedding_lr"]["full"]["b"] = _proba_class1(
        classifiers["embedding_lr"]["full"]["pipeline"], Xb_bge)
    P["embedding_lr"]["cleaned"]["a"] = _proba_class1(
        classifiers["embedding_lr"]["cleaned"]["pipeline"], Xa_bge)
    P["embedding_lr"]["cleaned"]["b"] = _proba_class1(
        classifiers["embedding_lr"]["cleaned"]["pipeline"], Xb_bge)
    P["modernbert_lr"]["full"]["a"] = _proba_class1(
        classifiers["modernbert_lr"]["full"]["pipeline"], Xa_mb)
    P["modernbert_lr"]["full"]["b"] = _proba_class1(
        classifiers["modernbert_lr"]["full"]["pipeline"], Xb_mb)
    P["modernbert_lr"]["cleaned"]["a"] = _proba_class1(
        classifiers["modernbert_lr"]["cleaned"]["pipeline"], Xa_mb)
    P["modernbert_lr"]["cleaned"]["b"] = _proba_class1(
        classifiers["modernbert_lr"]["cleaned"]["pipeline"], Xb_mb)

    # --- Per-pair records ---------------------------------------------------
    records: list[dict] = []
    for i, pid in enumerate(pair_ids):
        rec: dict = {"pair_id": int(pid)}
        for fam in FAMILIES:
            for split in SPLITS:
                pa = float(P[fam][split]["a"][i])
                pb = float(P[fam][split]["b"][i])
                picked = "a" if pa > pb else "b"
                rec[f"{fam}_{split}_P_a"] = pa
                rec[f"{fam}_{split}_P_b"] = pb
                rec[f"{fam}_{split}_picked"] = picked
                rec[f"{fam}_{split}_correct"] = bool(picked == "a")
        records.append(rec)

    # --- Per-classifier aggregates (over all 20) ----------------------------
    aggregates: dict[str, dict[str, dict[str, float]]] = {
        fam: {split: {} for split in SPLITS} for fam in FAMILIES
    }
    for fam in FAMILIES:
        for split in SPLITS:
            pa = P[fam][split]["a"]
            pb = P[fam][split]["b"]
            correct = (pa > pb).astype(int)
            aggregates[fam][split] = {
                "pair_accuracy": float(correct.mean()),
                "n_correct": int(correct.sum()),
                "n_total": int(n),
                "mean_margin": float((pa - pb).mean()),
                "mean_P_a": float(pa.mean()),
                "mean_P_b": float(pb.mean()),
            }

    # --- Faithful-only subset (if judge file present) -----------------------
    if judge_by_id:
        faithful_idx = [i for i, pid in enumerate(pair_ids)
                        if judge_by_id.get(pid, {}).get("pair_faithful", False)]
        if faithful_idx:
            print(f"\nFaithful-subset aggregates (n={len(faithful_idx)}):")
            for fam in FAMILIES:
                for split in SPLITS:
                    pa = P[fam][split]["a"][faithful_idx]
                    pb = P[fam][split]["b"][faithful_idx]
                    correct = (pa > pb).astype(int)
                    aggregates[fam][split][
                        "pair_accuracy_faithful_only"
                    ] = float(correct.mean())
                    aggregates[fam][split][
                        "n_correct_faithful_only"
                    ] = int(correct.sum())
                    aggregates[fam][split][
                        "n_total_faithful_only"
                    ] = int(len(faithful_idx))
                    aggregates[fam][split][
                        "mean_margin_faithful_only"
                    ] = float((pa - pb).mean())

    # --- Comparison table ---------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 7 - 3-FAMILY PAIR-ACCURACY COMPARISON TABLE")
    print("=" * 72)
    print(f"  {'family':<22s} | {'pair_acc_full':>13s} | "
          f"{'pair_acc_cleaned':>16s} | "
          f"{'acc_gap':>8s} | {'margin_gap':>10s}")
    print("  " + "-" * 84)
    family_summary = {}
    for fam in FAMILIES:
        af = aggregates[fam]["full"]["pair_accuracy"]
        ac = aggregates[fam]["cleaned"]["pair_accuracy"]
        mf = aggregates[fam]["full"]["mean_margin"]
        mc = aggregates[fam]["cleaned"]["mean_margin"]
        gap = ac - af
        mgap = mc - mf
        print(f"  {fam:<22s} | {af:>13.4f} | {ac:>16.4f} | "
              f"{gap:>+8.4f} | {mgap:>+10.4f}")
        family_summary[fam] = {
            "pair_acc_full": af,
            "pair_acc_cleaned": ac,
            "acc_gap_cleaned_minus_full": gap,
            "mean_margin_full": mf,
            "mean_margin_cleaned": mc,
            "margin_gap_cleaned_minus_full": mgap,
        }

    print(f"\n  Notes:")
    print(f"  - acc_gap > 0 means cleaning HELPED (cleaned classifier "
          f"more robust to surface swap).")
    print(f"  - margin_gap > 0 means cleaning made the model lean MORE "
          f"toward the true (A) side.")
    if judge_by_id:
        n_faith = sum(1 for r in judge_by_id.values()
                      if r.get("pair_faithful"))
        print(f"\n  judge says pair_faithful: {n_faith}/{n}")
        if n_faith > 0:
            print(f"\n  FAITHFUL-ONLY subset table (n={n_faith}):")
            for fam in FAMILIES:
                af = aggregates[fam]["full"].get(
                    "pair_accuracy_faithful_only", float("nan"))
                ac = aggregates[fam]["cleaned"].get(
                    "pair_accuracy_faithful_only", float("nan"))
                print(f"    {fam:<22s} acc_full={af:.4f} "
                      f"acc_clean={ac:.4f} gap={ac-af:+.4f}")

    # --- Per-pair preview ---------------------------------------------------
    print("\nPer-pair correctness flags (1 = classifier picked A = true side):")
    hdr = ("  pid | judge_faith | "
           "surf_F surf_C | bge_F bge_C | mb_F mb_C")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in records:
        pid = r["pair_id"]
        jf = "T" if judge_by_id.get(pid, {}).get("pair_faithful") else "F"
        sf = int(r["surface_lr_full_correct"])
        sc = int(r["surface_lr_cleaned_correct"])
        ef = int(r["embedding_lr_full_correct"])
        ec = int(r["embedding_lr_cleaned_correct"])
        mf = int(r["modernbert_lr_full_correct"])
        mc = int(r["modernbert_lr_cleaned_correct"])
        print(f"  {pid:>3d} |     {jf}      |   "
              f"{sf}      {sc}    |   {ef}     {ec}    |  "
              f"{mf}    {mc}")

    # --- Write output -------------------------------------------------------
    out = {
        "pair_records": records,
        "family_aggregates": aggregates,
        "family_summary_table": family_summary,
        "n_pairs": int(n),
        "classifier_pickles": {
            "surface_lr_full": str(SURFACE_FULL_PKL),
            "surface_lr_cleaned": str(SURFACE_CLEAN_PKL),
            "embedding_lr_full": str(EMBED_FULL_PKL),
            "embedding_lr_cleaned": str(EMBED_CLEAN_PKL),
            "modernbert_lr_full": str(MBERT_FULL_PKL),
            "modernbert_lr_cleaned": str(MBERT_CLEAN_PKL),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON}")
    print("Step 6/7 complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in score_paired_tqa.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
