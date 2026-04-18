#!/usr/bin/env python3
"""
Step 10 (extended for ModernBERT) - Score each of the 20 Stage-0
adversarial statements with all six trained classifiers:

    surface_lr_full.pkl      trained on full 790-pair / 1580-row surface10
    surface_lr_cleaned.pkl   trained on tau=0.52 528-pair / 1056-row surface10
    embedding_lr_full.pkl    trained on full BGE-large-en-v1.5 CLS
    embedding_lr_cleaned.pkl trained on tau=0.52 BGE-large-en-v1.5
    modernbert_lr_full.pkl   trained on full ModernBERT-base CLS        <-- NEW
    modernbert_lr_cleaned.pkl trained on tau=0.52 ModernBERT-base CLS    <-- NEW

Per-statement pipeline:

  1. Surface: extract_surface10 -> override len_gap with each pipeline's
     persisted train_len_gap_mean -> predict_proba class 1.

  2. BGE-large (sentence-transformers, normalize_embeddings=True,
     no prefix) -> predict_proba class 1.

  3. ModernBERT-base (transformers AutoTokenizer + AutoModel, CLS token
     from last_hidden_state, NO normalization, max_length=512, eval()+
     no_grad()) -> predict_proba class 1.

For every classifier we look up the class-1 column via pipeline.steps[-1]
[1].classes_ rather than hard-coding 1, so a future pickle with flipped
classes_ would not silently mis-score.

Reproducibility contract:

    The BGE and surface scores are deterministic given the same inputs and
    pipelines. On every run after the first, we load the previous
    stage0/stage0_classifier_scores.json (if it exists) and assert that
    each of the four existing P_truthful columns reproduces to within
    1e-9. If any value drifts beyond that tolerance, the run aborts with
    a clear error so a real regression cannot sneak into scale-up.

Output schema (sorted by id) - 7 keys per record:

    id,
    surface_lr_full_P_truthful, surface_lr_cleaned_P_truthful,
    embedding_lr_full_P_truthful, embedding_lr_cleaned_P_truthful,
    modernbert_lr_full_P_truthful, modernbert_lr_cleaned_P_truthful
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
STAGE0_DIR = REPO_ROOT / "stage0"
GEN_JSON = STAGE0_DIR / "stage0_generations.json"
OUT_JSON = STAGE0_DIR / "stage0_classifier_scores.json"

ARTIFACTS = REPO_ROOT / "artifacts"
SURFACE_FULL_PKL = ARTIFACTS / "surface_lr_full.pkl"
SURFACE_CLEAN_PKL = ARTIFACTS / "surface_lr_cleaned.pkl"
EMBED_FULL_PKL = ARTIFACTS / "embedding_lr_full.pkl"
EMBED_CLEAN_PKL = ARTIFACTS / "embedding_lr_cleaned.pkl"
MBERT_FULL_PKL = ARTIFACTS / "modernbert_lr_full.pkl"
MBERT_CLEAN_PKL = ARTIFACTS / "modernbert_lr_cleaned.pkl"

# BGE encoding config (mirrors scripts/build_bge_embeddings.py exactly)
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
BGE_BATCH = 20
BGE_NORMALIZE = True

# ModernBERT encoding config (mirrors scripts/build_modernbert_embeddings.py
# exactly - same model, same pooling, same max_length, no normalization).
MBERT_MODEL_ID = "answerdotai/ModernBERT-base"
MBERT_BATCH = 20
MBERT_MAX_LENGTH = 512

REPRO_TOL = 1e-9

# --- HF cache / offline setup ----------------------------------------------
_HF_HOME_DEFAULT = ARTIFACTS / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME_DEFAULT / "hub"))
_HF_HOME_DEFAULT.mkdir(parents=True, exist_ok=True)

# Both BGE (Step 6) and ModernBERT (post-Stage-0 Step B) are already cached
# under artifacts/hf_cache. Force offline so the HF libraries reuse the
# exact snapshots used at training time and do not probe huggingface.co.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --- IO helpers ------------------------------------------------------------
def _load_generations() -> list[dict]:
    if not GEN_JSON.exists():
        raise RuntimeError(
            f"{GEN_JSON} does not exist. Run Step 8 (generate_stage0.py) "
            f"first so this file contains all 20 generator records."
        )
    with open(GEN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 20:
        raise RuntimeError(
            f"{GEN_JSON} must contain exactly 20 records; got "
            f"{len(data) if isinstance(data, list) else type(data).__name__}."
        )
    return sorted(data, key=lambda r: r["id"])


def _load_prior_scores() -> dict[int, dict] | None:
    """Return the prior stage0_classifier_scores.json keyed by id, or None
    if the file does not yet exist. Used for the reproducibility check."""
    if not OUT_JSON.exists():
        return None
    with open(OUT_JSON, "r", encoding="utf-8") as f:
        prior = json.load(f)
    if not isinstance(prior, list):
        raise RuntimeError(
            f"{OUT_JSON} exists but is not a JSON list. Refusing to merge."
        )
    return {int(r["id"]): r for r in prior}


def _load_surface_pickle(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} pickle: {path}")
    art = joblib.load(path)
    if not isinstance(art, dict):
        raise RuntimeError(
            f"{label}: expected dict artifact, got {type(art).__name__}")
    for k in ("pipeline", "feature_cols", "train_len_gap_mean"):
        if k not in art:
            raise RuntimeError(f"{label}: artifact missing required key {k!r}")
    return art


def _load_embed_pickle(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} pickle: {path}")
    art = joblib.load(path)
    if not isinstance(art, dict) or "pipeline" not in art:
        raise RuntimeError(f"{label}: artifact missing required 'pipeline' key")
    return art


# --- Feature / encoding helpers -------------------------------------------
def _build_surface_matrix(
    statements: list[str],
    feature_cols: list[str],
    len_gap_mean: float,
) -> np.ndarray:
    """Build (N, 10) surface feature matrix in `feature_cols` order, with
    the `len_gap` column overridden by `len_gap_mean`."""
    from scripts.surface_features_text import extract_surface10
    rows = []
    for s in statements:
        feats = extract_surface10(s)
        feats["len_gap"] = float(len_gap_mean)
        rows.append([float(feats[c]) for c in feature_cols])
    return np.asarray(rows, dtype=float)


def _encode_bge(statements: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"Loading BGE model {BGE_MODEL_ID} ...")
    model = SentenceTransformer(BGE_MODEL_ID)
    dim = model.get_sentence_embedding_dimension()
    print(f"  loaded; embedding dim = {dim}")
    X = model.encode(
        statements,
        batch_size=BGE_BATCH,
        normalize_embeddings=BGE_NORMALIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    X = np.asarray(X, dtype=np.float32)
    if X.shape != (len(statements), dim):
        raise RuntimeError(f"Unexpected BGE-encoded shape: {X.shape}")
    return X


def _encode_modernbert(statements: list[str]) -> np.ndarray:
    """Mirrors build_modernbert_embeddings.py:
       - AutoTokenizer / AutoModel from transformers
       - CLS token of last_hidden_state
       - NO normalization
       - max_length = 512
       - eval() + torch.no_grad()"""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Loading ModernBERT {MBERT_MODEL_ID} (device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MBERT_MODEL_ID)
    model = AutoModel.from_pretrained(MBERT_MODEL_ID)
    model.eval()
    model.to(device)

    hidden = int(getattr(model.config, "hidden_size", 0))
    print(f"  loaded; hidden_size={hidden}")

    out_rows: list[np.ndarray] = []
    n = len(statements)
    with torch.no_grad():
        for start in range(0, n, MBERT_BATCH):
            batch = statements[start:start + MBERT_BATCH]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MBERT_MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]
            out_rows.append(cls.detach().cpu().to(torch.float32).numpy())
    X = np.concatenate(out_rows, axis=0).astype(np.float32)
    if X.shape != (n, hidden):
        raise RuntimeError(f"Unexpected ModernBERT-encoded shape: {X.shape}")
    if not np.isfinite(X).all():
        raise RuntimeError("ModernBERT embedding contained NaN/Inf")
    return X


def _proba_class1(pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipeline.steps[-1][1]
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 not in classes:
        raise RuntimeError(
            f"Trained classifier does not have class label 1 in classes_"
            f"={classes}. P(truthful-looking) is undefined."
        )
    col = classes.index(1)
    proba = pipeline.predict_proba(X)
    return np.asarray(proba[:, col], dtype=float)


# --- Reproducibility check -------------------------------------------------
REPRO_COLS = [
    "surface_lr_full_P_truthful",
    "surface_lr_cleaned_P_truthful",
    "embedding_lr_full_P_truthful",
    "embedding_lr_cleaned_P_truthful",
]


def _assert_existing_cols_reproduce(
    new_records: list[dict],
    prior_by_id: dict[int, dict],
    tol: float = REPRO_TOL,
) -> None:
    """For every slot that was present in the prior output, assert the four
    pre-existing columns reproduce to within `tol`. Fails hard on any
    slot/column that drifts."""
    drifts: list[str] = []
    for rec in new_records:
        sid = int(rec["id"])
        prior = prior_by_id.get(sid)
        if prior is None:
            continue
        for col in REPRO_COLS:
            if col not in prior:
                continue
            new_v = float(rec[col])
            old_v = float(prior[col])
            diff = abs(new_v - old_v)
            if diff > tol:
                drifts.append(
                    f"slot {sid:02d} / {col}: |new {new_v:.12f} - "
                    f"old {old_v:.12f}| = {diff:.3e} > {tol:.0e}"
                )
    if drifts:
        lines = "\n  ".join(drifts)
        raise RuntimeError(
            "Reproducibility check FAILED. Existing classifier columns "
            "drifted between runs. Stopping so a real regression does not "
            "slip through.\n  " + lines
        )
    if prior_by_id:
        n_checked = sum(
            1 for rec in new_records if int(rec["id"]) in prior_by_id
        )
        print(f"  Reproducibility: {len(REPRO_COLS)} existing columns x "
              f"{n_checked} slots all reproduce within {tol:.0e}.")
    else:
        print("  Reproducibility: no prior output found; nothing to check.")


def main() -> int:
    print("=" * 72)
    print("STEP 10 - score_singleton.py  (6-classifier edition)")
    print("=" * 72)

    gens = _load_generations()
    statements = [r["statement"] for r in gens]
    ids = [int(r["id"]) for r in gens]
    print(f"Loaded {len(gens)} generator records from {GEN_JSON}")

    prior_by_id = _load_prior_scores()
    if prior_by_id is not None:
        print(f"Prior scores file present ({len(prior_by_id)} records). "
              "Will enforce reproducibility on existing columns.")
    else:
        print("No prior scores file; this is the first 6-column run.")

    surf_full = _load_surface_pickle(SURFACE_FULL_PKL, "surface_lr_full")
    surf_clean = _load_surface_pickle(SURFACE_CLEAN_PKL, "surface_lr_cleaned")
    emb_full = _load_embed_pickle(EMBED_FULL_PKL, "embedding_lr_full")
    emb_clean = _load_embed_pickle(EMBED_CLEAN_PKL, "embedding_lr_cleaned")
    mbert_full = _load_embed_pickle(MBERT_FULL_PKL, "modernbert_lr_full")
    mbert_clean = _load_embed_pickle(MBERT_CLEAN_PKL, "modernbert_lr_cleaned")

    print(
        f"  surface_lr_full       cv_auc={surf_full['cv_auc_group5']:.4f}  "
        f"train_len_gap_mean={surf_full['train_len_gap_mean']:.6f}"
    )
    print(
        f"  surface_lr_cleaned    cv_auc={surf_clean['cv_auc_group5']:.4f}  "
        f"train_len_gap_mean={surf_clean['train_len_gap_mean']:.6f}"
    )
    print(f"  embedding_lr_full     cv_auc={emb_full['cv_auc_group5']:.4f}")
    print(f"  embedding_lr_cleaned  cv_auc={emb_clean['cv_auc_group5']:.4f}")
    print(f"  modernbert_lr_full    cv_auc={mbert_full['cv_auc_group5']:.4f}")
    print(f"  modernbert_lr_cleaned cv_auc={mbert_clean['cv_auc_group5']:.4f}")

    # --- Surface scoring ----------------------------------------------------
    print("\nBuilding surface feature matrix (full pipeline's len_gap mean) ...")
    X_surf_full = _build_surface_matrix(
        statements, surf_full["feature_cols"], surf_full["train_len_gap_mean"]
    )
    print("Building surface feature matrix (cleaned pipeline's len_gap mean) ...")
    X_surf_clean = _build_surface_matrix(
        statements, surf_clean["feature_cols"], surf_clean["train_len_gap_mean"]
    )
    if surf_full["feature_cols"] != surf_clean["feature_cols"]:
        raise RuntimeError(
            "feature_cols disagree between surface_lr_full and "
            "surface_lr_cleaned pickles."
        )
    print(f"  feature_cols = {surf_full['feature_cols']}")
    p_surf_full = _proba_class1(surf_full["pipeline"], X_surf_full)
    p_surf_clean = _proba_class1(surf_clean["pipeline"], X_surf_clean)

    # --- BGE scoring --------------------------------------------------------
    print("\nEncoding 20 statements with BGE-large-en-v1.5 ...")
    X_bge = _encode_bge(statements)
    print(f"  X_bge shape = {X_bge.shape}")
    p_emb_full = _proba_class1(emb_full["pipeline"], X_bge)
    p_emb_clean = _proba_class1(emb_clean["pipeline"], X_bge)

    # --- ModernBERT scoring -------------------------------------------------
    print("\nEncoding 20 statements with ModernBERT-base (CLS, no norm) ...")
    X_mbert = _encode_modernbert(statements)
    print(f"  X_modernbert shape = {X_mbert.shape}")
    p_mbert_full = _proba_class1(mbert_full["pipeline"], X_mbert)
    p_mbert_clean = _proba_class1(mbert_clean["pipeline"], X_mbert)

    # --- Pack records -------------------------------------------------------
    records: list[dict] = []
    for i, sid in enumerate(ids):
        records.append(
            {
                "id": int(sid),
                "surface_lr_full_P_truthful":    float(p_surf_full[i]),
                "surface_lr_cleaned_P_truthful": float(p_surf_clean[i]),
                "embedding_lr_full_P_truthful":  float(p_emb_full[i]),
                "embedding_lr_cleaned_P_truthful": float(p_emb_clean[i]),
                "modernbert_lr_full_P_truthful":  float(p_mbert_full[i]),
                "modernbert_lr_cleaned_P_truthful": float(p_mbert_clean[i]),
            }
        )
    records.sort(key=lambda r: r["id"])

    # --- Reproducibility assertion BEFORE writing the new JSON -------------
    print("\nReproducibility check on pre-existing columns "
          f"(tol={REPRO_TOL:.0e}) ...")
    _assert_existing_cols_reproduce(records, prior_by_id or {})

    # --- Write output -------------------------------------------------------
    STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(records)} scored records -> {OUT_JSON}")

    # --- Console summary ---------------------------------------------------
    topics = {int(r["id"]): r.get("topic", "?") for r in gens}
    print("\nPer-slot P(truthful-looking):")
    hdr = (f"  slot | topic           | surf_full | surf_clean "
           "| bge_full | bge_clean | mbert_full | mbert_clean")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in records:
        print(
            f"  {r['id']:>4d} | {topics[r['id']]:<15s} "
            f"| {r['surface_lr_full_P_truthful']:>9.4f} "
            f"| {r['surface_lr_cleaned_P_truthful']:>10.4f} "
            f"| {r['embedding_lr_full_P_truthful']:>8.4f} "
            f"| {r['embedding_lr_cleaned_P_truthful']:>9.4f} "
            f"| {r['modernbert_lr_full_P_truthful']:>10.4f} "
            f"| {r['modernbert_lr_cleaned_P_truthful']:>11.4f}"
        )

    arr_sf = np.array([r["surface_lr_full_P_truthful"] for r in records])
    arr_sc = np.array([r["surface_lr_cleaned_P_truthful"] for r in records])
    arr_ef = np.array([r["embedding_lr_full_P_truthful"] for r in records])
    arr_ec = np.array([r["embedding_lr_cleaned_P_truthful"] for r in records])
    arr_mf = np.array([r["modernbert_lr_full_P_truthful"] for r in records])
    arr_mc = np.array([r["modernbert_lr_cleaned_P_truthful"] for r in records])

    print("\nFamily comparison table:")
    print("  Classifier family   |  mean P_truthful |  fooling rate (full) "
          "|  mean(full - cleaned)")
    print("  " + "-" * 90)

    def _row(name: str, full: np.ndarray, clean: np.ndarray) -> None:
        m_full = float(full.mean())
        n_fool = int((full > 0.5).sum())
        gap = float(full.mean() - clean.mean())
        print(f"  {name:<19s} |  {m_full:>14.4f}  |  "
              f"{n_fool:>2d}/20 ({n_fool/20:.2f})  |  {gap:+.4f}")

    _row("surface_lr",      arr_sf, arr_sc)
    _row("embedding_lr(BGE)", arr_ef, arr_ec)
    _row("modernbert_lr",   arr_mf, arr_mc)

    # --- ModernBERT-specific counters --------------------------------------
    d_mbert = arr_mf - arr_mc
    n_mbert_pos = int((d_mbert > 0).sum())
    n_mbert_neg = int((d_mbert < 0).sum())
    n_mbert_tie = int((d_mbert == 0).sum())
    print(
        f"\n  ModernBERT fooling rate (full > 0.5): "
        f"{int((arr_mf > 0.5).sum())}/20"
    )
    print(
        f"  ModernBERT within-statement delta (full - cleaned): "
        f"pos={n_mbert_pos}  neg={n_mbert_neg}  tie={n_mbert_tie}  "
        f"mean={d_mbert.mean():+.4f}  min={d_mbert.min():+.4f}  "
        f"max={d_mbert.max():+.4f}"
    )

    # --- Low-P flags (< 0.3 on modernbert_full) ----------------------------
    print("\nSlots where modernbert_full < 0.3 (NOT fooled, interesting):")
    any_low = False
    for r in records:
        if r["modernbert_lr_full_P_truthful"] < 0.3:
            any_low = True
            print(
                f"  slot {r['id']:02d} [{topics[r['id']]}]: "
                f"modernbert_full={r['modernbert_lr_full_P_truthful']:.4f}  "
                f"modernbert_cleaned="
                f"{r['modernbert_lr_cleaned_P_truthful']:.4f}"
            )
    if not any_low:
        print("  (none)")

    # --- Wrong-direction flags (modernbert cleaned > full) -----------------
    print("\nSlots where modernbert_cleaned > modernbert_full "
          "(wrong direction):")
    any_wrong = False
    for r in records:
        if (r["modernbert_lr_cleaned_P_truthful"]
                > r["modernbert_lr_full_P_truthful"]):
            any_wrong = True
            d = (r["modernbert_lr_full_P_truthful"]
                 - r["modernbert_lr_cleaned_P_truthful"])
            print(f"  slot {r['id']:02d} [{topics[r['id']]}]: "
                  f"modernbert_full={r['modernbert_lr_full_P_truthful']:.4f}  "
                  f"modernbert_cleaned="
                  f"{r['modernbert_lr_cleaned_P_truthful']:.4f}  "
                  f"(Δ={d:+.4f})")
    if not any_wrong:
        print("  (none - all 20 slots have modernbert_cleaned "
              "<= modernbert_full)")

    print("\nStep 10 complete (6-classifier edition).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in score_singleton.py:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
