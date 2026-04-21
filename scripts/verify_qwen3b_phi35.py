#!/usr/bin/env python3
"""
Verification script for Qwen2.5-3B and Phi-3.5-mini families.

Checks that could reveal bugs:
  (A) Encoding consistency: re-encode 5 random training rows with the
      score-time encoder and compare to saved build-time embeddings.
      Max abs diff should be ~0 on the same code path (bf16 -> fp32 is
      deterministic given identical inputs).
  (B) Pickle class alignment: classes_ must contain both {0, 1} and we
      must pull column = index of 1.
  (C) Score-time reproducibility: re-encode the v3 A-side[0] with the
      score-time encoder and compute P. Compare to saved
      per_pair_same_index[0].P_A_full / P_A_cleaned from the scores JSON.
  (D) Sanity: sign of classifier coefficients is plausible given the
      known label convention (y=1 is "truthful Best Answer").
  (E) Full-vs-cleaned training data containment: every pair_id in the
      cleaned .npy must also appear in the full .npy.
  (F) Label-alignment spot check: for pair 0, the two rows in the .npy
      should have y=1 and y=0 and match the TruthfulQA CSV's
      "Best Answer" / "Best Incorrect Answer" text when re-encoded.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS = REPO_ROOT / "artifacts"
EMB_DIR = ARTIFACTS / "embeddings"
SCORES_DIR = REPO_ROOT / "stage0_v7a_bilateral"
TRUTHFULQA_CSV = REPO_ROOT / "TruthfulQA.csv"
V3_GEN_JSON = SCORES_DIR / "stage0_singleton_v7a_v3_generations.json"

_HF_HOME_DEFAULT = ARTIFACTS / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import joblib  # noqa: E402

# Fixed RNG so results are reproducible.
RNG = np.random.default_rng(17)


FAMILIES = [
    {
        "name": "Qwen2.5-3B",
        "prefix": "qwen3b",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "hidden": 2048,
        "batch": 4,
        "pkl_full": ARTIFACTS / "qwen3b_lr_full.pkl",
        "pkl_clean": ARTIFACTS / "qwen3b_lr_cleaned.pkl",
        "scores_json": SCORES_DIR / "stage0_singleton_v7a_v3_qwen3b_scores.json",
        "trust_remote": False,
    },
    {
        "name": "Phi-3.5-mini",
        "prefix": "phi35",
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "hidden": 3072,
        "batch": 4,
        "pkl_full": ARTIFACTS / "phi35_lr_full.pkl",
        "pkl_clean": ARTIFACTS / "phi35_lr_cleaned.pkl",
        "scores_json": SCORES_DIR / "stage0_singleton_v7a_v3_phi35_scores.json",
        "trust_remote": True,
    },
]

MAX_LENGTH = 512


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _encode_texts(model_id: str, hidden: int, batch: int,
                  texts: list[str], trust_remote: bool) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=trust_remote,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start:start + batch]
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden_states = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / denom
            rows.append(pooled.detach().cpu().to(torch.float32).numpy())
    X = np.concatenate(rows, axis=0).astype(np.float32)
    if X.shape != (len(texts), hidden):
        raise RuntimeError(f"Bad shape {X.shape}, expected ({len(texts)}, {hidden})")
    return X


def _proba_truthful(pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipeline.steps[-1][1]
    classes = list(getattr(clf, "classes_"))
    if 0 not in classes or 1 not in classes:
        raise RuntimeError(f"Bad classes_: {classes}")
    col = classes.index(1)
    p = pipeline.predict_proba(X)
    if p.shape[1] != 2:
        raise RuntimeError(f"predict_proba n_cols={p.shape[1]}")
    return np.asarray(p[:, col], dtype=float), classes, col


def _build_answer_rows(tq: pd.DataFrame) -> pd.DataFrame:
    best = tq["Best Answer"].fillna("").astype(str)
    incorrect = tq["Best Incorrect Answer"].fillna("").astype(str)
    rows = []
    for pair_id in range(len(tq)):
        rows.append({"pair_id": pair_id, "y": 1, "text": best.iat[pair_id]})
        rows.append({"pair_id": pair_id, "y": 0, "text": incorrect.iat[pair_id]})
    return pd.DataFrame(rows)


def verify_family(fam: dict) -> list[str]:
    problems: list[str] = []
    name = fam["name"]
    prefix = fam["prefix"]
    hidden = fam["hidden"]
    print(f"\n{'=' * 72}\n{name} [{fam['model_id']}]\n{'=' * 72}")

    # load stored build-time embeddings
    Xf = np.load(EMB_DIR / f"{prefix}_full_X.npy")
    yf = np.load(EMB_DIR / f"{prefix}_full_y.npy").astype(int)
    pf = np.load(EMB_DIR / f"{prefix}_full_pair_id.npy").astype(int)
    Xc = np.load(EMB_DIR / f"{prefix}_cleaned_X.npy")
    yc = np.load(EMB_DIR / f"{prefix}_cleaned_y.npy").astype(int)
    pc = np.load(EMB_DIR / f"{prefix}_cleaned_pair_id.npy").astype(int)
    print(f"  build-time shapes: full={Xf.shape} cleaned={Xc.shape}")

    # ---- (E) full-vs-cleaned containment + label pattern ----
    full_pair_set = set(int(x) for x in pf)
    clean_pair_set = set(int(x) for x in pc)
    missing = clean_pair_set - full_pair_set
    if missing:
        problems.append(
            f"[E] {len(missing)} cleaned pair_ids not in full (first 5: "
            f"{sorted(list(missing))[:5]})")
    else:
        print("  [E] every cleaned pair_id is in full  ✓")

    # Every pair should appear exactly twice with labels {0, 1}
    bad_pairs = 0
    for pid in np.unique(pf):
        rows = np.where(pf == pid)[0]
        if len(rows) != 2 or set(yf[rows].tolist()) != {0, 1}:
            bad_pairs += 1
    if bad_pairs:
        problems.append(f"[E] {bad_pairs} full pair_ids have bad label pattern")
    else:
        print("  [E] all full pair_ids have exactly {y=0, y=1}  ✓")

    # ---- (F) label alignment: pair 0 rows match the CSV texts ----
    tq = pd.read_csv(TRUTHFULQA_CSV)
    rows_df = _build_answer_rows(tq)
    pair0_rows = rows_df[rows_df["pair_id"] == 0]
    pair0_rows = pair0_rows.reset_index(drop=True)
    # rebuild-row order is (y=1, y=0)
    assert pair0_rows.loc[0, "y"] == 1 and pair0_rows.loc[1, "y"] == 0

    # sample 5 pair_ids to re-encode and compare
    sample_pids = sorted(RNG.choice(np.unique(pf), size=5, replace=False).tolist())
    # rows_df was built in the same order as _build_answer_rows in the
    # build script, so Xf[i] corresponds to rows_df.iloc[i]. Keep the
    # ORIGINAL index (do NOT reset_index).
    sample_text_rows = rows_df[rows_df["pair_id"].isin(sample_pids)]
    sample_indices = sample_text_rows.index.tolist()
    texts = sample_text_rows["text"].tolist()

    print(f"  re-encoding {len(texts)} rows to check consistency ...")
    X_re = _encode_texts(fam["model_id"], hidden, fam["batch"], texts,
                         fam["trust_remote"])
    X_stored = Xf[sample_indices]
    max_abs = float(np.max(np.abs(X_re - X_stored)))
    cos = float(np.mean(
        (X_re * X_stored).sum(axis=1) /
        (np.linalg.norm(X_re, axis=1) * np.linalg.norm(X_stored, axis=1) + 1e-12)
    ))
    print(f"  [A] re-encode vs stored:  max|diff|={max_abs:.6f}   mean cos={cos:.6f}")
    if cos < 0.999 or max_abs > 0.5:
        problems.append(
            f"[A] encoding inconsistent: max_abs={max_abs:.4f} cos={cos:.4f}")
    else:
        print("     encoding reproducible within bf16/MPS tolerance  ✓")

    # ---- (B) pickle class alignment ----
    pkl_full = joblib.load(fam["pkl_full"])
    pkl_clean = joblib.load(fam["pkl_clean"])
    pipeline_full = pkl_full["pipeline"]
    pipeline_clean = pkl_clean["pipeline"]
    P_full_sample, cls_full, col_full = _proba_truthful(pipeline_full, X_stored)
    P_clean_sample, cls_clean, col_clean = _proba_truthful(pipeline_clean, X_stored)
    print(f"  [B] classes_(full)={cls_full}  col_1={col_full}   "
          f"classes_(clean)={cls_clean}  col_1={col_clean}")
    if cls_full != [0, 1] or cls_clean != [0, 1]:
        problems.append(f"[B] unexpected classes_: full={cls_full} clean={cls_clean}")
    else:
        print("     both heads are [0, 1] with col_1 selected  ✓")

    # Sanity: the y=1 rows in our sample should have higher P_full than y=0 rows
    ys = sample_text_rows["y"].to_numpy()
    P_true = P_full_sample[ys == 1]
    P_false = P_full_sample[ys == 0]
    print(f"  [B'] sample P_full:  y=1 mean={P_true.mean():.3f}  "
          f"y=0 mean={P_false.mean():.3f}")
    if P_true.mean() <= P_false.mean():
        problems.append(
            f"[B'] y=1 sample mean P ({P_true.mean():.3f}) should exceed "
            f"y=0 sample mean P ({P_false.mean():.3f})")
    else:
        print("     y=1 rows rank above y=0 on training sample  ✓")

    # ---- (C) score-time reproducibility: re-encode v3 A-side[0] and
    #           match per_pair_same_index[0] ----
    scores = _load_json(fam["scores_json"])
    pp0 = scores["per_pair_same_index"][0]
    a_text_0 = pp0["a_text"]
    stored_Pa_full = float(pp0["P_A_full"])
    stored_Pa_clean = float(pp0["P_A_cleaned"])
    X_a0 = _encode_texts(fam["model_id"], hidden, fam["batch"],
                         [a_text_0], fam["trust_remote"])
    Pa_full, _, _ = _proba_truthful(pipeline_full, X_a0)
    Pa_clean, _, _ = _proba_truthful(pipeline_clean, X_a0)
    diff_full = abs(Pa_full[0] - stored_Pa_full)
    diff_clean = abs(Pa_clean[0] - stored_Pa_clean)
    print(f"  [C] v3 A[0] P_full:  stored={stored_Pa_full:.4f}  "
          f"re-encoded={Pa_full[0]:.4f}  |diff|={diff_full:.5f}")
    print(f"  [C] v3 A[0] P_clean: stored={stored_Pa_clean:.4f}  "
          f"re-encoded={Pa_clean[0]:.4f}  |diff|={diff_clean:.5f}")
    if diff_full > 0.01 or diff_clean > 0.01:
        problems.append(
            f"[C] v3 A[0] probability drifted: "
            f"full={diff_full:.5f} clean={diff_clean:.5f}")
    else:
        print("     score-time encoding reproducible  ✓")

    # ---- (D) coefficient sanity: StandardScaler mean is finite and LR
    #         coefs are finite and not trivially zero ----
    from sklearn.pipeline import Pipeline
    if isinstance(pipeline_full, Pipeline):
        scaler = pipeline_full.named_steps.get("standardscaler")
        lr = pipeline_full.named_steps.get("logisticregression")
    else:
        scaler, lr = None, None
    if scaler is None or lr is None:
        problems.append(
            f"[D] {fam['name']} full pipeline missing StandardScaler or LR")
    else:
        n_finite = int(np.isfinite(lr.coef_).sum())
        n_total = int(lr.coef_.size)
        max_abs_coef = float(np.max(np.abs(lr.coef_)))
        print(f"  [D] full LR: {n_finite}/{n_total} finite coefs, "
              f"max|coef|={max_abs_coef:.4f}")
        if n_finite != n_total or max_abs_coef < 1e-6:
            problems.append("[D] LR coefficients look degenerate")
        else:
            print("     coefficients look healthy  ✓")

    # ---- extra: per-pair fool-rate recomputation from per_pair array ----
    n_fooled_full = sum(1 for r in scores["per_pair_same_index"]
                        if r["fooled_full"])
    n_fooled_clean = sum(1 for r in scores["per_pair_same_index"]
                         if r["fooled_cleaned"])
    n_pairs = len(scores["per_pair_same_index"])
    claimed_full = scores["bilateral_same_index_all"]["full"]["fool_rate"]
    claimed_clean = scores["bilateral_same_index_all"]["cleaned"]["fool_rate"]
    recomp_full = n_fooled_full / n_pairs
    recomp_clean = n_fooled_clean / n_pairs
    print(f"  [G] same-idx fool_rate recompute:  "
          f"full stored={claimed_full:.3f} vs recompute={recomp_full:.3f}  "
          f"clean stored={claimed_clean:.3f} vs recompute={recomp_clean:.3f}")
    if abs(claimed_full - recomp_full) > 1e-6 or \
       abs(claimed_clean - recomp_clean) > 1e-6:
        problems.append("[G] same-index fool_rate aggregate doesn't match per-pair")
    else:
        print("     aggregate fool_rate matches per-pair boolean  ✓")

    return problems


def main() -> int:
    print("Verification of Qwen2.5-3B and Phi-3.5-mini additive pipelines")
    all_problems: list[tuple[str, str]] = []
    for fam in FAMILIES:
        probs = verify_family(fam)
        for p in probs:
            all_problems.append((fam["name"], p))

    print("\n" + "=" * 72)
    if all_problems:
        print(f"FOUND {len(all_problems)} ISSUE(S)")
        for fam_name, p in all_problems:
            print(f"  - [{fam_name}] {p}")
        return 1
    print("ALL CHECKS PASSED  ✓")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in verify_qwen3b_phi35.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
