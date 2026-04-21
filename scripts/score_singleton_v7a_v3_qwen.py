#!/usr/bin/env python3
"""
Score v7(a) v3 A-side with the NEW Qwen2.5-0.5B-Instruct LR family, and
pair against the §5.1 B-side (scored fresh with Qwen, since the existing
stage0/stage0_classifier_scores.json has no Qwen column).

Does NOT re-score v3 against the 6 canonical pickles - that's already
done in stage0_v7a_bilateral/stage0_singleton_v7a_v3_classifier_scores.json.

Outputs:
    stage0_v7a_bilateral/stage0_singleton_v7a_v3_qwen_scores.json

Headline:
    - v3 A-side unilateral with Qwen full/cleaned (mean_P, P>0.5 rate)
    - v3 A-side x §5.1 B-side bilateral with Qwen full/cleaned
      (same-index n=20 and all-pairs n=400)
    - fool-drop (full - cleaned) for each variant
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

OUT_DIR = REPO_ROOT / "stage0_v7a_bilateral"
V3_GEN_JSON = OUT_DIR / "stage0_singleton_v7a_v3_generations.json"
V3_JUDGE_JSON = OUT_DIR / "stage0_singleton_v7a_v3_judge.json"
OUT_JSON = OUT_DIR / "stage0_singleton_v7a_v3_qwen_scores.json"

SIDE_B_GEN = REPO_ROOT / "stage0" / "stage0_generations.json"

ARTIFACTS = REPO_ROOT / "artifacts"
QWEN_PKL_FULL = ARTIFACTS / "qwen_lr_full.pkl"
QWEN_PKL_CLEAN = ARTIFACTS / "qwen_lr_cleaned.pkl"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_DIM_EXPECTED = 896
MAX_LENGTH = 512
BATCH_SIZE = 16

_HF_HOME_DEFAULT = ARTIFACTS / "hf_cache"
os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_HOME_DEFAULT / "hub"))
_HF_HOME_DEFAULT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing pickle {path}")
    art = joblib.load(path)
    if "pipeline" not in art:
        raise RuntimeError(f"Pickle has no 'pipeline' key: {path}")
    return art


def _proba_truthful(pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipeline.steps[-1][1]
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 not in classes:
        raise RuntimeError(f"Classifier missing class 1; classes={classes}")
    col = classes.index(1)
    return np.asarray(pipeline.predict_proba(X)[:, col], dtype=float)


def _encode_qwen(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / denom
            rows.append(pooled.detach().cpu().to(torch.float32).numpy())
    X = np.concatenate(rows, axis=0).astype(np.float32)
    if X.shape != (len(texts), EMBED_DIM_EXPECTED):
        raise RuntimeError(f"Unexpected encoded shape: {X.shape}")
    return X


def _uni(pa: np.ndarray, mask: np.ndarray | None = None) -> dict:
    arr = pa if mask is None else pa[mask]
    n = int(arr.shape[0])
    if n == 0:
        return {"n": 0, "mean_P": float("nan"), "robust_rate": float("nan")}
    return {
        "n": n,
        "mean_P": float(arr.mean()),
        "robust_rate": float((arr > 0.5).mean()),
    }


def _bi_same(pa: np.ndarray, pb: np.ndarray,
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


def _bi_all(pa: np.ndarray, pb: np.ndarray,
            mask_a: np.ndarray | None = None) -> dict:
    a = pa if mask_a is None else pa[mask_a]
    if a.shape[0] == 0 or pb.shape[0] == 0:
        return {"n_pairs": 0, "fool_rate": float("nan")}
    grid = pb[None, :] > a[:, None]
    return {
        "n_pairs": int(a.shape[0] * pb.shape[0]),
        "fool_rate": float(grid.mean()),
    }


def main() -> int:
    print("=" * 72)
    print("score_singleton_v7a_v3_qwen.py  (Qwen2.5-0.5B-Instruct family)")
    print("=" * 72)

    gens = sorted(_load_json(V3_GEN_JSON), key=lambda r: int(r["id"]))
    a_ids = [int(r["id"]) for r in gens]
    a_topics = [str(r["topic"]) for r in gens]
    a_texts = [str(r["statement"]) for r in gens]

    if V3_JUDGE_JSON.exists():
        jrec = {int(r["id"]): r for r in _load_json(V3_JUDGE_JSON)}
        judge_mask = np.array(
            [bool(jrec.get(i, {}).get("judge_agrees_true", False))
             for i in a_ids], dtype=bool)
        print(f"Loaded v3 judge: {sum(judge_mask)}/{len(a_ids)} TRUE")
    else:
        judge_mask = np.ones(len(a_ids), dtype=bool)
        print("No v3 judge file; treating all A-sides as judge-passed.")

    b_gens = sorted(_load_json(SIDE_B_GEN), key=lambda r: int(r["id"]))
    if len(b_gens) != 20:
        raise RuntimeError(f"Expected 20 §5.1 B-side rows, got {len(b_gens)}")
    b_ids = [int(r["id"]) for r in b_gens]
    b_topics = [str(r["topic"]) for r in b_gens]
    b_texts = [str(r["statement"]) for r in b_gens]

    print("\nLoading Qwen pickles ...")
    art_full = _load_pickle(QWEN_PKL_FULL)
    art_clean = _load_pickle(QWEN_PKL_CLEAN)
    print(f"  qwen_lr_full    CV AUC: {art_full.get('cv_auc_group5'):.4f}")
    print(f"  qwen_lr_cleaned CV AUC: {art_clean.get('cv_auc_group5'):.4f}")

    print("\nEncoding A-side (v3) with Qwen ...")
    Xa = _encode_qwen(a_texts)
    print(f"  Xa shape = {Xa.shape}")

    print("Encoding B-side (§5.1) with Qwen ...")
    Xb = _encode_qwen(b_texts)
    print(f"  Xb shape = {Xb.shape}")

    PA = {
        "full":    _proba_truthful(art_full["pipeline"], Xa),
        "cleaned": _proba_truthful(art_clean["pipeline"], Xa),
    }
    PB = {
        "full":    _proba_truthful(art_full["pipeline"], Xb),
        "cleaned": _proba_truthful(art_clean["pipeline"], Xb),
    }

    uni_all = {s: _uni(PA[s], None) for s in ("full", "cleaned")}
    uni_judge = {s: _uni(PA[s], judge_mask) for s in ("full", "cleaned")}
    bi_same_all = {s: _bi_same(PA[s], PB[s], None) for s in ("full", "cleaned")}
    bi_same_judge = {
        s: _bi_same(PA[s], PB[s], judge_mask) for s in ("full", "cleaned")}
    bi_all_all = {s: _bi_all(PA[s], PB[s], None) for s in ("full", "cleaned")}
    bi_all_judge = {
        s: _bi_all(PA[s], PB[s], judge_mask) for s in ("full", "cleaned")}

    n_a = len(a_ids)
    n_j = int(judge_mask.sum())

    def _print_uni(title, agg, n):
        print("\n" + "=" * 80)
        print(f"QWEN UNILATERAL A-SIDE (n={n})  {title}")
        print("=" * 80)
        print(f"  {'split':<8s} {'mean P(truthful)':>18s} "
              f"{'P>0.5 rate':>12s}  {'cleaned - full':>16s}")
        print("  " + "-" * 70)
        mf = agg["full"]["mean_P"]; mc = agg["cleaned"]["mean_P"]
        rf = agg["full"]["robust_rate"]; rc = agg["cleaned"]["robust_rate"]
        shift = mc - mf
        print(f"  {'full':<8s} {mf:>18.4f} {rf:>11.3f}")
        print(f"  {'cleaned':<8s} {mc:>18.4f} {rc:>11.3f}   "
              f"{shift:>+16.4f}")

    def _print_bi_same(title, agg, n):
        print("\n" + "=" * 80)
        print(f"QWEN BILATERAL SAME-INDEX (n={n})  {title}")
        print("=" * 80)
        print(f"  {'split':<8s} {'fool_rate':>12s} "
              f"{'mean gap (B-A)':>15s}  {'fool_drop':>10s}")
        print("  " + "-" * 60)
        ff = agg["full"]["fool_rate"]; fc = agg["cleaned"]["fool_rate"]
        gf = agg["full"]["mean_gap_B_minus_A"]
        gc = agg["cleaned"]["mean_gap_B_minus_A"]
        drop = ff - fc
        print(f"  {'full':<8s} {ff:>12.4f} {gf:>+15.4f}")
        print(f"  {'cleaned':<8s} {fc:>12.4f} {gc:>+15.4f}  {drop:>+10.4f}")

    def _print_bi_all(title, agg, na, nb):
        print("\n" + "=" * 80)
        print(f"QWEN BILATERAL ALL-PAIRS ({na}x{nb}={na*nb})  {title}")
        print("=" * 80)
        ff = agg["full"]["fool_rate"]; fc = agg["cleaned"]["fool_rate"]
        drop = ff - fc
        print(f"  full fool    {ff:.4f}")
        print(f"  clean fool   {fc:.4f}")
        print(f"  fool_drop    {drop:+.4f}")

    _print_uni("all A-sides", uni_all, n_a)
    if n_j != n_a:
        _print_uni("judge-passed", uni_judge, n_j)
    _print_bi_same("all A-sides x §5.1 B-side (same idx)", bi_same_all, n_a)
    if n_j != n_a:
        _print_bi_same("judge-passed A x B (same idx)", bi_same_judge, n_j)
    _print_bi_all("all A x all B", bi_all_all, n_a, len(b_ids))
    if n_j != n_a:
        _print_bi_all("judge-passed A x all B", bi_all_judge, n_j, len(b_ids))

    per_pair = []
    for i, sid in enumerate(a_ids):
        per_pair.append({
            "a_id": int(sid), "a_topic": a_topics[i], "a_text": a_texts[i],
            "a_judge_agrees_true": bool(judge_mask[i]),
            "b_id": int(b_ids[i]), "b_topic": b_topics[i],
            "b_text": b_texts[i],
            "P_A_full": float(PA["full"][i]),
            "P_A_cleaned": float(PA["cleaned"][i]),
            "P_B_full": float(PB["full"][i]),
            "P_B_cleaned": float(PB["cleaned"][i]),
            "fooled_full": bool(PB["full"][i] > PA["full"][i]),
            "fooled_cleaned": bool(PB["cleaned"][i] > PA["cleaned"][i]),
        })

    payload = {
        "family": "qwen",
        "model_id": MODEL_ID,
        "embed_dim": EMBED_DIM_EXPECTED,
        "a_ids": a_ids, "b_ids": b_ids,
        "n_a": n_a, "n_b": len(b_ids),
        "n_a_judge_agrees_true": n_j,
        "qwen_lr_full_cv_auc": float(art_full.get("cv_auc_group5")),
        "qwen_lr_cleaned_cv_auc": float(art_clean.get("cv_auc_group5")),
        "unilateral_all": uni_all,
        "unilateral_judge": uni_judge,
        "bilateral_same_index_all": bi_same_all,
        "bilateral_same_index_judge": bi_same_judge,
        "bilateral_all_pairs_all": bi_all_all,
        "bilateral_all_pairs_judge": bi_all_judge,
        "per_pair_same_index": per_pair,
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
        print("NON-RECOVERABLE FAILURE in score_singleton_v7a_v3_qwen.py:",
              file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
