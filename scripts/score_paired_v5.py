#!/usr/bin/env python3
"""
v5 paired-probe scoring over six existing Stage-0 classifier pickles.
No retraining.
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

GEN_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations_v5.json"
JUDGE_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_judge_v5.json"
OUT_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_classifier_scores_v5.json"

ARTIFACTS = REPO_ROOT / "artifacts"
PICKLE_PATHS: dict[tuple[str, str], Path] = {
    ("surface_lr", "full"): ARTIFACTS / "surface_lr_full.pkl",
    ("surface_lr", "cleaned"): ARTIFACTS / "surface_lr_cleaned.pkl",
    ("BGE-large", "full"): ARTIFACTS / "embedding_lr_full.pkl",
    ("BGE-large", "cleaned"): ARTIFACTS / "embedding_lr_cleaned.pkl",
    ("ModernBERT-base", "full"): ARTIFACTS / "modernbert_lr_full.pkl",
    ("ModernBERT-base", "cleaned"): ARTIFACTS / "modernbert_lr_cleaned.pkl",
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
STRATEGIES = ["negation_opener", "hedging", "authority"]


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
    return sorted(data, key=lambda r: int(r["pair_id"]))


def _load_judge() -> dict[int, dict]:
    if not JUDGE_JSON.exists():
        return {}
    with open(JUDGE_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {int(r["pair_id"]): r for r in d}


def _proba_truthful(pipeline, X: np.ndarray) -> np.ndarray:
    clf = pipeline.steps[-1][1]
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 not in classes:
        raise RuntimeError(f"Classifier missing class 1; classes={classes}")
    col = classes.index(1)
    return np.asarray(pipeline.predict_proba(X)[:, col], dtype=float)


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
    return np.asarray(Xa, dtype=float), np.asarray(Xb, dtype=float)


def _encode_bge(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_MODEL_ID)
    X = model.encode(
        texts,
        batch_size=BGE_BATCH,
        normalize_embeddings=BGE_NORMALIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
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


def _aggregate(pa: np.ndarray, pb: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if mask is not None:
        pa = pa[mask]
        pb = pb[mask]
    n = int(pa.shape[0])
    if n == 0:
        return {
            "pair_accuracy": float("nan"),
            "n_correct": 0,
            "n_total": 0,
            "mean_P_A": float("nan"),
            "mean_P_B": float("nan"),
            "mean_gap": float("nan"),
        }
    picked_A = (pa > pb).astype(int)
    return {
        "pair_accuracy": float(picked_A.mean()),
        "n_correct": int(picked_A.sum()),
        "n_total": n,
        "mean_P_A": float(pa.mean()),
        "mean_P_B": float(pb.mean()),
        "mean_gap": float((pa - pb).mean()),
    }


def main() -> int:
    global GEN_JSON, JUDGE_JSON, OUT_JSON
    p = argparse.ArgumentParser()
    p.add_argument("--gen-json", type=str, default=None)
    p.add_argument("--judge-json", type=str, default=None)
    p.add_argument("--out-json", type=str, default=None)
    args = p.parse_args()
    if args.gen_json:
        GEN_JSON = Path(args.gen_json).resolve()
    if args.judge_json:
        JUDGE_JSON = Path(args.judge_json).resolve()
    if args.out_json:
        OUT_JSON = Path(args.out_json).resolve()

    print("=" * 72)
    print("score_paired_v5.py")
    print("=" * 72)

    pairs = _load_pairs()
    judges = _load_judge()
    n = len(pairs)
    pids = [int(p["pair_id"]) for p in pairs]
    a_texts = [str(p["a_side"]) for p in pairs]
    b_texts = [str(p["b_side"]) for p in pairs]
    strategies = [str(p.get("b_cue_strategy", "unknown")) for p in pairs]
    pair_passes_mask = np.array(
        [bool(judges.get(pid, {}).get("pair_passes", False)) for pid in pids],
        dtype=bool,
    )
    n_pass = int(pair_passes_mask.sum())
    print(f"Loaded {n} pairs; judge pair_passes={n_pass}/{n}")

    pickles: dict[tuple[str, str], dict] = {}
    for (fam, split), path in PICKLE_PATHS.items():
        pickles[(fam, split)] = _load_pickle(f"{fam}/{split}", path)
        print(f"  {fam:<16s} {split:<8s} <- {path.relative_to(REPO_ROOT)}")
    surf_cols = pickles[("surface_lr", "full")]["feature_cols"]

    Xa_surf, Xb_surf = _build_surface10(a_texts, b_texts, surf_cols)
    X_bge = _encode_bge(a_texts + b_texts)
    Xa_bge, Xb_bge = X_bge[:n], X_bge[n:]
    X_mb = _encode_modernbert(a_texts + b_texts)
    Xa_mb, Xb_mb = X_mb[:n], X_mb[n:]
    feat_by_family = {
        "surface_lr": (Xa_surf, Xb_surf),
        "BGE-large": (Xa_bge, Xb_bge),
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

    per_pair: list[dict] = []
    for i, pid in enumerate(pids):
        rec = {
            "pair_id": int(pid),
            "b_cue_strategy": strategies[i],
            "a_side_text": a_texts[i],
            "b_side_text": b_texts[i],
            "judge_pair_passes": bool(pair_passes_mask[i]),
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

    aggregates_all: dict[str, dict] = {}
    aggregates_pass: dict[str, dict] = {}
    for fam in FAMILIES:
        for split in SPLITS:
            key = f"{fam}_{split}"
            pa = P[(fam, split)]["A"]
            pb = P[(fam, split)]["B"]
            aggregates_all[key] = _aggregate(pa, pb, None)
            aggregates_pass[key] = _aggregate(pa, pb, pair_passes_mask)

    cleaning_effect_all: dict[str, dict] = {}
    cleaning_effect_pass: dict[str, dict] = {}
    for fam in FAMILIES:
        for src, dst in [(aggregates_all, cleaning_effect_all),
                         (aggregates_pass, cleaning_effect_pass)]:
            full_acc = src[f"{fam}_full"]["pair_accuracy"]
            clean_acc = src[f"{fam}_cleaned"]["pair_accuracy"]
            dst[fam] = {
                "full_acc": full_acc,
                "cleaned_acc": clean_acc,
                "delta": (clean_acc - full_acc)
                if (full_acc == full_acc and clean_acc == clean_acc) else float("nan"),
            }

    strategy_mask = {
        s: np.array([x == s for x in strategies], dtype=bool) for s in STRATEGIES
    }
    per_strategy_pass: dict[str, dict] = {}
    for s in STRATEGIES:
        mask_s = strategy_mask[s] & pair_passes_mask
        per_strategy_pass[s] = {"n_pairs": int(mask_s.sum()), "families": {}}
        for fam in FAMILIES:
            full = _aggregate(P[(fam, "full")]["A"], P[(fam, "full")]["B"], mask_s)
            clean = _aggregate(P[(fam, "cleaned")]["A"], P[(fam, "cleaned")]["B"], mask_s)
            delta = clean["pair_accuracy"] - full["pair_accuracy"] \
                if (full["pair_accuracy"] == full["pair_accuracy"]
                    and clean["pair_accuracy"] == clean["pair_accuracy"]) else float("nan")
            per_strategy_pass[s]["families"][fam] = {
                "full": full,
                "cleaned": clean,
                "delta": delta,
            }

    def _print_headline(title: str, agg: dict, n_used: int) -> None:
        print("\n" + "=" * 88)
        print(f"{title} (n={n_used})")
        print("=" * 88)
        print(f"  {'Classifier':<17s} {'TrainedOn':<10s} {'Pair acc':>12s} {'mean gap':>10s}")
        print("  " + "-" * 60)
        for fam in FAMILIES:
            for split in SPLITS:
                a = agg[f"{fam}_{split}"]
                print(f"  {fam:<17s} {split:<10s} {a['n_correct']:>4d}/{a['n_total']:<3d} ({a['pair_accuracy']:.3f}) {a['mean_gap']:>+10.4f}")

    _print_headline("V5 PAIRED PROBE RESULTS - ALL PAIRS", aggregates_all, n)
    _print_headline("V5 PAIRED PROBE RESULTS - JUDGE-PASSED SUBSET", aggregates_pass, n_pass)

    print("\nPER-STRATEGY BREAKDOWN (judge-passed only):")
    print(f"  {'Strategy':<17s} {'surf_lr delta':>14s} {'BGE delta':>12s} {'MBERT delta':>12s}")
    print("  " + "-" * 62)
    for s in STRATEGIES:
        d_surf = per_strategy_pass[s]["families"]["surface_lr"]["delta"]
        d_bge = per_strategy_pass[s]["families"]["BGE-large"]["delta"]
        d_mb = per_strategy_pass[s]["families"]["ModernBERT-base"]["delta"]
        print(f"  {s:<17s} {d_surf:>+14.4f} {d_bge:>+12.4f} {d_mb:>+12.4f}")

    print("\nCLEANING EFFECT (all 20):")
    for fam in FAMILIES:
        print(f"  {fam:<17s} delta={cleaning_effect_all[fam]['delta']:+.4f}")

    payload = {
        "v5_pair_ids": pids,
        "n_pairs": int(n),
        "n_pair_passes": int(n_pass),
        "generation_source": str(GEN_JSON.relative_to(REPO_ROOT)),
        "judge_source": str(JUDGE_JSON.relative_to(REPO_ROOT)),
        "classifier_pickles": {
            f"{fam}_{split}": str(path.relative_to(REPO_ROOT))
            for (fam, split), path in PICKLE_PATHS.items()
        },
        "per_pair": per_pair,
        "aggregates_all": aggregates_all,
        "aggregates_pair_passes": aggregates_pass,
        "cleaning_effect_all": cleaning_effect_all,
        "cleaning_effect_pair_passes": cleaning_effect_pass,
        "per_strategy_pair_passes_only": per_strategy_pass,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("NON-RECOVERABLE FAILURE in score_paired_v5.py:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
