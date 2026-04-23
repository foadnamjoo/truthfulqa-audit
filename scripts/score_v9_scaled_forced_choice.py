#!/usr/bin/env python3
"""
v9 SCALED same-question forced-choice aggregator (n=79).

Reads:
  - stage0_v9_same_question/v9_scaled_scores_light.json     (3 light fams)
  - stage0_v9_same_question/v9_scaled_<fam>_scores.json     (4 heavy fams)
  - stage0_v9_same_question/v9_scaled_judge.json            (pair_passes)

Per family × variant reports:
  pair_accuracy, 95% Wilson CI, binomial p-value vs chance (0.5).
Per family also: McNemar p-value for full vs cleaned on picks.

Output: stage0_v9_same_question/v9_scaled_forced_choice_results.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V9_DIR    = REPO_ROOT / "stage0_v9_same_question"
LIGHT_SCORES = V9_DIR / "v9_scaled_scores_light.json"
OUT_JSON  = V9_DIR / "v9_scaled_forced_choice_results.json"

LIGHT = ["surface_lr", "BGE-large", "ModernBERT-base"]
HEAVY_FILES = {
    "Qwen2.5-0.5B": V9_DIR / "v9_scaled_qwen_scores.json",
    "Qwen2.5-1.5B": V9_DIR / "v9_scaled_qwen15b_scores.json",
    "SmolLM2-1.7B": V9_DIR / "v9_scaled_smollm2_scores.json",
    "Qwen2.5-3B":   V9_DIR / "v9_scaled_qwen3b_scores.json",
    "Llama-3.2-3B": V9_DIR / "v9_scaled_llama32_3b_scores.json",
    "Phi-3.5-mini": V9_DIR / "v9_scaled_phi35_scores.json",
}
FAMILY_ORDER = LIGHT + list(HEAVY_FILES.keys())


# --- stats helpers (stdlib only) -------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z*z / n
    centre = (p + z*z / (2*n)) / denom
    half   = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_binom(n: int, k: int) -> float:
    return (math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1))


def binom_pmf(k: int, n: int, p: float = 0.5) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return math.exp(_log_binom(n, k) + k*math.log(p) + (n-k)*math.log(1-p))


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value (doubled min-tail)."""
    if n == 0:
        return 1.0
    mu = n * p
    # Sum of PMFs as extreme or more extreme than k, on the near side.
    if k <= mu:
        tail = sum(binom_pmf(i, n, p) for i in range(0, k + 1))
    else:
        tail = sum(binom_pmf(i, n, p) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact McNemar on (b, c) discordant pair counts.  H0: p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return binom_two_sided_p(k, n, 0.5)


def _metrics(pA, pB, n_total: int) -> dict:
    correct = sum(1 for a, b in zip(pA, pB) if a > b)
    ties    = sum(1 for a, b in zip(pA, pB) if a == b)
    p = correct / n_total
    lo, hi = wilson_ci(correct, n_total)
    p_vs_chance = binom_two_sided_p(correct, n_total, 0.5)
    return {
        "n": n_total,
        "pair_correct_count": correct,
        "ties": ties,
        "pair_accuracy": p,
        "wilson95_low":   lo,
        "wilson95_high":  hi,
        "binom_p_vs_chance": p_vs_chance,
        "mean_P_gap_A_minus_B": sum(a - b for a, b in zip(pA, pB)) / n_total if n_total else 0.0,
    }


def _mcnemar_full_vs_cleaned(pA_f, pB_f, pA_c, pB_c) -> dict:
    # Pair-wise classifier picks under full and cleaned. A=1 if P_A > P_B.
    n = len(pA_f)
    pick_f = [1 if a > b else 0 for a, b in zip(pA_f, pB_f)]
    pick_c = [1 if a > b else 0 for a, b in zip(pA_c, pB_c)]
    # Only discordant picks matter.
    b = sum(1 for f, c in zip(pick_f, pick_c) if f == 0 and c == 1)   # cleaned flipped to correct
    c = sum(1 for f, c in zip(pick_f, pick_c) if f == 1 and c == 0)   # cleaned flipped to wrong
    return {
        "n": n,
        "flips_full_wrong_to_cleaned_right": b,
        "flips_full_right_to_cleaned_wrong": c,
        "mcnemar_p": mcnemar_exact_p(b, c),
        "net_flip":  b - c,
    }


def _block(pA_f, pB_f, pA_c, pB_c, n_total) -> dict:
    full  = _metrics(pA_f, pB_f, n_total)
    clean = _metrics(pA_c, pB_c, n_total)
    return {
        "full": full, "cleaned": clean,
        "drop": full["pair_accuracy"] - clean["pair_accuracy"],
        "gap_drop": full["mean_P_gap_A_minus_B"] - clean["mean_P_gap_A_minus_B"],
        "mcnemar_full_vs_cleaned": _mcnemar_full_vs_cleaned(pA_f, pB_f, pA_c, pB_c),
    }


def _light_probs(per_pair, fam: str, pids: list[int]) -> tuple:
    by_pid = {r["pair_id"]: r for r in per_pair}
    f_pA = [by_pid[p]["scores"][f"{fam}_full"]["P_A"]    for p in pids]
    f_pB = [by_pid[p]["scores"][f"{fam}_full"]["P_B"]    for p in pids]
    c_pA = [by_pid[p]["scores"][f"{fam}_cleaned"]["P_A"] for p in pids]
    c_pB = [by_pid[p]["scores"][f"{fam}_cleaned"]["P_B"] for p in pids]
    return f_pA, f_pB, c_pA, c_pB


def _heavy_probs(rows_by_pid, pids: list[int]) -> tuple:
    f_pA = [rows_by_pid[p]["P_A_truthful_full"]    for p in pids]
    f_pB = [rows_by_pid[p]["P_B_truthful_full"]    for p in pids]
    c_pA = [rows_by_pid[p]["P_A_truthful_cleaned"] for p in pids]
    c_pB = [rows_by_pid[p]["P_B_truthful_cleaned"] for p in pids]
    return f_pA, f_pB, c_pA, c_pB


def main() -> int:
    light = json.load(open(LIGHT_SCORES))
    per_pair = light["per_pair"]
    all_pids = sorted(int(r["pair_id"]) for r in per_pair)
    n = len(all_pids)
    print("=" * 110)
    print(f"v9 SCALED SAME-QUESTION FORCED-CHOICE — {len(FAMILY_ORDER)} families, n={n}")
    print(f"  light source: {LIGHT_SCORES.relative_to(REPO_ROOT)}")
    print("=" * 110)

    results = {"n": n, "pair_ids": all_pids, "families": {}}

    for fam in LIGHT:
        fp, fb, cp, cb = _light_probs(per_pair, fam, all_pids)
        results["families"][fam] = {"status": "ok", **_block(fp, fb, cp, cb, n)}

    for fam, path in HEAVY_FILES.items():
        if not path.exists():
            results["families"][fam] = {"status": "missing",
                                        "path": str(path.relative_to(REPO_ROOT))}
            continue
        rows = json.load(open(path))
        by_pid = {int(r["pair_id"]): r for r in rows}
        missing = [p for p in all_pids if p not in by_pid]
        if missing:
            results["families"][fam] = {"status": "incomplete",
                                        "missing": missing[:5],
                                        "path": str(path.relative_to(REPO_ROOT))}
            continue
        fp, fb, cp, cb = _heavy_probs(by_pid, all_pids)
        results["families"][fam] = {"status": "ok",
                                    "path": str(path.relative_to(REPO_ROOT)),
                                    **_block(fp, fb, cp, cb, n)}

    def _sig(p):
        if p is None: return ""
        if p < 0.001:  return "***"
        if p < 0.01:   return "**"
        if p < 0.05:   return "*"
        if p < 0.10:   return "."
        return ""

    print(f"\n{'Family':<17} {'full_acc (95% CI)':>22} {'p_chance':>9}   "
          f"{'cleaned_acc (95% CI)':>24} {'p_chance':>9}   "
          f"{'drop':>7}  {'McNemar p':>11}  {'flips +/-':>10}")
    print("-" * 124)
    for fam in FAMILY_ORDER:
        b = results["families"].get(fam, {})
        if b.get("status") != "ok":
            print(f"{fam:<17}  MISSING: {b.get('path')}")
            continue
        f = b["full"]; c = b["cleaned"]; m = b["mcnemar_full_vs_cleaned"]
        full_str  = (f"{f['pair_accuracy']:.3f} "
                     f"[{f['wilson95_low']:.2f},{f['wilson95_high']:.2f}]")
        clean_str = (f"{c['pair_accuracy']:.3f} "
                     f"[{c['wilson95_low']:.2f},{c['wilson95_high']:.2f}]")
        pf = f"{f['binom_p_vs_chance']:.3f}{_sig(f['binom_p_vs_chance'])}"
        pc = f"{c['binom_p_vs_chance']:.3f}{_sig(c['binom_p_vs_chance'])}"
        pm = f"{m['mcnemar_p']:.3f}{_sig(m['mcnemar_p'])}"
        flips = (f"+{m['flips_full_wrong_to_cleaned_right']}/"
                 f"-{m['flips_full_right_to_cleaned_wrong']}")
        print(f"{fam:<17} {full_str:>22} {pf:>9}   "
              f"{clean_str:>24} {pc:>9}   "
              f"{-b['drop']:>+7.3f}  {pm:>11}  {flips:>10}")

    V9_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print("\nColumn notes:")
    print("  p_chance     : two-sided exact binomial p-value vs 0.5")
    print("  drop column  : cleaned_acc - full_acc  (positive = cleaning HELPS)")
    print("  McNemar p    : exact test for full vs cleaned picks on same pairs")
    print("  flips +/-    : +k = cleaning flipped k wrong -> right;  -k = right -> wrong")
    print("  Sig: *** p<.001  ** p<.01  * p<.05  . p<.10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
