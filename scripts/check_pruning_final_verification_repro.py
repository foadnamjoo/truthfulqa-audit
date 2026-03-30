#!/usr/bin/env python3
"""
Re-run locked pruning final verification and check exact reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "truthfulqa_pruning_final_verification"
    lock_path = out_dir / "LOCK_CONFIG.json"
    if not lock_path.exists():
        print(f"Missing lock config: {lock_path}", file=sys.stderr)
        return 1
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required_lock_keys = [
        "n_seeds",
        "base_seed",
        "audit_profile",
        "min_keep",
        "target_audit_auc",
        "max_drop_fraction",
        "holdout_fraction",
        "w_heldout",
        "w_size",
        "w_imbalance",
        "sha256",
    ]
    missing_keys = [k for k in required_lock_keys if k not in lock]
    if missing_keys:
        print(
            f"LOCK_CONFIG missing required keys: {', '.join(missing_keys)}",
            file=sys.stderr,
        )
        return 1

    cmd = [
        sys.executable,
        str((root / "scripts" / "truthfulqa_pruning_final_verification.py").resolve()),
        "--n-seeds",
        str(lock["n_seeds"]),
        "--base-seed",
        str(lock["base_seed"]),
        "--audit-profile",
        str(lock["audit_profile"]),
        "--min-keep",
        str(lock["min_keep"]),
        "--target-audit-auc",
        str(lock["target_audit_auc"]),
        "--max-drop-fraction",
        str(lock["max_drop_fraction"]),
        "--holdout-fraction",
        str(lock["holdout_fraction"]),
        "--w-heldout",
        str(lock["w_heldout"]),
        "--w-size",
        str(lock["w_size"]),
        "--w-imbalance",
        str(lock["w_imbalance"]),
    ]
    print("Re-running locked verification with LOCK_CONFIG parameters...")
    try:
        subprocess.run(cmd, cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"Verification rerun failed (exit={exc.returncode}). "
            "Check script output above for details.",
            file=sys.stderr,
        )
        return int(exc.returncode) if exc.returncode else 1

    files = [
        out_dir / "fixed_kept_count_summary.csv",
        out_dir / "fixed_kept_count_per_seed.csv",
        out_dir / "method_stability_summary.csv",
        out_dir / "multi_seed_results.csv",
    ]
    expected = lock["sha256"]
    ok = True
    for p in files:
        cur = sha256(p)
        exp = expected.get(p.name, "")
        same = cur == exp
        print(f"{p.name}: {'MATCH' if same else 'DIFF'}")
        if not same:
            print(f"  expected={exp}")
            print(f"  current ={cur}")
            ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

