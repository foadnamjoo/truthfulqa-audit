#!/usr/bin/env python3
"""
Smoke test: toy audit CSV, improved pruning script, paper10 default scorer.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import truthfulqa_paper_audit as tpa


def _toy_audit(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "Type": "T",
                "Category": "C",
                "Question": f"Q{i}",
                "Best Answer": "a",
                "Best Incorrect Answer": "b",
                "neg_lead_true": int(rng.random() > 0.7),
                "neg_lead_false": int(rng.random() > 0.8),
                "neg_cnt_true": int(rng.integers(0, 2)),
                "neg_cnt_false": int(rng.integers(0, 2)),
                "len_gap": float(rng.random()),
                "hedge_rate_true": float(rng.random() * 0.2),
                "hedge_rate_false": float(rng.random() * 0.2),
                "auth_rate_true": float(rng.random() * 0.1),
                "auth_rate_false": float(rng.random() * 0.1),
                "temporal_fragile": 0,
                "word_count_true": int(rng.integers(5, 20)),
                "word_count_false": int(rng.integers(5, 20)),
                "sent_count_true": 1,
                "sent_count_false": 1,
                "avg_token_len_true": 5.0,
                "avg_token_len_false": 5.0,
                "type_token_true": 0.9,
                "type_token_false": 0.9,
                "punc_rate_true": 0.0,
                "punc_rate_false": 0.0,
                "style_violation": int(rng.random() > 0.5),
            }
        )
    return pd.DataFrame(rows)


def test_paper10_default_scorer() -> None:
    audit = _toy_audit(30)
    df = tpa.build_answer_level_audit_frame(audit, profile="paper10", copy_audit_meta=False)
    assert list(df.columns)[:10] == tpa.FEAT_COLS_PAPER10
    r = tpa.paper_compatible_audit_oof_auc(df, profile="paper10", seed=0, n_splits=3)
    assert 0.4 <= r.auc_oof <= 1.0
    assert r.audit_profile == "paper10"
    assert len(r.feature_columns) == 10


def test_expanded13_optional() -> None:
    audit = _toy_audit(20)
    df = tpa.build_answer_level_audit_frame(audit, profile="expanded13", copy_audit_meta=False)
    r = tpa.paper_compatible_audit_oof_auc(df, profile="expanded13", seed=1, n_splits=3)
    assert len(r.feature_columns) == 13


def test_subprocess_pruning_writes_outputs() -> None:
    audit = _toy_audit(45)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "audits").mkdir()
        audit.to_csv(root / "audits" / "truthfulqa_style_audit.csv", index=False)
        cmd = [
            sys.executable,
            str(_SCRIPTS / "run_truthfulqa_pruning_improved.py"),
            "--root",
            str(root),
            "--audit-csv",
            "audits/truthfulqa_style_audit.csv",
            "--min-keep",
            "15",
            "--train-fraction",
            "0.7",
            "--cv-splits",
            "3",
            "--methods",
            "backward_elim",
            "--modes",
            "all_features",
            "--max-drop-fraction",
            "0.35",
            "--n-multistarts",
            "2",
        ]
        subprocess.check_call(cmd, cwd=str(root))
        out = root / "results" / "truthfulqa_pruning_improved"
        assert (out / "improved_search_results.csv").exists()
        assert (out / "best_subset_ids.json").exists()
        assert (out / "best_subset.csv").exists()
        assert (out / "dropped_example_explanations.csv").exists()
        data = json.loads((out / "best_subset_ids.json").read_text())
        assert "pair_ids" in data


def main() -> None:
    test_paper10_default_scorer()
    test_expanded13_optional()
    test_subprocess_pruning_writes_outputs()
    print("test_pruning_improved_pipeline: OK")


if __name__ == "__main__":
    main()
