# TruthfulQA Surface-form Confound Audit

An audit of **surface-form asymmetries** in the improved binary-choice [TruthfulQA](https://github.com/sylinrl/TruthfulQA) setting (2025). The notebook checks whether reference answer pairs exhibit shortcut-learnable cues (length, hedging, negation, etc.) that are detectable above chance—and whether that leakage may affect benchmark interpretation.

## Features

- **Lexicon-based surface-form features**: negation (lead/count), hedging, authority cues, length gap, shallow text stats
- **Grouped evaluation**: GroupKFold by question pair so both answers stay in the same fold (no train/test leakage)
- **Null baseline**: Within-pair label-swap null to test whether AUC exceeds chance
- **Category analysis**: Classifier performance by TruthfulQA category (magnitude may vary by category; small categories can be noisy)
- **Negation ablation**: Full vs no-negation feature set to test if the effect is driven only by negation
- **Clean vs confounded split**: Heuristic flagging of pairs with large asymmetries; optional benchmark-impact analysis with real predictions (`model_predictions.csv`)

## Quick start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Regenerate the notebook** (optional; only if you edit the builder)
   ```bash
   python build_audit_notebook.py
   ```

3. **Run the audit**
   - Open `TruthfulQA_Style_Confound_Audit.ipynb` in Jupyter or VS Code and run all cells.
   - Data is loaded from the official TruthfulQA CSV on GitHub; no local data files required.

4. **Benchmark impact (§7c)**  
   - **Real predictions**: put real model outputs in `model_predictions.csv` (schema: `model_name`, `pair_id`, `correct`). The notebook will prefer this file automatically.\n+   - **Synthetic demo only**: `example_model_predictions.csv` is a synthetic file for demonstrating the §7c plumbing. **It is NOT real model output and must not be interpreted as empirical evidence.** To regenerate it: `python make_example_predictions.py`.

## Repository layout

| File | Description |
|------|-------------|
| `build_audit_notebook.py` | Script that generates the audit notebook |
| `TruthfulQA_Style_Confound_Audit.ipynb` | Main audit notebook (run this) |
| `model_predictions.csv` | Real model predictions for §7c (you provide this) |
| `example_model_predictions.csv` | **Synthetic demo** predictions for §7c plumbing only (not evidence) |
| `make_example_predictions.py` | Generates `example_model_predictions.csv` (synthetic demo) |
| `audits/truthfulqa_style_audit.csv` | Exported audit table (created when you run the notebook §12) |

## References

- **Lin, S., Hilton, J., & Evans, O.** (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*.
- [TruthfulQA repository](https://github.com/sylinrl/TruthfulQA)
- **Evans, O., Chua, J., & Lin, S.** (2025). New, improved multiple-choice TruthfulQA (binary-choice format).

## License

MIT (or your chosen license).
