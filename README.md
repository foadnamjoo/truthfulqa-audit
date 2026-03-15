# TruthfulQA Style Confound Audit

An audit of **surface-form asymmetries** in the improved binary-choice [TruthfulQA](https://github.com/sylinrl/TruthfulQA) setting (2025). The notebook checks whether reference answer pairs exhibit shortcut-learnable cues (length, hedging, negation, etc.) that are detectable above chance—and whether that leakage may affect benchmark interpretation.

## Features

- **Lexicon-based surface-form features**: negation (lead/count), hedging, authority cues, length gap, shallow text stats
- **Grouped evaluation**: GroupKFold by question pair so both answers stay in the same fold (no train/test leakage)
- **Null baseline**: Within-pair label-swap null to test whether AUC exceeds chance
- **Category analysis**: Classifier performance by TruthfulQA category (broad vs concentrated effect)
- **Negation ablation**: Full vs no-negation feature set to test if the effect is driven only by negation
- **Clean vs confounded split**: Heuristic flagging of pairs with large asymmetries; optional benchmark-impact analysis with `model_predictions.csv`

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
   To run the model comparison (accuracy on clean vs confounded pairs), add a `model_predictions.csv` in the notebook directory with columns: `model_name`, `pair_id` (0..N-1), `correct` (1 if model chose Best Answer, 0 otherwise). See the notebook for the exact schema and example.

## Repository layout

| File | Description |
|------|-------------|
| `build_audit_notebook.py` | Script that generates the audit notebook |
| `TruthfulQA_Style_Confound_Audit.ipynb` | Main audit notebook (run this) |
| `audits/truthfulqa_style_audit.csv` | Exported audit table (created when you run the notebook §12) |

## References

- **Lin, S., Hilton, J., & Evans, O.** (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*.
- [TruthfulQA repository](https://github.com/sylinrl/TruthfulQA)
- **Evans, O., Chua, J., & Lin, S.** (2025). New, improved multiple-choice TruthfulQA (binary-choice format).

## License

MIT (or your chosen license).
