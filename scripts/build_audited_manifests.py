import pandas as pd
from pathlib import Path

base = Path('data/subsets/TruthfulQA-Audited')

# feature_balanced manifest — pull AUC stats from the existing
# truthfulqaPro/subset_manifest.csv
src = pd.read_csv('truthfulqaPro/subset_manifest.csv')
fb_rows = []
for _, r in src.iterrows():
    K = int(r['target_kept_count'])
    fb_rows.append({
        'subset_name': f'tqa_K{K}',
        'n_pairs': K,
        'csv_path': f'feature_balanced/tqa_K{K}.csv',
        'pair_ids_path':
            f'feature_balanced/pair_ids/pair_ids_K{K}.json',
        'seed': 42,
        'mean_heldout_auc': float(r['mean_heldout_auc']),
        'std_heldout_auc': float(r['std_heldout_auc']),
        'paper_role': 'feature_balanced_baseline',
    })
pd.DataFrame(fb_rows).to_csv(
    base / 'feature_balanced_manifest.csv', index=False)
print('Wrote feature_balanced_manifest.csv with',
      len(fb_rows), 'rows')

# surface_audited manifest — three rows hardcoded from
# results/audit_prune_extended/summary_table.csv and
# results/subset_evaluation_preservation/summary_table.csv
sa_rows = [
    {
        'subset_name': 'tqa_tau052',
        'tau': 0.52, 'n_pairs': 528,
        'csv_path': 'surface_audited/tqa_tau052.csv',
        'pair_ids_path':
            'surface_audited/pair_ids/pair_ids_tau052.json',
        'grouped_cv_auc': 0.5130,
        'grouped_cv_accuracy': 0.5170,
        'spearman_rho': 0.9802,
        'kendall_tau': 0.9222,
        'construction_method': 'imbalance_audit_prune',
        'paper_role': 'primary',
    },
    {
        'subset_name': 'tqa_tau053',
        'tau': 0.53, 'n_pairs': 536,
        'csv_path': 'surface_audited/tqa_tau053.csv',
        'pair_ids_path':
            'surface_audited/pair_ids/pair_ids_tau053.json',
        'grouped_cv_auc': 0.5298,
        'grouped_cv_accuracy': 0.5252,
        'spearman_rho': 0.9714,
        'kendall_tau': 0.9000,
        'construction_method': 'imbalance_audit_prune',
        'paper_role': 'alternative',
    },
    {
        'subset_name': 'tqa_tau054',
        'tau': 0.54, 'n_pairs': 536,
        'csv_path': 'surface_audited/tqa_tau054.csv',
        'pair_ids_path':
            'surface_audited/pair_ids/pair_ids_tau054.json',
        'grouped_cv_auc': 0.5299,
        'grouped_cv_accuracy': 0.5336,
        'spearman_rho': 0.9791,
        'kendall_tau': 0.9222,
        'construction_method': 'imbalance_audit_prune',
        'paper_role': 'alternative',
    },
]
pd.DataFrame(sa_rows).to_csv(
    base / 'surface_audited_manifest.csv', index=False)
print('Wrote surface_audited_manifest.csv with',
      len(sa_rows), 'rows')
