import pandas as pd
import json
import sys
from pathlib import Path

base = Path('data/subsets/TruthfulQA-Audited')
results = []

# surface_audited
for tag in ['tau052','tau053','tau054']:
    csv_p = base / f'surface_audited/tqa_{tag}.csv'
    js_p = base / f'surface_audited/pair_ids/pair_ids_{tag}.json'
    try:
        csv_df = pd.read_csv(csv_p, usecols=['pair_id'])
        with open(js_p) as fh:
            js = json.load(fh)
        js_pairs = set(js if isinstance(js, list)
                       else js.get('pair_ids', []))
        csv_pairs = set(csv_df['pair_id'].astype(int).tolist())
        match = csv_pairs == js_pairs
        print(f'{tag:10s}: csv={len(csv_pairs):4d} '
              f'json={len(js_pairs):4d} match={match}')
        results.append(match)
    except Exception as e:
        print(f'{tag:10s}: ERROR {e}')
        results.append(False)
    sys.stdout.flush()

# feature_balanced
for K in [300,350,400,450,500,550,595,650]:
    csv_p = base / f'feature_balanced/tqa_K{K}.csv'
    js_p = base / f'feature_balanced/pair_ids/pair_ids_K{K}.json'
    try:
        csv_df = pd.read_csv(csv_p, usecols=['pair_id'])
        with open(js_p) as fh:
            js = json.load(fh)
        js_pairs = set(js if isinstance(js, list)
                       else js.get('pair_ids', []))
        csv_pairs = set(csv_df['pair_id'].astype(int).tolist())
        match = csv_pairs == js_pairs
        print(f'K{K:4d}     : csv={len(csv_pairs):4d} '
              f'json={len(js_pairs):4d} match={match}')
        results.append(match)
    except Exception as e:
        print(f'K{K:4d}     : ERROR {e}')
        results.append(False)
    sys.stdout.flush()

print('-' * 70)
print('ALL_MATCH' if all(results) else 'SOME_MISMATCHED')
sys.exit(0 if all(results) else 1)
