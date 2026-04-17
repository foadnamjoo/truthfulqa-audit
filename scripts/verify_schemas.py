import pandas as pd
import sys
from pathlib import Path

base = Path('data/subsets/TruthfulQA-Audited')
expected = ['pair_id','Type','Category','Question',
            'Best Answer','Best Incorrect Answer','subset_name']

all_files = sorted(base.rglob('tqa_*.csv'))
print(f'Checking {len(all_files)} CSV files')
print('-' * 70)

all_ok = True
for f in all_files:
    try:
        df = pd.read_csv(f)
        cols = list(df.columns)
        ok = (cols == expected)
        status = 'OK ' if ok else 'BAD'
        rel = f.relative_to(base)
        print(f'[{status}] {rel} | rows={len(df)}')
        if not ok:
            all_ok = False
            print(f'      expected: {expected}')
            print(f'      got:      {cols}')
    except Exception as e:
        all_ok = False
        print(f'[ERR] {f.relative_to(base)}: {e}')
    sys.stdout.flush()

print('-' * 70)
print('ALL_OK' if all_ok else 'SOME_FAILED')
sys.exit(0 if all_ok else 1)
