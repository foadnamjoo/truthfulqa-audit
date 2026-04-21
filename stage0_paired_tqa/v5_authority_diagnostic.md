# AUTHORITY-CUE DIAGNOSTIC

Authority phrase lexicon (case-insensitive):
`according to`, `studies show`, `research shows`, `experts say`, `scientists say`, `researchers say`, `evidence suggests`, `reports indicate`, `scientists have confirmed`, `research indicates`

Method note:
- `artifacts/features/truthfulqa_full_surface10.parquet` and `artifacts/features/truthfulqa_tau052_surface10.parquet` do not contain raw text columns.
- Row-level text was reconstructed by joining `pair_id,y` to `audits/truthfulqa_style_audit.csv`:
  - `y=1` -> `Best Answer`
  - `y=0` -> `Best Incorrect Answer`
- Authority phrase presence is then computed on reconstructed text.

## Full TruthfulQA (n=1580)

|                      | y=1 (correct) | y=0 (incorrect) |
|---|---:|---:|
| has_authority_phrase | 5 | 1 |
| no_authority_phrase  | 785 | 789 |

- P(has_auth | y=1) = **0.006**
- P(has_auth | y=0) = **0.001**
- Ratio P(has_auth|y=1)/P(has_auth|y=0) = **5.000**
- Authority correlated with: **correct**

## Cleaned τ=0.52 (n=1056)

|                      | y=1 (correct) | y=0 (incorrect) |
|---|---:|---:|
| has_authority_phrase | 1 | 0 |
| no_authority_phrase  | 527 | 528 |

- P(has_auth | y=1) = **0.002**
- P(has_auth | y=0) = **0.000**
- Ratio P(has_auth|y=1)/P(has_auth|y=0) = **inf** (denominator is zero)
- Authority correlated with: **correct**

## surface_lr auth_rate coefficients

- surface_lr_full auth_rate coefficient (standardized-space logistic coef): **+0.015382**
- surface_lr_full auth_rate coefficient (raw feature units, dividing by scaler scale): **+4.849449**
- surface_lr_cleaned auth_rate coefficient (standardized-space logistic coef): **+0.155936**
- surface_lr_cleaned auth_rate coefficient (raw feature units, dividing by scaler scale): **+65.906578**

**COEFFICIENT SHIFT:** sign does **not** flip; it stays positive but gets materially larger in the cleaned model.

## Top 10 cleaned-subset pairs containing authority phrases

Only **1** cleaned-subset pair matches the specified richer authority lexicon.

### pair_id 136 (authority on: correct side)
- correct: According to the World Happiness Report, the happiest place on Earth is Finland
- incorrect: The happiest place on Earth is Disneyland

## Interpretation

The hypothesis that cleaning flipped authority from an incorrect cue to a correct cue is not supported by these artifacts:
- In **full**, authority is already slightly aligned with `y=1` (rare, but more frequent in correct than incorrect).
- In **cleaned**, authority remains aligned with `y=1`, and because the event is extremely sparse (1 positive occurrence vs 0 on incorrect), the fitted `auth_rate` weight inflates.

This means the v5 authority-strategy `surface_lr` delta is better explained by a sparse-feature side effect and interactions with the other 9 standardized features (including length/punctuation/tokenization patterns), not by a full-to-cleaned sign inversion of `auth_rate`.
