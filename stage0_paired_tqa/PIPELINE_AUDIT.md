# PIPELINE_AUDIT

## Summary

### CRITICAL findings
- None found that directly inverts/scales/refits classifier inference or flips A/B scoring logic.

### WARNINGS
- Paired probe pair_ids overlap cleaned-training pair_ids (10/20 for v3/v4/v5), so pair_id-level leakage risk exists.
- Paired scorer scripts are not homogeneous on len_gap: score_paired_tqa.py uses train_len_gap_mean while later paired scorers use pair rel_len_gap(a,b).
- v5 means are >2σ from training means for 3 feature/side combinations (notably very long word_count).

### CLEAN checks
- All audited scorer scripts build surface vectors by iterating pickle-provided feature_cols order.
- All audited scorer scripts import surface_features_text for surface features.
- BGE and ModernBERT scorer configs match training builders on model ids, normalization policy, max_length, and CLS pooling/eval/no_grad pattern.
- No audited paired scorer inverts correctness direction (all treat picking A as correct).
- No scorer passes pair_id or explicit pair-structure metadata to classifier input tensors.
- No scorer script constructs StandardScaler or calls fit/fit_transform; trained pipelines are applied directly.
- Random factual-judge samples from v4/v5 are consistent with A=true, B=false labeling.
- Singleton and v5 paired scorers use the same extractor module (surface_features_text.py).

## 1) Classifier loading consistency

### surface_lr_full (`artifacts/surface_lr_full.pkl`)
- training-size metadata: n_rows=1580, n_pairs=790
- classes_: [0, 1] (len=2)
- coef shape: (1, 10)  feature_dim=10
- pipeline steps: [('standardscaler', 'StandardScaler'), ('logisticregression', 'LogisticRegression')]
- coef_[0][:5]: [0.13358117683725382, 0.7457241288159018, 0.23742921974859837, 0.015382473317125585, -0.006633161143615494]
- hash(coef_[0][:5]): `729881bdd2818847`
- surface feature order: ['neg_lead', 'neg_cnt', 'hedge_rate', 'auth_rate', 'len_gap', 'word_count', 'sent_count', 'avg_token_len', 'type_token', 'punc_rate']

### surface_lr_cleaned (`artifacts/surface_lr_cleaned.pkl`)
- training-size metadata: n_rows=1056, n_pairs=528
- classes_: [0, 1] (len=2)
- coef shape: (1, 10)  feature_dim=10
- pipeline steps: [('standardscaler', 'StandardScaler'), ('logisticregression', 'LogisticRegression')]
- coef_[0][:5]: [-0.018547786660570886, 0.04491316876031468, 0.2184005692264291, 0.15593647906907165, -0.009654302087421532]
- hash(coef_[0][:5]): `bf5b753bb6a856a5`
- surface feature order: ['neg_lead', 'neg_cnt', 'hedge_rate', 'auth_rate', 'len_gap', 'word_count', 'sent_count', 'avg_token_len', 'type_token', 'punc_rate']

### embedding_lr_full (`artifacts/embedding_lr_full.pkl`)
- training-size metadata: n_rows=1580, n_pairs=790
- classes_: [0, 1] (len=2)
- coef shape: (1, 1024)  feature_dim=1024
- pipeline steps: [('standardscaler', 'StandardScaler'), ('logisticregression', 'LogisticRegression')]
- coef_[0][:5]: [-0.01781136239570417, -0.05339804556759848, 0.5127524160077035, 0.04009804964616386, 0.08925939567676389]
- hash(coef_[0][:5]): `8ae2888c548f90b6`

### embedding_lr_cleaned (`artifacts/embedding_lr_cleaned.pkl`)
- training-size metadata: n_rows=1056, n_pairs=528
- classes_: [0, 1] (len=2)
- coef shape: (1, 1024)  feature_dim=1024
- pipeline steps: [('standardscaler', 'StandardScaler'), ('logisticregression', 'LogisticRegression')]
- coef_[0][:5]: [0.15739279652564625, -0.003235038853192789, 0.22758713296826014, 0.11042491323031646, 0.3020749810496753]
- hash(coef_[0][:5]): `c865855887b29437`

### modernbert_lr_full (`artifacts/modernbert_lr_full.pkl`)
- training-size metadata: n_rows=1580, n_pairs=790
- classes_: [0, 1] (len=2)
- coef shape: (1, 768)  feature_dim=768
- pipeline steps: [('standardscaler', 'StandardScaler'), ('logisticregression', 'LogisticRegression')]
- coef_[0][:5]: [-0.12147873438562515, -0.21957423601489726, -0.1442637341346511, -0.32672026008709915, -0.18599938450068426]
- hash(coef_[0][:5]): `59e54cf7ae2df241`

### modernbert_lr_cleaned (`artifacts/modernbert_lr_cleaned.pkl`)
- training-size metadata: n_rows=1056, n_pairs=528
- classes_: [0, 1] (len=2)
- coef shape: (1, 768)  feature_dim=768
- pipeline steps: [('standardscaler', 'StandardScaler'), ('logisticregression', 'LogisticRegression')]
- coef_[0][:5]: [-0.517348919783586, 0.0419244822400588, -0.6494467868637688, -0.5516169812143809, -0.028813105959357176]
- hash(coef_[0][:5]): `84ced2445e9343eb`

Scoring-script feature-order use:
- `score_singleton.py`: uses_surface_features_text=True, uses_pickle_feature_order=True
- `score_paired_tqa.py`: uses_surface_features_text=True, uses_pickle_feature_order=True
- `score_paired_classifiers_clean16.py`: uses_surface_features_text=True, uses_pickle_feature_order=True
- `score_paired_v4.py`: uses_surface_features_text=True, uses_pickle_feature_order=True
- `score_paired_v5.py`: uses_surface_features_text=True, uses_pickle_feature_order=True

## 2) Feature extraction consistency

- score_singleton uses surface extractor import: True
- score_paired_v5 uses surface extractor import: True
- score_singleton lower/strip/regex/tokenization source: `scripts/surface_features_text.py`
- score_paired_v5 lower/strip/regex/tokenization source: `scripts/surface_features_text.py`
- Any tokenization/regex differences between singleton and v5 are inherited from shared extractor, not scorer-local code.

## 3) len_gap handling

- `score_singleton.py`: mean-imputed constant (train_len_gap_mean)
- `score_paired_tqa.py`: mean-imputed constant
- `score_paired_classifiers_clean16.py`: pair rel_len_gap(a,b)
- `score_paired_v4.py`: pair rel_len_gap(a,b)
- `score_paired_v5.py`: pair rel_len_gap(a,b)

## 4) StandardScaler application

- `score_singleton.py`: contains `StandardScaler(...)`=False, fit/fit_transform calls=[]
- `score_paired_tqa.py`: contains `StandardScaler(...)`=False, fit/fit_transform calls=[]
- `score_paired_classifiers_clean16.py`: contains `StandardScaler(...)`=False, fit/fit_transform calls=[]
- `score_paired_v4.py`: contains `StandardScaler(...)`=False, fit/fit_transform calls=[]
- `score_paired_v5.py`: contains `StandardScaler(...)`=False, fit/fit_transform calls=[]

## 5) BGE and ModernBERT scoring consistency

- build_bge_embeddings.py: MODEL_ID="BAAI/bge-large-en-v1.5", normalize_embeddings=True
- build_modernbert_embeddings.py: MODEL_ID="answerdotai/ModernBERT-base", MAX_LENGTH=512, pooling=CLS, normalization=none
- scorer configs:
  - `score_singleton.py`: BGE_MODEL_ID="BAAI/bge-large-en-v1.5", BGE_NORMALIZE=True, MBERT_MODEL_ID="answerdotai/ModernBERT-base", MBERT_MAX_LENGTH=512, CLS=True, eval=True, no_grad=True
  - `score_paired_tqa.py`: BGE_MODEL_ID="BAAI/bge-large-en-v1.5", BGE_NORMALIZE=True, MBERT_MODEL_ID="answerdotai/ModernBERT-base", MBERT_MAX_LENGTH=512, CLS=True, eval=True, no_grad=True
  - `score_paired_classifiers_clean16.py`: BGE_MODEL_ID="BAAI/bge-large-en-v1.5", BGE_NORMALIZE=True, MBERT_MODEL_ID="answerdotai/ModernBERT-base", MBERT_MAX_LENGTH=512, CLS=True, eval=True, no_grad=True
  - `score_paired_v4.py`: BGE_MODEL_ID="BAAI/bge-large-en-v1.5", BGE_NORMALIZE=True, MBERT_MODEL_ID="answerdotai/ModernBERT-base", MBERT_MAX_LENGTH=512, CLS=True, eval=True, no_grad=True
  - `score_paired_v5.py`: BGE_MODEL_ID="BAAI/bge-large-en-v1.5", BGE_NORMALIZE=True, MBERT_MODEL_ID="answerdotai/ModernBERT-base", MBERT_MAX_LENGTH=512, CLS=True, eval=True, no_grad=True

## 6) A-side / B-side labeling consistency

- v3 random sample (judge is proposition/truth-class faithfulness, not factual correctness):
  - pid=41: a_same_prop=True, a_truthclass_same=False, pair_faithful=True, pair_truthclass_faithful=False
  - pid=115: a_same_prop=True, a_truthclass_same=True, pair_faithful=False, pair_truthclass_faithful=True
  - pid=202: a_same_prop=True, a_truthclass_same=True, pair_faithful=False, pair_truthclass_faithful=True
  - pid=216: a_same_prop=True, a_truthclass_same=True, pair_faithful=False, pair_truthclass_faithful=False
  - pid=482: a_same_prop=True, a_truthclass_same=True, pair_faithful=False, pair_truthclass_faithful=True
- v4 random sample (factual judge):
  - pid=37: a_correct=True, b_correct=False, pair_passes=True
  - pid=136: a_correct=True, b_correct=False, pair_passes=True
  - pid=153: a_correct=True, b_correct=False, pair_passes=True
  - pid=216: a_correct=True, b_correct=False, pair_passes=True
  - pid=222: a_correct=True, b_correct=False, pair_passes=True
- v5 random sample (factual judge):
  - pid=37: a_correct=True, b_correct=False, pair_passes=True
  - pid=101: a_correct=True, b_correct=False, pair_passes=True
  - pid=115: a_correct=True, b_correct=False, pair_passes=True
  - pid=357: a_correct=True, b_correct=False, pair_passes=True
  - pid=688: a_correct=True, b_correct=False, pair_passes=True

## 7) Scoring correctness direction

- ('score_singleton.py', 'singleton_n/a')
- ('score_paired_tqa.py', True, True, False)
- ('score_paired_classifiers_clean16.py', True, True, False)
- ('score_paired_v4.py', True, True, False)
- ('score_paired_v5.py', True, True, False)

## 8) Training data sanity

- cleaned retained pair_ids: 528
- overlap with cleaned training pair_ids: v3=10/20, v4=10/20, v5=10/20

## 9) Singleton vs paired input format

- `score_singleton.py`: explicit pair_id near predict_proba call = False
- `score_paired_tqa.py`: explicit pair_id near predict_proba call = False
- `score_paired_classifiers_clean16.py`: explicit pair_id near predict_proba call = False
- `score_paired_v4.py`: explicit pair_id near predict_proba call = False
- `score_paired_v5.py`: explicit pair_id near predict_proba call = False

## 10) Feature coverage on paired v5 data

| feature | v5_A_mean | train_y1_mean | train_y1_std | z(A vs y1) | v5_B_mean | train_y0_mean | train_y0_std | z(B vs y0) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| neg_cnt | 0.7000 | 0.5152 | 0.6278 | +0.29 | 0.3500 | 0.1468 | 0.4625 | +0.44 |
| hedge_rate | 0.0075 | 0.0051 | 0.0232 | +0.10 | 0.0155 | 0.0018 | 0.0159 | +0.86 |
| auth_rate | 0.0000 | 0.0001 | 0.0027 | -0.04 | 0.0091 | 0.0001 | 0.0036 | +2.53 |
| neg_lead | 0.1000 | 0.2494 | 0.4326 | -0.35 | 0.3000 | 0.0608 | 0.2389 | +1.00 |
| word_count | 30.3500 | 9.3747 | 4.0514 | +5.18 | 25.5000 | 8.6342 | 3.7729 | +4.47 |

>2σ flags:
- side=B, feature=auth_rate, v5_mean=0.0091, train_mean=0.0001, train_std=0.0036, z=+2.53
- side=A, feature=word_count, v5_mean=30.3500, train_mean=9.3747, train_std=4.0514, z=+5.18
- side=B, feature=word_count, v5_mean=25.5000, train_mean=8.6342, train_std=3.7729, z=+4.47

## 11) Pair_id disjointness

- v5 pair_ids overlap cleaned training set: 10/20
- v4 pair_ids overlap cleaned training set: 10/20
- v3 pair_ids overlap cleaned training set: 10/20
- v4-multi-seed overlap by seed:
- seed=1: 23/40
- seed=7: 26/40
- seed=42: 23/40

## 12) Versioning and reproducibility

### v1
- generate: scripts/generate_stage0_paired_tqa.py (historical v1 prompt no longer separately versioned)
- judge: scripts/judge_stage0_paired_tqa.py
- score: scripts/score_paired_tqa.py
### v2
- generate: scripts/generate_stage0_paired_tqa.py (v2 prompt no longer separately versioned)
- judge: scripts/judge_stage0_paired_tqa.py
- score: scripts/score_paired_tqa.py
### v3
- generate: scripts/generate_stage0_paired_tqa.py
- judge: scripts/judge_stage0_paired_tqa.py; scripts/rejudge_paired_tqa_truthclass.py
- score: scripts/score_paired_tqa.py; scripts/score_paired_classifiers_clean16.py
### v4
- generate: scripts/generate_stage0_paired_v4.py
- judge: scripts/judge_stage0_paired_v4.py
- score: scripts/score_paired_v4.py
### v4-multi-seed
- generate: scripts/generate_stage0_paired_v4_multi.py
- judge: scripts/judge_stage0_paired_v4.py (--gen-json)
- score: scripts/score_paired_v4.py (--gen-json/--judge-json); scripts/aggregate_v4_multi_seed.py
### v5
- generate: scripts/generate_stage0_paired_v5.py
- judge: scripts/judge_stage0_paired_v5.py
- score: scripts/score_paired_v5.py

Scoring-script functional differences:
- score_paired_tqa.py uses train_len_gap_mean; score_paired_v4.py/v5.py use pair rel_len_gap(a,b).
- Schema shift: v3 scorer consumes a_side_rewritten/b_side_rewritten; v4/v5 scorers consume a_side/b_side.
- Judge subset metric differs: v3 uses pair_faithful (same proposition), v4/v5 use pair_passes (A true and B false factuality).
- v5 scorer adds per-strategy breakdowns; v4 scorer reports only global aggregates.

- score_paired_tqa.py vs score_paired_v4.py diff lines: 778
- score_paired_v4.py vs score_paired_v5.py diff lines: 560

Diff excerpt: score_paired_tqa.py -> score_paired_v4.py
```diff
--- score_paired_tqa.py
+++ score_paired_v4.py
@@ -1,31 +1,25 @@
 #!/usr/bin/env python3
 """
-Step 6 + 7 (paired-pair audit probe) - score every pair's a_side_rewritten
-and b_side_rewritten with all SIX classifier pickles trained in Stage 0:
-
-    surface_lr_full.pkl            surface_lr_cleaned.pkl
-    embedding_lr_full.pkl   (BGE)  embedding_lr_cleaned.pkl   (BGE)
-    modernbert_lr_full.pkl         modernbert_lr_cleaned.pkl
-
-For each pair and each classifier:
-
-    P_a = predict_proba(a_side_rewritten)[:, 1]   # P(truthful-looking)
-    P_b = predict_proba(b_side_rewritten)[:, 1]
-    picked_side = "a" if P_a > P_b else "b"
-    correct = (picked_side == "a")          # A always carries the TRUE label
-
-Per-classifier aggregates over the 20 pairs:
-    pair_accuracy = mean(correct)
-    mean_margin   = mean(P_a - P_b)         # positive => leans toward A
-
-The 3-family comparison table compares pair_acc_full vs pair_acc_cleaned
-for each of {surface_lr, embedding_lr (BGE), modernbert_lr}.
-
-Outputs:
-    stage0_paired_tqa/stage0_paired_classifier_scores.json
-        list of 20 records, one per pair_id, with per-classifier
-        P(a), P(b), picked_side, correct_<family>_<train_set> flags.
-        Also includes per-family aggregates as a summary tail.
+v4 paired-probe scoring: send every v4 pair through all six existing
+Stage 0 classifier pickles. No retraining.
+
+Inputs:
+  stage0_paired_tqa/stage0_paired_generations_v4.json   (20 pairs)
+  stage0_paired_tqa/stage0_paired_judge_v4.json         (factual judge)
+
+For each pair and each (family, split) cell:
+  P_A = P_truthful(a_side), P_B = P_truthful(b_side)
+  picked  = "A" if P_A > P_B else "B"
+  correct = (picked == "A")     # A is the true-content side by design
+
+Aggregates per classifier (over ALL 20 pairs):
+  pair_accuracy, mean_P_A, mean_P_B, mean_gap = mean(P_A - P_B).
+
+We also report a sub-aggregate restricted to pair_passes-only (the
+subset where the GPT-5.4 judge confirmed both A=True and B=False) so
+that we can compare effects on the curated subset.
+
+Output: stage0_paired_tqa/stage0_paired_classifier_scores_v4.json.
 """
 from __future__ import annotations
 
@@ -38,33 +32,41 @@
 import joblib
 import numpy as np
 
-
 REPO_ROOT = Path(__file__).resolve().parent.parent
-GEN_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations.json"
-JUDGE_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_judge.json"
-OUT_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_classifier_scores.json"
+if str(REPO_ROOT) not in sys.path:
+    sys.path.insert(0, str(REPO_ROOT))
+
+DEFAULT_GEN_JSON = (REPO_ROOT / "stage0_paired_tqa"
+                    / "stage0_paired_generations_v4.json")
+DEFAULT_JUDGE_JSON = (REPO_ROOT / "stage0_paired_tqa"
+                      / "stage0_paired_judge_v4.json")
+DEFAULT_OUT_JSON = (REPO_ROOT / "stage0_paired_tqa"
+                    / "stage0_paired_classifier_scores_v4.json")
+# The multi-seed orchestrator overrides these via CLI flags.
+GEN_JSON: Path = DEFAULT_GEN_JSON
+JUDGE_JSON: Path = DEFAULT_JUDGE_JSON
+OUT_JSON: Path = DEFAULT_OUT_JSON
 
 ARTIFACTS = REPO_ROOT / "artifacts"
-SURFACE_FULL_PKL = ARTIFACTS / "surface_lr_full.pkl"
-SURFACE_CLEAN_PKL = ARTIFACTS / "surface_lr_cleaned.pkl"
-EMBED_FULL_PKL = ARTIFACTS / "embedding_lr_full.pkl"
-EMBED_CLEAN_PKL = ARTIFACTS / "embedding_lr_cleaned.pkl"
-MBERT_FULL_PKL = ARTIFACTS / "modernbert_lr_full.pkl"
-MBERT_CLEAN_PKL = ARTIFACTS / "modernbert_lr_cleaned.pkl"
-
-# BGE config (mirrors scripts/build_bge_embeddings.py and score_singleton.py)
+PICKLE_PATHS: dict[tuple[str, str], Path] = {
+    ("surface_lr",      "full"):    ARTIFACTS / "surface_lr_full.pkl",
+    ("surface_lr",      "cleaned"): ARTIFACTS / "surface_lr_cleaned.pkl",
+    ("BGE-large",       "full"):    ARTIFACTS / "embedding_lr_full.pkl",
+    ("BGE-large",       "cleaned"): ARTIFACTS / "embedding_lr_cleaned.pkl",
+    ("ModernBERT-base", "full"):    ARTIFACTS / "modernbert_lr_full.pkl",
+    ("ModernBERT-base", "cleaned"): ARTIFACTS / "modernbert_lr_cleaned.pkl",
+}
+
+# --- Encoder configs (match the existing build scripts) ---------------------
 BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
-BGE_BATCH = 20
+BGE_BATCH = 32
 BGE_NORMALIZE = True
 
-# ModernBERT config (mirrors scripts/build_modernbert_embeddings.py and
-# score_singleton.py - CLS pooling, NO normalization, max_length=512)
 MBERT_MODEL_ID = "answerdotai/ModernBERT-base"
-MBERT_BATCH = 20
+MBERT_BATCH = 32
 MBERT_MAX_LENGTH = 512
 
-
-# --- HF cache / offline setup (identical to score_singleton.py) ------------
+# HF cache / offline setup
 _HF_HOME_DEFAULT = ARTIFACTS / "hf_cache"
 os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
 os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
@@ -75,11 +77,23 @@
 os.environ.setdefault("OMP_NUM_THREADS", "1")
 os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
 
-
-# --- IO helpers ------------------------------------------------------------
... (658 more diff lines omitted)
```

Diff excerpt: score_paired_v4.py -> score_paired_v5.py
```diff
--- score_paired_v4.py
+++ score_paired_v5.py
@@ -1,28 +1,11 @@
 #!/usr/bin/env python3
 """
-v4 paired-probe scoring: send every v4 pair through all six existing
-Stage 0 classifier pickles. No retraining.
-
-Inputs:
-  stage0_paired_tqa/stage0_paired_generations_v4.json   (20 pairs)
-  stage0_paired_tqa/stage0_paired_judge_v4.json         (factual judge)
-
-For each pair and each (family, split) cell:
-  P_A = P_truthful(a_side), P_B = P_truthful(b_side)
-  picked  = "A" if P_A > P_B else "B"
-  correct = (picked == "A")     # A is the true-content side by design
-
-Aggregates per classifier (over ALL 20 pairs):
-  pair_accuracy, mean_P_A, mean_P_B, mean_gap = mean(P_A - P_B).
-
-We also report a sub-aggregate restricted to pair_passes-only (the
-subset where the GPT-5.4 judge confirmed both A=True and B=False) so
-that we can compare effects on the curated subset.
-
-Output: stage0_paired_tqa/stage0_paired_classifier_scores_v4.json.
+v5 paired-probe scoring over six existing Stage-0 classifier pickles.
+No retraining.
 """
 from __future__ import annotations
 
+import argparse
 import json
 import os
 import sys
@@ -36,37 +19,27 @@
 if str(REPO_ROOT) not in sys.path:
     sys.path.insert(0, str(REPO_ROOT))
 
-DEFAULT_GEN_JSON = (REPO_ROOT / "stage0_paired_tqa"
-                    / "stage0_paired_generations_v4.json")
-DEFAULT_JUDGE_JSON = (REPO_ROOT / "stage0_paired_tqa"
-                      / "stage0_paired_judge_v4.json")
-DEFAULT_OUT_JSON = (REPO_ROOT / "stage0_paired_tqa"
-                    / "stage0_paired_classifier_scores_v4.json")
-# The multi-seed orchestrator overrides these via CLI flags.
-GEN_JSON: Path = DEFAULT_GEN_JSON
-JUDGE_JSON: Path = DEFAULT_JUDGE_JSON
-OUT_JSON: Path = DEFAULT_OUT_JSON
+GEN_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_generations_v5.json"
+JUDGE_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_judge_v5.json"
+OUT_JSON = REPO_ROOT / "stage0_paired_tqa" / "stage0_paired_classifier_scores_v5.json"
 
 ARTIFACTS = REPO_ROOT / "artifacts"
 PICKLE_PATHS: dict[tuple[str, str], Path] = {
-    ("surface_lr",      "full"):    ARTIFACTS / "surface_lr_full.pkl",
-    ("surface_lr",      "cleaned"): ARTIFACTS / "surface_lr_cleaned.pkl",
-    ("BGE-large",       "full"):    ARTIFACTS / "embedding_lr_full.pkl",
-    ("BGE-large",       "cleaned"): ARTIFACTS / "embedding_lr_cleaned.pkl",
-    ("ModernBERT-base", "full"):    ARTIFACTS / "modernbert_lr_full.pkl",
+    ("surface_lr", "full"): ARTIFACTS / "surface_lr_full.pkl",
+    ("surface_lr", "cleaned"): ARTIFACTS / "surface_lr_cleaned.pkl",
+    ("BGE-large", "full"): ARTIFACTS / "embedding_lr_full.pkl",
+    ("BGE-large", "cleaned"): ARTIFACTS / "embedding_lr_cleaned.pkl",
+    ("ModernBERT-base", "full"): ARTIFACTS / "modernbert_lr_full.pkl",
     ("ModernBERT-base", "cleaned"): ARTIFACTS / "modernbert_lr_cleaned.pkl",
 }
 
-# --- Encoder configs (match the existing build scripts) ---------------------
 BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"
 BGE_BATCH = 32
 BGE_NORMALIZE = True
-
 MBERT_MODEL_ID = "answerdotai/ModernBERT-base"
 MBERT_BATCH = 32
 MBERT_MAX_LENGTH = 512
 
-# HF cache / offline setup
 _HF_HOME_DEFAULT = ARTIFACTS / "hf_cache"
 os.environ.setdefault("HF_HOME", str(_HF_HOME_DEFAULT))
 os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME_DEFAULT / "hub"))
@@ -77,13 +50,11 @@
 os.environ.setdefault("OMP_NUM_THREADS", "1")
 os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
 
-FULL_AUC_FLAG_THRESHOLD = 0.80
-
 FAMILIES = ["surface_lr", "BGE-large", "ModernBERT-base"]
 SPLITS = ["full", "cleaned"]
-
-
-# --- Loaders ---------------------------------------------------------------
+STRATEGIES = ["negation_opener", "hedging", "authority"]
+
+
 def _load_pickle(label: str, path: Path) -> dict:
     if not path.exists():
         raise FileNotFoundError(f"{label}: missing pickle {path}")
@@ -107,17 +78,15 @@
     return {int(r["pair_id"]): r for r in d}
 
 
-# --- P_truthful from a fitted Pipeline ------------------------------------
 def _proba_truthful(pipeline, X: np.ndarray) -> np.ndarray:
     clf = pipeline.steps[-1][1]
     classes = list(getattr(clf, "classes_", [0, 1]))
     if 1 not in classes:
-        raise RuntimeError(f"Classifier has no class 1; classes_={classes}")
+        raise RuntimeError(f"Classifier missing class 1; classes={classes}")
     col = classes.index(1)
     return np.asarray(pipeline.predict_proba(X)[:, col], dtype=float)
 
 
-# --- Surface10 featurizer (paired len_gap injection) ----------------------
 def _build_surface10(a_texts: list[str], b_texts: list[str],
                      feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
     from scripts.surface_features_text import extract_surface10, rel_len_gap
@@ -130,80 +99,73 @@
         fb["len_gap"] = gap
         Xa.append([float(fa[c]) for c in feature_cols])
         Xb.append([float(fb[c]) for c in feature_cols])
... (440 more diff lines omitted)
```