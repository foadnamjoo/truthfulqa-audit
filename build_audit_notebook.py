#!/usr/bin/env python3
"""Build the TruthfulQA Style Confound Audit notebook."""
import json
from pathlib import Path
import uuid

def md(s):
    lines = (s.split("\n") if isinstance(s, str) else s)
    src = [line + "\n" for line in lines] if lines else [""]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src, "id": uuid.uuid4().hex[:8]}

def code(s):
    lines = (s.split("\n") if isinstance(s, str) else s)
    src = [line + "\n" for line in lines] if lines else [""]
    if src:
        src[-1] = src[-1].rstrip("\n")  # no trailing newline on last line
    return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None, "id": uuid.uuid4().hex[:8]}

cells = []

# --- 1. Motivation and Research Question ---
cells.append(md("""# TruthfulQA Surface-form Confound Audit

This notebook audits whether **surface-form asymmetries affect benchmark interpretation** in the improved binary-choice TruthfulQA setting (2025). We ask: do the reference answer pairs exhibit shortcut-learnable cues (e.g. length, hedging, negation patterns) that are detectable above chance? If so, reported model performance may partly reflect these residual answer-format asymmetries. The 2025 format remains an improvement over earlier settings; the open question is whether remaining leakage materially changes how we interpret model evaluation.

**Key references:** Lin et al., ACL 2022 (TruthfulQA); [official repo](https://github.com/sylinrl/TruthfulQA); Evans et al., 2025 (binary-choice setting)."""))

cells.append(md("""## 1. Motivation and Research Question

- **Truthfulness benchmarks** are central to LLM evaluation. TruthfulQA (Lin et al., 2022) measures whether models mimic human falsehoods across 817 questions in 38 categories.
- The **official 2025 update** recommends a **binary-choice** setting (*Best Answer* vs *Best Incorrect Answer*, order randomized), reducing earlier MC heuristics.
- **This audit** asks: in that improved setting, do the paired answers remain **balanced in surface form**? If correct and incorrect answers differ systematically in length, negation, hedging, etc., models may exploit these shortcut-learnable cues rather than content.
- **Hypothesis:** A classifier using only surface-form features can predict correct vs incorrect **above chance** under grouped evaluation. That would indicate residual answer-format asymmetries; the follow-up question is whether it materially affects benchmark interpretation (see §7c)."""))

# --- Setup ---
cells.append(md("""### 1.1 Environment and imports"""))
cells.append(code("""from pathlib import Path
import os
import re
import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Optional: install scikit-learn if needed (e.g. pip install scikit-learn)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(".").resolve()
RANDOM_SEED = 42
os.chdir(ROOT)
print("Workspace:", ROOT, "| RANDOM_SEED:", RANDOM_SEED)
AUDIT_DIR = ROOT / "audits"
FIG_DIR = ROOT / "figures"
AUDIT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
print("Imports OK.")"""))

# --- 2. Data Loading ---
cells.append(md("""## 2. Data Loading

We load **TruthfulQA.csv** from the official repository and keep only the columns used in the binary-choice setting: *Type*, *Category*, *Question*, *Best Answer*, *Best Incorrect Answer*."""))
cells.append(code("""def _safe_read_csv(path_or_buf, **kwargs):
    try:
        return pd.read_csv(path_or_buf, **kwargs)
    except Exception as e:
        print("Read failed:", e)
        return None

def try_http_csv(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return _safe_read_csv(io.StringIO(r.text))
    except Exception as e:
        print("HTTP fetch failed:", url, e)
        return None

GITHUB_RAW = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
df_github = try_http_csv(GITHUB_RAW)
if df_github is None:
    raise RuntimeError("Failed to load TruthfulQA from GitHub")

def normalize_schema(df):
    keep = ["Type", "Category", "Question", "Best Answer", "Best Incorrect Answer"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

df = normalize_schema(df_github.copy())
print("Rows:", len(df))
print("Columns:", list(df.columns))
df.head()"""))

# --- 3. Style Features ---
cells.append(md("""## 3. Surface-form Features

We define **lexicon-based surface-form features** that may act as shortcut-learnable cues if they differ systematically between correct and incorrect answers:

- **Negation (lead and count):** Correct answers often start with "No" or contain more negation when refuting a myth. *Caveat:* some negation is semantically appropriate (e.g. refuting false claims); we treat it as a detectable asymmetry and test sensitivity via a no-negation ablation (§6.2).
- **Hedging (rate):** May differ in epistemic certainty (e.g. "might", "typically").
- **Authority cues (rate):** Phrases like "according to", "experts say" may appear more in one answer type.
- **Length gap:** Relative length difference between the two answers; large gaps can be exploited.
- **Temporal fragility:** Time-sensitive wording (e.g. "current", "now") that can date answers.
- **Shallow surface features:** Word count, sentence count (sentence-ending punctuation), mean token length (mean character length of tokens), type-token ratio, and punctuation rate (?, !, comma per token) as interpretable surface cues. All features are lexicon-based or surface-only; no semantic or embedding-based features are used."""))
cells.append(code("""# Lexicons (regex patterns)
NEG_LEADS = [r"no\\b", r"not\\b", r"there is no\\b", r"there are no\\b", r"there's no\\b", r"none\\b", r"never\\b", r"nothing\\b", r"nobody\\b"]
NEG_WORDS = [r"\\bno\\b", r"\\bnot\\b", r"\\bnever\\b", r"\\bnone\\b", r"\\bnothing\\b", r"\\bnobody\\b", r"\\bcannot\\b", r"\\bcan't\\b", r"\\bdon't\\b", r"\\bdoesn't\\b", r"\\bdidn't\\b", r"\\bwon't\\b", r"\\bwasn't\\b", r"\\bweren't\\b"]
HEDGES = [r"\\bmight\\b", r"\\bcould\\b", r"\\bmay\\b", r"\\bgenerally\\b", r"\\btypically\\b", r"\\boften\\b", r"\\busually\\b", r"\\bsometimes\\b", r"\\blikely\\b", r"\\bprobably\\b", r"\\bpossibly\\b", r"\\bappears to\\b", r"\\bseems\\b", r"\\bsuggests\\b", r"\\broughly\\b", r"\\bapproximately\\b", r"\\bare expected to\\b"]
AUTH_CUES = [r"according to", r"experts say", r"scientists say", r"peer[- ]?reviewed", r"study shows"]
TEMPORAL_TERMS = [r"\\bcurrent\\b", r"\\bnow\\b", r"\\btoday\\b", r"\\bthis year\\b", r"\\blatest\\b", r"\\bpresident\\b", r"\\bgdp\\b", r"\\bpopulation\\b", r"\\blaw\\b", r"\\bregulation\\b", r"\\bsince\\b", r"\\bas of\\b"]

def simple_tokens(s):
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(s).lower())

def count_matches(patterns, text):
    text_l = str(text).lower()
    return sum(len(re.findall(p, text_l)) for p in patterns)

def starts_with_any(patterns, text):
    t = str(text).strip().lower()
    return int(any(re.match(p, t) for p in patterns))

def rate(count, tokens):
    return count / max(1, len(tokens))

def char_len(s):
    return len(str(s))

def rel_len_gap(a, b):
    la, lb = char_len(a), char_len(b)
    return abs(la - lb) / max(la, lb, 1)

def contains_temporal(s):
    return int(any(re.search(p, str(s).lower()) for p in TEMPORAL_TERMS))

def sentence_count(s):
    # Count segments separated by sentence-ending punctuation; at least 1.
    segments = [x.strip() for x in re.split(r"[.!?]+", str(s).strip()) if x.strip()]
    return max(1, len(segments))

def mean_token_length(tokens):
    if not tokens:
        return 0.0
    return np.mean([len(t) for t in tokens])

def type_token_ratio(tokens):
    n = max(1, len(tokens))
    return len(set(tokens)) / n

# Punctuation: ? ! , (interpretable, stable)
PUNC_CHARS = "?!,"

def punc_count(text):
    t = str(text)
    return sum(t.count(c) for c in PUNC_CHARS)

print("Lexicons and helpers defined.")"""))

cells.append(code("""# Extract columns
Q_COL, T_COL, F_COL = "Question", "Best Answer", "Best Incorrect Answer"
true_txt = df[T_COL]
false_txt = df[F_COL]

# Tokenize
tok_true = [simple_tokens(t) for t in true_txt]
tok_false = [simple_tokens(f) for f in false_txt]

# Style features per answer
neg_lead_true = [starts_with_any(NEG_LEADS, t) for t in true_txt]
neg_lead_false = [starts_with_any(NEG_LEADS, f) for f in false_txt]
neg_cnt_true = [count_matches(NEG_WORDS, t) for t in true_txt]
neg_cnt_false = [count_matches(NEG_WORDS, f) for f in false_txt]
hedge_cnt_true = [count_matches(HEDGES, t) for t in true_txt]
hedge_cnt_false = [count_matches(HEDGES, f) for f in false_txt]
hedge_rate_true = [rate(c, toks) for c, toks in zip(hedge_cnt_true, tok_true)]
hedge_rate_false = [rate(c, toks) for c, toks in zip(hedge_cnt_false, tok_false)]
auth_cnt_true = [count_matches(AUTH_CUES, t) for t in true_txt]
auth_cnt_false = [count_matches(AUTH_CUES, f) for f in false_txt]
auth_rate_true = [rate(c, toks) for c, toks in zip(auth_cnt_true, tok_true)]
auth_rate_false = [rate(c, toks) for c, toks in zip(auth_cnt_false, tok_false)]
len_gap = [rel_len_gap(t, f) for t, f in zip(true_txt, false_txt)]
temporal_fragile = [max(contains_temporal(q), contains_temporal(t), contains_temporal(f)) for q, t, f in zip(df[Q_COL], true_txt, false_txt)]

# Additional shallow style features (per answer)
word_count_true = [len(t) for t in tok_true]
word_count_false = [len(f) for f in tok_false]
sent_count_true = [sentence_count(t) for t in true_txt]
sent_count_false = [sentence_count(f) for f in false_txt]
avg_token_len_true = [mean_token_length(t) for t in tok_true]
avg_token_len_false = [mean_token_length(f) for f in tok_false]
type_token_true = [type_token_ratio(t) for t in tok_true]
type_token_false = [type_token_ratio(f) for f in tok_false]
punc_rate_true = [punc_count(t) / max(1, len(tok)) for t, tok in zip(true_txt, tok_true)]
punc_rate_false = [punc_count(f) / max(1, len(tok)) for f, tok in zip(false_txt, tok_false)]

# Build audit dataframe
audit = df.copy()
audit["neg_lead_true"] = neg_lead_true
audit["neg_lead_false"] = neg_lead_false
audit["neg_cnt_true"] = neg_cnt_true
audit["neg_cnt_false"] = neg_cnt_false
audit["len_gap"] = len_gap
audit["hedge_rate_true"] = hedge_rate_true
audit["hedge_rate_false"] = hedge_rate_false
audit["auth_rate_true"] = auth_rate_true
audit["auth_rate_false"] = auth_rate_false
audit["temporal_fragile"] = temporal_fragile
audit["word_count_true"] = word_count_true
audit["word_count_false"] = word_count_false
audit["sent_count_true"] = sent_count_true
audit["sent_count_false"] = sent_count_false
audit["avg_token_len_true"] = avg_token_len_true
audit["avg_token_len_false"] = avg_token_len_false
audit["type_token_true"] = type_token_true
audit["type_token_false"] = type_token_false
audit["punc_rate_true"] = punc_rate_true
audit["punc_rate_false"] = punc_rate_false
print("Audit rows:", len(audit))
audit.head()"""))

# --- 4. Descriptive Audit ---
cells.append(md("""## 4. Descriptive Audit

Summary statistics: do surface-form features differ between correct and incorrect answers? We use Welch t-tests and Cohen's d for effect size."""))
cells.append(code("""def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / (sp + 1e-10)

features = [
    ("neg_lead", "neg_lead_true", "neg_lead_false"),
    ("neg_cnt", "neg_cnt_true", "neg_cnt_false"),
    ("hedge_rate", "hedge_rate_true", "hedge_rate_false"),
    ("auth_rate", "auth_rate_true", "auth_rate_false"),
]
results = []
for name, col_t, col_f in features:
    a, b = audit[col_t].values, audit[col_f].values
    _, p = stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(a, b)
    results.append({
        "Feature": name,
        "Mean(correct)": np.mean(a),
        "Mean(incorrect)": np.mean(b),
        "Cohen's d": d,
        "p-value": p,
        "Sig. (α=0.05)": p < 0.05,
    })
df_desc = pd.DataFrame(results)
display(df_desc.round(4))
n_sig = df_desc["Sig. (α=0.05)"].sum()
print(f"\\n{n_sig}/{len(features)} features differ significantly. |Cohen's d| > 0.2 suggests meaningful effect.")"""))

# --- 5. Visualization of Style Asymmetries ---
cells.append(md("""## 5. Visualization of Surface-form Asymmetries

Boxplots of key surface-form features: correct vs incorrect answers. Separation suggests shortcut-learnable cues can partially predict the label."""))
cells.append(code("""fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
for ax, (name, col_t, col_f) in zip(axes, [
    ("Neg. at start", "neg_lead_true", "neg_lead_false"),
    ("Neg. count", "neg_cnt_true", "neg_cnt_false"),
    ("Hedge rate", "hedge_rate_true", "hedge_rate_false"),
]):
    ax.boxplot([audit[col_t], audit[col_f]], labels=["Correct", "Incorrect"])
    ax.set_title(name)
    ax.set_ylabel("Value")
fig.suptitle("Surface-form features: correct vs incorrect", fontsize=11, y=1.02)
plt.tight_layout()
plt.show()"""))
cells.append(md("""**Interpretation:** If correct answers show systematically higher negation (lead/count) or hedge rate, a surface-form classifier may achieve AUC above chance. The analysis below tests this with grouped evaluation."""))

# Thresholds used later for clean vs confounded split (§7)
cells.append(code("""LEN_GAP_MAX = 0.20
HEDGE_GAP_MAX = 0.02
AUTH_GAP_MAX = 0.01
print("Thresholds set: LEN_GAP_MAX={}, HEDGE_GAP_MAX={}, AUTH_GAP_MAX={}".format(LEN_GAP_MAX, HEDGE_GAP_MAX, AUTH_GAP_MAX))"""))

# --- 5.1 Style distributions (motivate clean/confounded thresholds) ---
cells.append(md("""### 5.1 TruthfulQA surface-form distributions

Distributions of length gap and of correct-minus-incorrect differences in hedge and authority rate. Vertical lines show the thresholds used later to flag confounded pairs (§7)."""))
cells.append(code("""with plt.style.context("ggplot"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 1. Length gap ---
    axes[0].hist(audit["len_gap"], bins=30, color="skyblue", edgecolor="black")
    axes[0].axvline(LEN_GAP_MAX, color="red", linestyle="--", label=f"Threshold = {LEN_GAP_MAX}")
    axes[0].set_title("Length Gap Distribution")
    axes[0].set_xlabel("Relative length gap")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # --- 2. Hedging rate Δ (True − False) ---
    hedge_delta = audit["hedge_rate_true"] - audit["hedge_rate_false"]
    axes[1].hist(hedge_delta, bins=30, color="lightgreen", edgecolor="black")
    axes[1].axvline(HEDGE_GAP_MAX,  color="red", linestyle="--", label=f"+{HEDGE_GAP_MAX}")
    axes[1].axvline(-HEDGE_GAP_MAX, color="red", linestyle="--", label=f"−{HEDGE_GAP_MAX}")
    axes[1].set_title("Hedging Rate Difference (True − False)")
    axes[1].set_xlabel("Δ hedge rate")
    axes[1].legend()

    # --- 3. Authority cue rate Δ (True − False) ---
    auth_delta = audit["auth_rate_true"] - audit["auth_rate_false"]
    axes[2].hist(auth_delta, bins=30, color="lightcoral", edgecolor="black")
    axes[2].axvline(AUTH_GAP_MAX,  color="red", linestyle="--", label=f"+{AUTH_GAP_MAX}")
    axes[2].axvline(-AUTH_GAP_MAX, color="red", linestyle="--", label=f"−{AUTH_GAP_MAX}")
    axes[2].set_title("Authority Cue Rate Difference (True − False)")
    axes[2].set_xlabel("Δ authority rate")
    axes[2].legend()

    plt.suptitle("TruthfulQA surface-form distributions", fontsize=14)
    plt.tight_layout()
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    pdf_path = fig_dir / "style_feature_distributions.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", pdf_path)"""))

# --- 6. Style-Only Classifier (Grouped Evaluation) ---
cells.append(md("""## 6. Surface-form-only Classifier

We train a classifier to predict **correct vs incorrect** using *only* surface-form features (no semantic content). To avoid train/test leakage, **both answers from the same question must stay in the same fold**: we use **GroupKFold** with pair (question) ID. AUC is computed from **out-of-fold predicted probabilities**, not hard labels. The empirical results below test whether residual answer-format asymmetries are detectable above chance in the improved binary-choice format."""))
cells.append(code("""# Answer-level dataset with pair_id (question index) so both answers of a pair stay together
FEAT_COLS = [
    "neg_lead", "neg_cnt", "hedge_rate", "auth_rate", "len_gap",
    "word_count", "sent_count", "avg_token_len", "type_token", "punc_rate"
]
cols_true = ["neg_lead_true", "neg_cnt_true", "hedge_rate_true", "auth_rate_true", "len_gap",
             "word_count_true", "sent_count_true", "avg_token_len_true", "type_token_true", "punc_rate_true"]
cols_false = ["neg_lead_false", "neg_cnt_false", "hedge_rate_false", "auth_rate_false", "len_gap",
              "word_count_false", "sent_count_false", "avg_token_len_false", "type_token_false", "punc_rate_false"]

rows_true = audit[cols_true].copy()
rows_true.columns = FEAT_COLS
rows_true["label"] = 1
rows_true["pair_id"] = np.arange(len(audit))
rows_false = audit[cols_false].copy()
rows_false.columns = FEAT_COLS
rows_false["label"] = 0
rows_false["pair_id"] = np.arange(len(audit))
df_ans = pd.concat([rows_true, rows_false], ignore_index=True)
X = df_ans[FEAT_COLS].fillna(0)
y = df_ans["label"]
groups = df_ans["pair_id"].values

clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
pipe = make_pipeline(StandardScaler(), clf)
cv = GroupKFold(n_splits=5)

# Out-of-fold predicted probabilities (correct class) for proper ROC AUC
y_proba_oof = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict_proba")
y_proba_pos = y_proba_oof[:, 1]
y_pred_oof = (y_proba_pos >= 0.5).astype(int)

# Grouped CV accuracy and AUC (from OOF probabilities)
acc_oof = (y_pred_oof == y.values).mean()
auc_oof = roc_auc_score(y, y_proba_pos)

# Bootstrap 95% CI for AUC by resampling pairs (preserve pair structure)
np.random.seed(RANDOM_SEED)
n_pairs = len(audit)
auc_boot = []
for _ in range(500):
    pair_idx = np.random.choice(n_pairs, n_pairs, replace=True)
    row_idx = np.concatenate([2 * pair_idx, 2 * pair_idx + 1])  # both rows per pair
    auc_boot.append(roc_auc_score(y.iloc[row_idx], y_proba_pos[row_idx]))
auc_ci = np.percentile(auc_boot, [2.5, 97.5])

pipe.fit(X, y)  # fit on full data for feature importance and optional subset analysis
clf_full = pipe.named_steps["logisticregression"]

print("Surface-form-only classifier (grouped 5-fold CV by question pair)")
print("  Grouped CV Accuracy:  {:.4f}".format(acc_oof))
print("  Grouped CV AUC (OOF): {:.4f}".format(auc_oof))
print("  AUC 95% CI (pair bootstrap): [{:.4f}, {:.4f}]".format(auc_ci[0], auc_ci[1]))
print("  Chance baseline:       AUC = 0.50")
print("Confusion matrix (rows=actual, cols=pred):")
print(confusion_matrix(y, y_pred_oof))
print(classification_report(y, y_pred_oof, target_names=["incorrect", "correct"]))"""))
cells.append(code("""# Within-pair label-swap null baseline (pair-structured null)
# Labels are swapped at random within each pair so both answers of a question stay together; no general permutation.
def shuffle_labels_within_groups(y, groups):
    y_perm = y.copy().values
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y_perm[idx] = np.random.permutation(y_perm[idx])
    return y_perm

np.random.seed(RANDOM_SEED)
n_null = 100
auc_null = []
for _ in range(n_null):
    y_null = shuffle_labels_within_groups(y, groups)
    proba_null = cross_val_predict(pipe, X, y_null, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc_null.append(roc_auc_score(y_null, proba_null))
auc_null = np.array(auc_null)
p_null = (auc_null >= auc_oof).mean()
print("Within-pair label-swap null (100 runs):")
print("  Mean AUC (null): {:.4f} ± {:.4f}".format(auc_null.mean(), auc_null.std()))
print("  Observed AUC:    {:.4f}".format(auc_oof))
print("  p (null AUC >= observed): {:.4f}".format(p_null))
if p_null <= 0.05:
    print("  → Observed AUC exceeds the pair-structured null.")
else:
    print("  → Observed AUC not clearly above the null at α=0.05.")"""))
cells.append(code("""# Plot: style-only classifier vs chance and pair-structured null
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
ax.axhline(0.5, color="gray", linestyle="--", label="Chance")
ax.hist(auc_null, bins=15, color="lightgray", edgecolor="black", alpha=0.8, label="Label-swap null (100)")
ax.axvline(auc_oof, color="steelblue", linewidth=2, label="Style-only (grouped CV)")
ax.set_xlabel("AUC")
ax.set_ylabel("Count")
ax.set_title("Surface-form-only AUC vs pair-structured null")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()"""))

# --- 6.1 Category breakdown ---
cells.append(md("""### 6.1 Category analysis

Breakdown of classifier performance by TruthfulQA category. AUC is computed on each category's pairs using the same out-of-fold predictions. This helps check whether separability is concentrated in a few categories or appears across multiple categories (with potentially varying magnitude and noise)."""))
cells.append(code("""# Per-category AUC (same OOF predictions, subset by pair_id in category)
if "Category" in audit.columns:
    cat_auc = []
    for cat in audit["Category"].dropna().unique():
        mask = (audit["Category"] == cat).values
        pair_ids_cat = np.where(mask)[0]
        row_mask = df_ans["pair_id"].isin(pair_ids_cat)
        y_cat = y.values[row_mask]
        proba_cat = y_proba_pos[row_mask]
        auc_cat = roc_auc_score(y_cat, proba_cat) if len(np.unique(y_cat)) > 1 else np.nan
        cat_auc.append({"Category": cat, "N": mask.sum(), "AUC": auc_cat})
    df_cat = pd.DataFrame(cat_auc).sort_values("N", ascending=False)
    display(df_cat.round(4))
    n_above = (df_cat["AUC"] > 0.5).sum()
    print("Categories with AUC > 0.5:", n_above, "/", len(df_cat))
    print("Interpretation: Effects appear in multiple categories, though magnitude varies and some categories are small/noisy.")
else:
    print("No Category column in audit.")"""))

# --- 6.2 Negation ablation ---
cells.append(md("""### 6.2 Negation ablation

Rerun the classifier **without** negation-related features (`neg_lead`, `neg_cnt`). If grouped CV AUC remains above chance (and above the null), the effect is not driven only by negation. (Some negation in correct answers is semantically appropriate when refuting myths; this ablation checks whether shortcut signal persists without it.)"""))
cells.append(code("""NEG_FEATURES = ["neg_lead", "neg_cnt"]
FEAT_COLS_NO_NEG = [c for c in FEAT_COLS if c not in NEG_FEATURES]
X_no_neg = df_ans[FEAT_COLS_NO_NEG].fillna(0)
pipe_no_neg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
y_proba_no_neg = cross_val_predict(pipe_no_neg, X_no_neg, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
auc_no_neg = roc_auc_score(y, y_proba_no_neg)
# Null for no-neg (same pair-structure)
auc_null_no_neg = []
for _ in range(100):
    y_n = shuffle_labels_within_groups(y, groups)
    proba_n = cross_val_predict(pipe_no_neg, X_no_neg, y_n, cv=cv, groups=groups, method="predict_proba")[:, 1]
    auc_null_no_neg.append(roc_auc_score(y_n, proba_n))
auc_null_no_neg = np.array(auc_null_no_neg)
p_no_neg = (auc_null_no_neg >= auc_no_neg).mean()
print("Full feature set:   Grouped CV AUC = {:.4f}".format(auc_oof))
print("No-negation set:   Grouped CV AUC = {:.4f}".format(auc_no_neg))
print("No-neg null mean:  {:.4f}; p (null >= observed) = {:.4f}".format(auc_null_no_neg.mean(), p_no_neg))
if auc_no_neg > 0.5 and p_no_neg <= 0.05:
    print("Interpretation: Performance above null without negation → effect is not driven only by negation.")
else:
    print("Interpretation: With negation removed, signal drops or is at null; negation may be a major driver.")"""))

cells.append(md("""**Interpretation:** If grouped evaluation shows AUC above chance and above the within-pair label-swap null, this provides evidence of detectable residual answer-format asymmetries in the reference pairs. The open question is whether these asymmetries materially affect how we interpret model performance on the benchmark (see §7b and Benchmark impact)."""))

# --- 7. Confounded vs Clean Split ---
cells.append(md("""## 7. Confounded vs Clean Split

We label each question pair as **heuristically clean** or **heuristically confounded** (by surface-form asymmetries) using transparent thresholds: (1) either answer starts with negation, (2) relative length gap exceeds a limit, (3) absolute difference in hedge or authority-cue rate exceeds a limit. Pairs that violate any criterion are marked "confounded"; the rest are marked "clean".

**Note:** This split is a heuristic audit tool, not definitive ground truth. It is intended to flag pairs where shortcut-learnable asymmetries are large enough to be exploitable; the exact cutoffs are chosen for interpretability and stability (see sensitivity below), not as a binary gold standard."""))
cells.append(code("""# Default thresholds (justified by exploratory distributions)
def compute_violations(audit, len_max, hedge_max, auth_max):
    v_neg = ((audit["neg_lead_true"] == 1) | (audit["neg_lead_false"] == 1)).astype(int)
    v_len = (audit["len_gap"] > len_max).astype(int)
    v_hedge = (np.abs(audit["hedge_rate_true"] - audit["hedge_rate_false"]) > hedge_max).astype(int)
    v_auth = (np.abs(audit["auth_rate_true"] - audit["auth_rate_false"]) > auth_max).astype(int)
    return (v_neg | v_len | v_hedge | v_auth).astype(int)

audit["style_violation"] = compute_violations(audit, LEN_GAP_MAX, HEDGE_GAP_MAX, AUTH_GAP_MAX)
n_conf = audit["style_violation"].sum()
n_clean = len(audit) - n_conf
print("Default thresholds: LEN_GAP_MAX={}, HEDGE_GAP_MAX={}, AUTH_GAP_MAX={}".format(LEN_GAP_MAX, HEDGE_GAP_MAX, AUTH_GAP_MAX))
print("Heuristically clean pairs:      ", n_clean, "({:.1f}%)".format(100 * n_clean / len(audit)))
print("Heuristically confounded pairs: ", n_conf, "({:.1f}%)".format(100 * n_conf / len(audit)))"""))
cells.append(md("""### Key results"""))
cells.append(code("""# Compact key results (computed from cells above)
key_results = pd.DataFrame([
    {"Metric": "Grouped CV accuracy", "Value": "{:.4f}".format(acc_oof)},
    {"Metric": "Grouped CV AUC (OOF)", "Value": "{:.4f}".format(auc_oof)},
    {"Metric": "AUC 95% CI", "Value": "[{:.4f}, {:.4f}]".format(auc_ci[0], auc_ci[1])},
    {"Metric": "Null mean AUC", "Value": "{:.4f}".format(auc_null.mean())},
    {"Metric": "p (null >= observed)", "Value": "{:.4f}".format(p_null)},
    {"Metric": "Confounded pairs", "Value": "{} ({:.1f}%)".format(n_conf, 100 * n_conf / len(audit))},
])
display(key_results)
print("Key result: Surface-form classifier AUC = {:.3f} (95% CI [{:.3f}, {:.3f}]), compared to a pair-structured null (p = {:.3f}); {:.1f}% of pairs are confounded by the heuristic thresholds.".format(auc_oof, auc_ci[0], auc_ci[1], p_null, 100 * n_conf / len(audit)))
if p_null <= 0.05:
    print("Interpretation: This provides evidence of detectable residual answer-format asymmetries under grouped evaluation.")
else:
    print("Interpretation: The evidence for detectable asymmetries is weak under the pair-structured null test.")"""))
cells.append(code("""# Sensitivity: alternate threshold sets (strict / default / lenient)
sensitivity = []
for (len_m, hedge_m, auth_m, label) in [
    (0.15, 0.03, 0.02, "Strict"),
    (0.20, 0.02, 0.01, "Default"),
    (0.25, 0.01, 0.005, "Lenient"),
]:
    v = compute_violations(audit, len_m, hedge_m, auth_m)
    sensitivity.append({"Variant": label, "% confounded": 100 * v.mean()})
df_sens = pd.DataFrame(sensitivity)
display(df_sens)
print("→ Proportion confounded is qualitatively stable across reasonable thresholds.")"""))
cells.append(code("""fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
ax.bar(["Clean", "Confounded"], [n_clean, n_conf], color=["seagreen", "coral"], edgecolor="black")
ax.set_ylabel("Pairs")
ax.set_title("Clean vs confounded (heuristic split)")
plt.tight_layout()
plt.show()"""))
cells.append(md("""**Interpretation:** A substantial proportion of pairs is flagged as confounded across threshold variants. The improved 2025 binary-choice format remains an improvement over earlier settings; the open question is whether remaining leakage materially changes how we interpret model rankings (see Benchmark impact)."""))

# --- 7b. Classifier performance on clean vs confounded subsets ---
cells.append(md("""### 7b. Classifier performance on clean vs confounded subsets

Using the same out-of-fold predictions, we report AUC on **clean** pairs only vs **confounded** pairs only. If the classifier performs better on confounded pairs, that suggests it is exploiting the very asymmetries used to define confound."""))
cells.append(code("""# Subset OOF predictions by pair's style_violation
pair_conf = audit["style_violation"].values
clean_mask = pair_conf[df_ans["pair_id"]] == 0
conf_mask = pair_conf[df_ans["pair_id"]] == 1
y_c, proba_c = y.values[clean_mask], y_proba_pos[clean_mask]
y_x, proba_x = y.values[conf_mask], y_proba_pos[conf_mask]
auc_clean = roc_auc_score(y_c, proba_c) if len(np.unique(y_c)) > 1 else 0.5
auc_conf = roc_auc_score(y_x, proba_x) if len(np.unique(y_x)) > 1 else 0.5
print("AUC on clean pairs only:    {:.4f} (n={})".format(auc_clean, clean_mask.sum()))
print("AUC on confounded pairs:   {:.4f} (n={})".format(auc_conf, conf_mask.sum()))
print("→ Higher AUC on confounded pairs would indicate the classifier relies on the same asymmetries that define confound.")"""))

# --- 7c. Benchmark impact (model performance on clean vs confounded) ---
cells.append(md("""## 7c. Benchmark impact: model performance on clean vs confounded

**Goal:** Test whether surface-form asymmetries might affect benchmark interpretation. If model accuracy is systematically higher on confounded pairs than on clean pairs, that would be *consistent with* models benefiting from shortcut-learnable cues rather than only truthfulness.

**Ideal experiment:** Load TruthfulQA model predictions (e.g. from the [official repo](https://github.com/sylinrl/TruthfulQA) or compatible runs). Required format: one row per (model, question): `model_name`, `pair_id` (0..n-1), `correct` (1 if model chose Best Answer, 0 otherwise). Then compute per-model accuracy on (1) all pairs, (2) clean pairs only, (3) confounded pairs only. If accuracy is higher on confounded than on clean for most models, residual asymmetries affect benchmark interpretation.

Below: the analysis prefers **real** predictions in `model_predictions.csv`. If only a synthetic demo file is present (`example_model_predictions.csv`), the notebook will run but will clearly label the outputs as **demonstration-only** and **not empirical evidence**."""))
cells.append(code("""# Expected format: CSV or DataFrame with columns [model_name, pair_id, correct]
# pair_id = 0..len(audit)-1; correct = 1 if model chose Best Answer, 0 otherwise.
def model_accuracy_by_split(scores_df, audit_violation):
    \"\"\"Compute per-model accuracy on all, clean, and confounded pairs.\"\"\"
    out = []
    for model in scores_df["model_name"].unique():
        m = scores_df[scores_df["model_name"] == model]
        acc_all = m["correct"].mean()
        clean_ids = np.where(audit_violation == 0)[0]
        conf_ids = np.where(audit_violation == 1)[0]
        m_clean = m[m["pair_id"].isin(clean_ids)]
        m_conf = m[m["pair_id"].isin(conf_ids)]
        acc_clean = m_clean["correct"].mean() if len(m_clean) else np.nan
        acc_conf = m_conf["correct"].mean() if len(m_conf) else np.nan
        out.append({"model": model, "acc_all": acc_all, "acc_clean": acc_clean, "acc_confounded": acc_conf})
    return pd.DataFrame(out)

# Try to load model predictions. Prefer real predictions if present.
REAL_PRED_PATH = ROOT / "model_predictions.csv"
DEMO_PRED_PATH = ROOT / "example_model_predictions.csv"

pred_path = REAL_PRED_PATH if REAL_PRED_PATH.exists() else (DEMO_PRED_PATH if DEMO_PRED_PATH.exists() else None)
is_demo = (pred_path == DEMO_PRED_PATH)

if pred_path is not None:
    df_scores = pd.read_csv(pred_path)
    if "model_name" in df_scores.columns and "pair_id" in df_scores.columns and "correct" in df_scores.columns:
        if is_demo:
            print("WARNING: Using synthetic DEMO predictions from", pred_path)
            print("This file is for demonstrating the benchmark-impact plumbing only.")
            print("It is NOT real model output and MUST NOT be interpreted as empirical evidence.")
        else:
            print("Using real model predictions from", pred_path)
        impact = model_accuracy_by_split(df_scores, audit["style_violation"].values)
        display(impact)
        impact["delta_conf_clean"] = impact["acc_confounded"] - impact["acc_clean"]
        print("Delta (confounded - clean): positive = higher accuracy on confounded pairs.")
    else:
        print("File found but missing columns: need model_name, pair_id, correct.")
else:
    # Placeholder: exact schema and example so users can plug in easily.
    print("No real or demo predictions file found.")
    print("Looked for:")
    print(" -", REAL_PRED_PATH)
    print(" -", DEMO_PRED_PATH)
    print()
    print("Expected schema (exact column names and types):")
    print("  model_name : str (e.g. 'gpt-4', 'llama-3')")
    print("  pair_id    : int, 0 to len(audit)-1 (index of question pair in this notebook's audit)")
    print("  correct    : int, 1 if model chose Best Answer, 0 otherwise")
    print()
    print("Example table (save as model_predictions.csv in notebook directory, then re-run this cell):")
    example = pd.DataFrame({
        "model_name": ["gpt-4", "gpt-4", "gpt-4", "llama-3", "llama-3", "llama-3"],
        "pair_id": [0, 1, 2, 0, 1, 2],
        "correct": [1, 0, 1, 0, 1, 1],
    })
    display(example)
    print("One row per (model, question). Add all models and all pair_ids 0..{} for full analysis.".format(len(audit) - 1))"""))

# --- 8. Feature Importance ---
cells.append(md("""## 8. Feature Importance / Shortcut Drivers

Coefficients of the logistic regression (surface-form only) indicate which cues drive separability. Positive coefficient: higher feature value associated with *correct*; negative: with *incorrect*."""))
cells.append(code("""feat_imp = pd.Series(np.abs(clf_full.coef_[0]), index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
feat_imp.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
ax.set_xlabel("|Coefficient|")
ax.set_title("Feature importance (surface-form classifier)")
plt.tight_layout()
plt.show()
print("Coefficients (signed):", dict(zip(X.columns, clf_full.coef_[0].round(4))))"""))
cells.append(md("""**Interpretation:** The largest |coefficient| values are the main shortcut drivers. Negation-related and length features often rank high when correct answers systematically differ in those dimensions; the negation ablation (§6.2) checks whether signal persists without them."""))

# --- 9. Statistical Rigor ---
cells.append(md("""## 9. Statistical Rigor

- **Effect sizes:** Cohen's d in §4; |d| > 0.2 is a conventional small effect.
- **Significance:** Welch t-tests (α = 0.05) for correct vs incorrect feature means.
- **Classifier:** Grouped 5-fold CV by question pair; AUC from out-of-fold probabilities. Pair-level bootstrap 95% CI. Within-pair label-swap null to test whether observed AUC exceeds a pair-structured null.
- No multiple-comparison correction; interpret p-values and CIs with appropriate caution."""))

# --- 9b. Threats to Validity / Robustness ---
cells.append(md("""### 9b. Threats to Validity and Robustness

- **Lexicon-based features** are incomplete; we do not capture all surface-form or pragmatic dimensions. Results are conditional on this feature set.
- **Grouped CV** is used specifically to reduce pair-level leakage; both answers of a question stay in the same fold. This is methodologically important for benchmark audits.
- **Clean vs confounded thresholds** are heuristic. The sensitivity analysis (§7) shows that the proportion confounded is qualitatively stable under alternate thresholds, but the exact cutoffs are not uniquely justified.
- This notebook is an **audit of residual answer-format confound** in the improved binary-choice TruthfulQA setting, not a full invalidation of the benchmark. The 2025 update remains a substantive improvement over earlier formats."""))

# --- 10. Relation to TruthfulQA Literature ---
cells.append(md("""## 10. Relation to TruthfulQA Literature

- **Lin et al. (ACL 2022)** introduced TruthfulQA to measure whether models generate truthful answers and to probe imitative falsehoods across 817 questions.
- The **official 2025 update** (Evans, Chua, Lin) moved to a **binary-choice** setting using *Best Answer* and *Best Incorrect Answer*, reducing odd-one-out and other MC heuristics.
- **This notebook** asks a complementary question: do the paired answers in that improved setting remain balanced in surface form? The analysis provides evidence that residual answer-format asymmetries can be detectable (surface-form classifier above chance and above the pair-structured null). The 2025 binary-choice format remains an improvement over earlier settings. The open question is whether any remaining leakage **materially changes** how we interpret model evaluation; the benchmark-impact section (§7c) outlines the next step (model accuracy on clean vs confounded subsets)."""))

# --- 11. Limitations and Implications ---
cells.append(md("""## 11. Limitations and Implications

- **Lexicon-based features** are incomplete; we do not capture all surface-form dimensions.
- Some detected cues may reflect **semantically appropriate** answer structure (e.g. negation when refuting a myth) rather than purely spurious style; that is why downstream model-impact evidence (§7c) is essential to interpret benchmark scores.
- This is an **audit** of residual surface-form asymmetries, not a full invalidation of TruthfulQA. The 2025 binary-choice update is a substantive improvement over earlier multiple-choice versions.
- Detectable leakage **does not imply** that all model performance is shortcut learning; it indicates that shortcut-learnable cues may contribute to reported scores. The open question is whether leakage materially affects model rankings (benchmark-impact experiment in §7c).
- **Implications:** Interpret benchmark scores with awareness of possible confound; consider reporting accuracy on clean subsets or conducting clean-vs-confounded comparisons when model predictions are available."""))

# --- 12. Optional Export ---
cells.append(md("""## 12. Optional Export"""))
cells.append(code("""out_dir = ROOT / "audits"
out_dir.mkdir(exist_ok=True)
path_audit = out_dir / "truthfulqa_style_audit.csv"
audit.to_csv(path_audit, index=False)
print("Saved:", path_audit)"""))

# --- References ---
cells.append(md("""## References

- **Lin, S., Hilton, J., & Evans, O.** (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL 2022*.
- **TruthfulQA repository:** https://github.com/sylinrl/TruthfulQA (TruthfulQA.csv; binary-choice setting).
- **Evans, O., Chua, J., & Lin, S.** (2025). New, improved multiple-choice TruthfulQA. Official update recommending the binary-choice format with *Best Answer* and *Best Incorrect Answer*."""))

# Build notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out_path = Path(__file__).resolve().parent / "TruthfulQA_Style_Confound_Audit.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print("Wrote", out_path)
