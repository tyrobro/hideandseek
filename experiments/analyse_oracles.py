# experiments/analyse_oracles.py   (CPU only; needs pandas, numpy, scipy, scikit-learn)
# Q1 / Q5 / part of Q3: does the watermark leave ANY measurable trace in the generated text,
# using oracle knowledge of the biased token sets, and what does each watermark cost in PPL?
#   python experiments/analyse_oracles.py data/wm_dataset.jsonl

import json, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

path = sys.argv[1]
df = pd.read_json(path, lines=True)
try:
    print("generation settings:", json.load(open(path + ".meta.json")))
except Exception:
    pass
print(f"{len(df)} rows; classes: {df.label.value_counts().to_dict()}\n")

SYN = (100, 199)        # ids biased by the syntactic watermark (+1.5*intensity)
SEM = (1010, 4009)      # ids biased by the semantic watermark (+1.0*intensity)

def rate(ids, lo, hi):
    return float(np.mean([lo <= t <= hi for t in ids])) if len(ids) else np.nan

rows = []
for r in df.itertuples():
    ids = r.ids
    win = ids[r.wm_start: r.wm_end + 1]
    out = ids[:r.wm_start] + ids[r.wm_end + 1:]
    rows.append(dict(
        syn_all=rate(ids, *SYN), sem_all=rate(ids, *SEM),
        syn_win=rate(win, *SYN), sem_win=rate(win, *SEM),
        syn_out=rate(out, *SYN), sem_out=rate(out, *SEM),
        uniq=len(set(ids)) / max(1, len(ids)),
    ))
df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
df["logppl"] = np.log(df.ppl.clip(lower=1.0))

# ── A. fluency cost of each watermark (frozen A) ────────────────────────────
print("=== A. PPL by label x intensity (median / p90) ===")
print(df.groupby(["label", "intensity"]).ppl.agg(["median", lambda x: x.quantile(.9), "count"])
        .round(1).rename(columns={"<lambda_0>": "p90"}).to_string())

# ── Q5: PPL vs generation length ────────────────────────────────────────────
print("\n=== Q5. PPL vs generated length (all labels) ===")
df["len_bin"] = pd.cut(df.n_new, [-1, 5, 15, 30, 60, 1000])
print(df.groupby("len_bin", observed=True).ppl.agg(["median", "min", "count"]).round(1).to_string())
print("share with PPL < 5:", round((df.ppl < 5).mean(), 4), "| of those, n_new median:",
      df[df.ppl < 5].n_new.median(), "| hit EOS:", round(df[df.ppl < 5].hit_eos.mean(), 2) if (df.ppl < 5).any() else "n/a")
print("EOS-early share (n_new < 80):", round((df.n_new < 80).mean(), 3))

# ── B. oracle detectability (AUROC vs 'none') ───────────────────────────────
def auroc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5: return np.nan
    return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])

none = df[df.label == "none"]
print("\n=== B. AUROC of oracle features, watermark vs none (0.5 = no signal) ===")
res = []
for lab, feat in [("syntactic", "syn_win"), ("syntactic", "syn_all"),
                  ("semantic", "sem_win"), ("semantic", "sem_all"),
                  ("token", "uniq"), ("token", "logppl"),
                  ("positional", "uniq"), ("positional", "logppl")]:
    for it in sorted(df.intensity.unique()):
        s = df[(df.label == lab) & (df.intensity == it)]
        res.append(dict(label=lab, feature=feat, intensity=it, auroc=round(auroc(s[feat], none[feat]), 3)))
print(pd.DataFrame(res).pivot_table(index=["label", "feature"], columns="intensity", values="auroc").to_string())

print("\n=== B2. positional vs none: two-sample KS on logppl and uniq (identical distributions expected) ===")
p = df[df.label == "positional"]
for f in ["logppl", "uniq", "n_new"]:
    print(f, "KS p =", round(stats.ks_2samp(p[f].dropna(), none[f].dropna()).pvalue, 3))

# ── C. localisation signal for the token-set watermarks ─────────────────────
print("\n=== C. in-window vs out-of-window token-set rate (mean) ===")
for lab, a, b in [("syntactic", "syn_win", "syn_out"), ("semantic", "sem_win", "sem_out")]:
    s = df[df.label == lab]
    print(f"{lab:10s} in-window {s[a].mean():.4f} | out-of-window {s[b].mean():.4f} | none (same window) "
          f"{none[a].mean():.4f}")

# ── D. what accuracy is reachable from simple features? ─────────────────────
print("\n=== D. 5-class accuracy from simple features (5-fold CV) ===")
feats = ["syn_all", "sem_all", "syn_win", "sem_win", "uniq", "logppl", "n_new"]
d = df.dropna(subset=feats).reset_index(drop=True)
y = d.label.values
cv = StratifiedKFold(5, shuffle=True, random_state=0)
pred = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), d[feats].values, y, cv=cv)
labs = sorted(set(y))
print("oracle-feature GBM accuracy:", round((pred == y).mean(), 3), "(chance 0.20)")
print(pd.DataFrame(confusion_matrix(y, pred, labels=labs), index=labs, columns=labs).to_string())

print("\n=== D2. TF-IDF (token-id uni+bigrams) logistic regression, 5-fold CV ===")
docs = [" ".join(map(str, ids)) for ids in d.ids]
X = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), max_features=60000, sublinear_tf=True).fit_transform(docs)
pred2 = cross_val_predict(LogisticRegression(max_iter=300, C=3.0), X, y, cv=cv)
print("accuracy:", round((pred2 == y).mean(), 3))
print(pd.DataFrame(confusion_matrix(y, pred2, labels=labs), index=labs, columns=labs).to_string())
