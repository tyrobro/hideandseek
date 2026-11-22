# tests/sim_detectability.py
# PROXY ONLY. Real texts come from GPT-2, whose next-token distributions are peaked and
# context-dependent, which shrinks every watermark effect relative to this i.i.d. Zipf model.
# Use it to see the *relative* ordering of the four types and a rough upper bound; the real
# decision is made by experiments/gate_watermarks.py on actual GPT-2 generations.
#   python tests/sim_detectability.py
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_watermark_v2 import fake_vocab, V
from environment.watermark_v2 import WatermarkEngine

rng = np.random.default_rng(0)
# Zipf-like base distribution; function words / punctuation sit at the top ranks like in real text
vocab = fake_vocab()
func_ids = list(range(0, 4)) + list(range(10, 10 + 48))
order = func_ids + [i for i in range(V) if i not in set(func_ids)]
base = np.zeros(V); base[order] = 1.0 / np.arange(1, V + 1) ** 1.05
base /= base.sum(); base_logit = torch.tensor(np.log(base), dtype=torch.float32)

def gen(eng, label, inten, start, n=80, win=17):
    toks = []
    for t in range(n):
        lg = base_logit.clone()[None]
        if label != "none" and start <= t < start + win:
            lg, _ = eng.apply(label, lg, t - start, win, inten)
        toks.append(int(torch.multinomial(lg.softmax(-1), 1)))
    return " ".join(f"t{i}" for i in toks)

def run(strength, per_class=150):
    eng = WatermarkEngine(vocab, V, strength=strength)
    classes = ["token", "syntactic", "semantic", "positional"]
    X, y, inten = [], [], []
    for c in classes:
        for _ in range(per_class):
            ii = random.choice([0.3, 0.6, 1.0]); st = random.choice([8, 36, 60])
            X.append(gen(eng, c, ii, st)); y.append(c); inten.append(ii)
    Xv = TfidfVectorizer(token_pattern=r"t\d+", sublinear_tf=True).fit_transform(X)
    pred = cross_val_predict(LogisticRegression(max_iter=2000, C=10), Xv, y, cv=5)
    y = np.array(y); inten = np.array(inten)
    acc = (pred == y).mean()
    per = {c: round(float((pred[y == c] == c).mean()), 2) for c in classes}
    hi = round(float((pred[inten == 1.0] == y[inten == 1.0]).mean()), 2)
    lo = round(float((pred[inten == 0.3] == y[inten == 0.3]).mean()), 2)
    print(f"strength {strength:>3}: 4-class acc {acc:.2f} (chance .25) | recall {per} | intensity 1.0: {hi}  0.3: {lo}")

random.seed(0); torch.manual_seed(0)
for s in (1.0, 2.0, 3.0):
    run(s)
