# experiments/probe_classifiers.py   (GPU)
# Q1: supervised detectability with B's own backbone. No RL.
#   (a) frozen DistilBERT features + logistic regression (mean-pool vs CLS/mean/max)
#   (b) fine-tuned DistilBERT with cross-entropy
# Also saves cached frozen features for exp2_b_learning.py, including a PLANTED-SIGNAL control
# (unwatermarked texts with a class-specific marker phrase appended).
#   python experiments/probe_classifiers.py --data data/wm_dataset.jsonl --out_dir data

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertModel, DistilBertTokenizerFast, DistilBertForSequenceClassification
from config import CONFIG

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/wm_dataset.jsonl")
ap.add_argument("--out_dir", default="data")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--skip_finetune", action="store_true")
args = ap.parse_args()

NAME = CONFIG["detector_model"]
classes = list(CONFIG["watermark_types"]) + ["none"]
df = pd.read_json(args.data, lines=True)
df = df[df.text.str.strip().str.len() > 0].reset_index(drop=True)
texts, y = df.text.tolist(), df.label.map({c: i for i, c in enumerate(classes)}).values
tr_idx, va_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0, stratify=y)
tok = DistilBertTokenizerFast.from_pretrained(NAME)

@torch.no_grad()
def featurize(txts, bs=64):
    enc = DistilBertModel.from_pretrained(NAME, attn_implementation="eager").to(args.device).eval()
    out = []
    for i in range(0, len(txts), bs):
        b = tok(txts[i:i + bs], return_tensors="pt", truncation=True, max_length=128, padding=True).to(args.device)
        h = enc(**b).last_hidden_state
        am = b["attention_mask"]
        m = am.clone(); m[:, 0] = 0
        m[torch.arange(len(m)), am.sum(1) - 1] = 0                      # drop [CLS] and [SEP]
        mf = m.unsqueeze(-1).float()
        mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
        mx = h.masked_fill(m.unsqueeze(-1) == 0, -1e4).max(1).values
        out.append(torch.cat([h[:, 0], mean, mx], -1).float().cpu())
    return torch.cat(out).numpy()

def report(name, y_true, y_pred, labs):
    acc = (y_true == y_pred).mean()
    rec = {labs[c]: round(float((y_pred[y_true == c] == c).mean()), 3) for c in range(len(labs)) if (y_true == c).any()}
    print(f"\n[{name}] val accuracy {acc:.3f} (chance {1 / len(labs):.2f}) | recall {rec}")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=range(len(labs))), index=labs, columns=labs).to_string())

# ── (a) frozen features ─────────────────────────────────────────────────────
print("featurising real texts ...")
X = featurize(texts)
np.savez(os.path.join(args.out_dir, "features_real.npz"), X=X, y=y, tr=tr_idx, va=va_idx)
D = X.shape[1] // 3
for pool, sl in [("mean only", slice(D, 2 * D)), ("cls+mean+max", slice(0, 3 * D))]:
    for k, labs in [("5-class", classes), ("4-class (no 'none')", classes[:4])]:
        keep = np.ones(len(y), bool) if k == "5-class" else y < 4
        tri, vai = tr_idx[keep[tr_idx]], va_idx[keep[va_idx]]
        sc = StandardScaler().fit(X[tri][:, sl])
        clf = LogisticRegression(max_iter=500, C=0.1).fit(sc.transform(X[tri][:, sl]), y[tri])
        report(f"frozen {pool}, {k}", y[vai], clf.predict(sc.transform(X[vai][:, sl])), labs)

# ── planted-signal control ──────────────────────────────────────────────────
MARK = [" zebra quartz harbor", " violin ember lattice", " orchard pixel monsoon", " granite saffron comet"]
base = df[df.label == "none"].text.tolist()
pt = [t + MARK[i % 4] for i, t in enumerate(base)]
py = np.array([i % 4 for i in range(len(base))])
Xp = featurize(pt)
ptr, pva = train_test_split(np.arange(len(py)), test_size=0.2, random_state=0, stratify=py)
np.savez(os.path.join(args.out_dir, "features_planted.npz"), X=Xp, y=py, tr=ptr, va=pva)
sc = StandardScaler().fit(Xp[ptr])
clf = LogisticRegression(max_iter=500, C=0.1).fit(sc.transform(Xp[ptr]), py[ptr])
report("PLANTED control, frozen cls+mean+max, logistic", py[pva], clf.predict(sc.transform(Xp[pva])), classes[:4])

# ── (b) fine-tuned DistilBERT ───────────────────────────────────────────────
if not args.skip_finetune:
    torch.manual_seed(0)
    model = DistilBertForSequenceClassification.from_pretrained(NAME, num_labels=len(classes),
                                                                attn_implementation="eager").to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    def batches(idx, bs, shuffle):
        idx = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            b = tok([texts[k] for k in j], return_tensors="pt", truncation=True, max_length=128, padding=True).to(args.device)
            yield b, torch.tensor(y[j]).to(args.device)
    for ep in range(args.epochs):
        model.train()
        for b, yy in batches(tr_idx, 16, True):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(**b).logits.float(), yy)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); preds = []
        with torch.no_grad():
            for b, yy in batches(va_idx, 64, False):
                preds.append(model(**b).logits.argmax(-1).cpu())
        report(f"fine-tuned DistilBERT, epoch {ep + 1}", y[va_idx], torch.cat(preds).numpy(), classes)
print("\nfeatures saved to", args.out_dir)
