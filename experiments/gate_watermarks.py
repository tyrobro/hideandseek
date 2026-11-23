# experiments/gate_watermarks.py
# GATE before any Run 16 training: are the v2 watermarks (a) detectable by a plain supervised
# classifier on real GPT-2 generations, and (b) cheap enough in fluency?  Picks wm_strength.
#
#   python experiments/gate_watermarks.py --n_per_class 300 --strengths 1 2 3 4
#
# For each strength it generates a balanced dataset (token/syntactic/semantic/positional/none)
# with the real WatermarkLogitsProcessor + frozen GPT-2 (experiments/gen_dataset.py), then reports
#   - 4-class TF-IDF logistic CV accuracy on the watermarked classes (chance 0.25)
#   - recall per class, accuracy at intensity 1.0 vs 0.3
#   - median PPL per class vs 'none', and unique-token ratio
#   - whether the watermark actually landed: in-window share of the class's token set vs 'none'
# PASS = 4-class accuracy >= --min_acc AND every class median PPL <= --max_ppl_ratio x 'none'.
# Recommendation = the smallest passing strength.
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--n_per_class", type=int, default=300)
ap.add_argument("--strengths", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
ap.add_argument("--min_acc", type=float, default=0.50)
ap.add_argument("--max_ppl_ratio", type=float, default=2.5)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--data_dir", default="data")
args = ap.parse_args()
os.makedirs(args.data_dir, exist_ok=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from transformers import GPT2Tokenizer
from config import CONFIG
from environment.watermark_v2 import WatermarkEngine

tok = GPT2Tokenizer.from_pretrained(CONFIG["base_model"])
V = 50257
eng_cache = {}
classes = ["token", "syntactic", "semantic", "positional"]
here = os.path.dirname(os.path.abspath(__file__))
summary = {}

for s in args.strengths:
    out = os.path.join(args.data_dir, f"gate_s{s}.jsonl")
    n_total = args.n_per_class * 5
    subprocess.run([sys.executable, os.path.join(here, "gen_dataset.py"), "--n", str(n_total),
                    "--out", out, "--strength", str(s), "--wm_version", "v2",
                    "--device", args.device], check=True)
    rows = [json.loads(l) for l in open(out)]
    rows = [r for r in rows if r["text"].strip()]
    eng = WatermarkEngine(tok.convert_ids_to_tokens(list(range(V))), V, strength=s)
    sets = {k: set(v) for k, v in eng.sets.items()}

    wm = [r for r in rows if r["label"] in classes]
    X = [r["text"] for r in wm]; y = np.array([r["label"] for r in wm])
    inten = np.array([r["intensity"] for r in wm])
    Xv = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=2).fit_transform(X)
    pred = cross_val_predict(LogisticRegression(max_iter=3000, C=10), Xv, y, cv=5)
    acc = float((pred == y).mean())
    recall = {c: round(float((pred[y == c] == c).mean()), 2) for c in classes}
    acc_hi = float((pred[inten == 1.0] == y[inten == 1.0]).mean())
    acc_lo = float((pred[inten == 0.3] == y[inten == 0.3]).mean())

    # one-token texts give a NaN perplexity; ignore non-finite values (a NaN made strength 3.0 fail by mistake)
    ok_ppl = lambda r: r["ppl"] is not None and np.isfinite(r["ppl"])
    ppl = {c: float(np.median([r["ppl"] for r in rows if r["label"] == c and ok_ppl(r)]))
           for c in classes + ["none"]}
    uniq = {c: float(np.mean([len(set(r["ids"])) / max(1, len(r["ids"])) for r in rows
                              if r["label"] == c and r["ids"]])) for c in classes + ["none"]}
    ratio = {c: ppl[c] / ppl["none"] for c in classes}

    # did the bias land? share of in-window tokens that belong to the class's own set
    land = {}
    for c in classes:
        inw, base = [], []
        for r in rows:
            ids = r["ids"]
            if r["label"] == c and len(ids) > r["wm_start"]:
                w = ids[r["wm_start"]: r["wm_end"] + 1]
                inw.append(np.mean([t in sets[c] for t in w]))
            if r["label"] == "none" and len(ids) > 20:
                base.append(np.mean([t in sets[c] for t in ids]))
        land[c] = (round(float(np.mean(inw)), 3), round(float(np.mean(base)), 3))

    ok = acc >= args.min_acc and max(ratio.values()) <= args.max_ppl_ratio
    summary[s] = dict(acc=acc, ok=ok)
    print(f"\n=== strength {s} ===")
    print(f"4-class acc {acc:.3f} (chance 0.25) | recall {recall}")
    print(f"acc at intensity 1.0: {acc_hi:.3f} | at 0.3: {acc_lo:.3f}")
    print("median PPL:", {c: round(v, 1) for c, v in ppl.items()}, "| ratio vs none:",
          {c: round(v, 2) for c, v in ratio.items()})
    print("unique-token ratio:", {c: round(v, 2) for c, v in uniq.items()})
    print("set share in window vs in 'none' text:", land)
    print("PASS" if ok else "FAIL")

passing = [s for s in args.strengths if summary[s]["ok"]]
print("\n=== gate summary ===")
for s in args.strengths:
    print(f"strength {s}: acc {summary[s]['acc']:.3f} {'PASS' if summary[s]['ok'] else 'FAIL'}")
if passing:
    print(f"\nRECOMMENDED wm_strength = {min(passing)}  (set it in config.py before Run 16)")
else:
    print("\nNO strength passes. Do NOT start Run 16: the watermarks are either undetectable or too "
          "destructive. Send me this output.")
