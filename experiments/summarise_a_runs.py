# experiments/summarise_a_runs.py  -- Q3 / Q4 tables from runs/exp_*/training_log.csv
import glob, os
import numpy as np, pandas as pd
pd.set_option("display.width", 220)
for f in sorted(glob.glob("runs/exp_*/training_log.csv")):
    d = pd.read_csv(f); d = d[d.braked == False] if "braked" in d else d
    name = f.split("/")[1]
    d["w"] = (d.episode - 1) // 500
    g = d.groupby("w")
    t = pd.DataFrame({
        "ppl_med": g.perplexity.median(), "ppl_p90": g.perplexity.quantile(.9),
        "uniq": g.uniq_ratio.mean(),
        "nov_mean": g.novelty_bonus.mean(), "nov_min": g.novelty_bonus.min(),
    })
    shares = d.groupby("w").chosen_type.value_counts(normalize=True).unstack(fill_value=0)
    t = t.join(shares); t["max_share"] = shares.max(axis=1)
    print(f"\n=== {name}: {len(d)} rows (per 500-episode window) ===")
    print(t.round(3).to_string())
