# experiments/exp2_b_learning.py   (CPU or GPU, minutes)
# Q2: why does B collapse under random labels, and does supervision / a smaller lr fix it?
# Trains B's type head (same 2304 -> 128 -> 4 MLP) on cached frozen features under
#   label sets : real (4 watermark types) | planted (strong marker signal) | shuffled (no signal)
#   learners   : supervised CE | RL batched (B's Run 15 update) | RL single-episode (Runs 12-14 update)
#   lr, entropy coefficient grid, 3 seeds, 10,000 "episodes" each (same sample budget as the real runs).
#   python experiments/exp2_b_learning.py --data_dir data

import argparse, itertools, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="data")
ap.add_argument("--episodes", type=int, default=10000)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", default="data/exp2_results.csv")
ap.add_argument("--no_standardise", action="store_true",
                help="feed raw cached features (as the real B did) instead of z-scored ones")
ap.add_argument("--pool", choices=["all", "mean"], default="all",
                help="all = CLS+mean+max (2304-d); mean = mean-pool only (768-d, as in Runs 11-14)")
ap.add_argument("--quick", action="store_true", help="reduced grid (lr 1e-4/3e-4 only)")
args = ap.parse_args()

def load(name, four_class):
    z = np.load(os.path.join(args.data_dir, name))
    X, y, tr, va = z["X"], z["y"], z["tr"], z["va"]
    if args.pool == "mean":
        D = X.shape[1] // 3
        X = X[:, D:2 * D]
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(X), 2000), rng.integers(0, len(X), 2000)
    cos = (X[a] * X[b]).sum(1) / (np.linalg.norm(X[a], axis=1) * np.linalg.norm(X[b], axis=1) + 1e-9)
    print(f"{name}: raw feature dim {X.shape[1]}, mean pairwise cosine similarity between random texts = {cos.mean():.3f}")
    if four_class:                       # drop the 'none' class (label 4)
        keep = y < 4
        tr, va = tr[keep[tr]], va[keep[va]]
    if args.no_standardise:
        X = torch.tensor(X, dtype=torch.float32)
    else:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        X = torch.tensor((X - mu) / sd, dtype=torch.float32)
    return X[tr], torch.tensor(y[tr]), X[va], torch.tensor(y[va])

def head(D):
    return nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, 4))

def evaluate(m, Xv, yv):
    with torch.no_grad():
        lg = m(Xv)
        pred = lg.argmax(-1)
        share = torch.bincount(pred, minlength=4).float() / len(pred)
        p = F.softmax(lg, -1)
        samp = (torch.multinomial(p, 1).squeeze(-1) == yv).float().mean().item()
        ent = float(-(p * torch.log(p + 1e-9)).sum(-1).mean())
    return dict(acc=(pred == yv).float().mean().item(), acc_sampled=samp,
                max_share=share.max().item(), entropy=ent)

def train(kind, data, lr, ent_coef, seed, episodes, clip_eps=0.2):
    Xt, yt, Xv, yv = data
    torch.manual_seed(seed); np.random.seed(seed)
    m = head(Xt.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=lr)
    n, hist, seen, base = len(Xt), [], 0, None
    next_eval = 500
    while seen < episodes:
        if kind == "sup":
            idx = torch.randint(0, n, (8,)); seen += 8
            loss = F.cross_entropy(m(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            bs = 8 if kind == "rl_batch" else 1
            idx = torch.randint(0, n, (bs,)); seen += bs
            x, y = Xt[idx], yt[idx]
            with torch.no_grad():
                lp_all = F.log_softmax(m(x), -1)
                a = torch.multinomial(lp_all.exp(), 1).squeeze(-1)
                old = lp_all.gather(1, a[:, None]).squeeze(1)
            r = (a == y).float()
            if kind == "rl_batch":                       # Run 15 style: batch-normalised advantage
                adv = (r - r.mean()) / r.std(unbiased=False).clamp(min=0.1)
            else:                                        # Runs 12-14 style: EMA baseline
                if base is None:
                    base = r.mean()
                adv = r - base
                base = 0.98 * base + 0.02 * r.mean()
            for _ in range(2):                           # ppo_epochs = 2
                lp = F.log_softmax(m(x), -1)
                new = lp.gather(1, a[:, None]).squeeze(1)
                ratio = torch.exp(new - old)
                obj = torch.min(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
                entropy = -(lp.exp() * lp).sum(-1).mean()
                loss = -obj.mean() - ent_coef * entropy
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if seen >= next_eval:
            hist.append(dict(episodes=seen, **evaluate(m, Xv, yv))); next_eval += 500
    return hist

real, planted = load("features_real.npz", True), load("features_planted.npz", False)
shuf = (real[0], real[1][torch.randperm(len(real[1]), generator=torch.Generator().manual_seed(1))], real[2],
        real[3][torch.randperm(len(real[3]), generator=torch.Generator().manual_seed(2))])
sets = {"real": real, "planted": planted, "shuffled": shuf}

lrs = (1e-4, 3e-4) if args.quick else (1e-5, 1e-4, 3e-4)
grid = [("sup", lr, 0.0) for lr in lrs]
grid += [("rl_batch", lr, e) for lr in lrs for e in ((0.01,) if args.quick else (0.01, 0.05))]
grid += [("rl_single", lr, 0.01) for lr in lrs]

rows = []
for (sname, data), (kind, lr, ent) in itertools.product(sets.items(), grid):
    for seed in range(args.seeds):
        h = train(kind, data, lr, ent, seed, args.episodes)
        last = h[-1]
        rows.append(dict(labels=sname, learner=kind, lr=lr, entropy_coef=ent, seed=seed,
                         first_collapse_ep=next((r["episodes"] for r in h if r["max_share"] > 0.9), np.nan), **last))
    print("done", sname, kind, lr, ent, flush=True)

res = pd.DataFrame(rows); res.to_csv(args.out, index=False)
pd.set_option("display.width", 200)
print("\n=== final val metrics (mean over seeds); chance acc = 0.25 ===")
print(res.groupby(["labels", "learner", "lr", "entropy_coef"])[["acc", "acc_sampled", "max_share", "entropy", "first_collapse_ep"]]
      .mean().round(3).to_string())