# tests/smoke_trainer.py - end-to-end CPU smoke test of train.py with tiny offline models.
#   python tests/smoke_trainer.py            (from the repo root)
# Checks the plumbing (phases, warm-up gate, B supervised loss, spans, final eval, checkpoint/resume),
# NOT learning quality.
import json, os, random, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from make_tiny_models import make

tmp = tempfile.mkdtemp(prefix="smoke_")
g, d = make(tmp)

import train
rng = random.Random(1)
WORDS = ["the", "of", "and", "river", "history", "market", "garden", "stone", "winter", "bridge", "open"]
train.load_dataset = lambda *a, **k: [{"text": " ".join(rng.choice(WORDS) for _ in range(40))} for _ in range(60)]

from config import CONFIG
CONFIG.update({
    "base_model": g, "detector_model": d, "num_episodes": 60, "b_warmup_episodes": 24,
    "b_update_batch": 4, "log_every": 5, "save_every": 30, "sample_every": 30,
    "final_eval_episodes": 12, "warmup_min_acc": 0.0, "wm_strength": 2.0,
    "checkpoint_dir": f"{tmp}/ckpt", "log_path": f"{tmp}/res/training_log.csv",
    "plot_path": f"{tmp}/res/curves.png",
})
tr = train.Trainer(CONFIG); tr.run()

df = pd.read_csv(CONFIG["log_path"])
print(df[["episode", "phase", "type_correct", "iou", "perplexity", "uniq_ratio", "max_share20"]].tail(6))
assert set(df.phase.dropna()) == {"warmup", "game"}, df.phase.unique()
assert not tr.aborted
ev = json.load(open(f"{tmp}/res/final_eval.json")); print({k: ev[k] for k in ["n", "type_acc", "iou_model", "iou_full_text", "iou_best_fixed_window"]})
assert ev["n"] > 0 and os.path.exists(f"{tmp}/res/samples.txt")
assert tr.agent_a.backbone_frozen and not any(p.requires_grad for p in tr.agent_a.model.parameters())
assert tr.agent_b.norm.count > 0 and not tr.agent_b.norm.track
print("norm count", float(tr.agent_b.norm.count))

# resume from the checkpoint and run a few more episodes
CONFIG["num_episodes"] = 66
tr2 = train.Trainer(CONFIG); print("resume start episode:", tr2.start_episode); tr2.run()

# warm-up gate must abort when B cannot reach the required accuracy
CONFIG.update({"warmup_min_acc": 0.99, "num_episodes": 60, "checkpoint_dir": f"{tmp}/ckpt2",
               "log_path": f"{tmp}/res2/training_log.csv"})
tr3 = train.Trainer(CONFIG); tr3.run()
assert tr3.aborted, "warm-up gate did not abort"
shutil.rmtree(tmp)
print("\nSMOKE TEST PASSED")
