# experiments/run_cfg.py
# Runs one Trainer in its OWN process with config overrides applied before anything is built
# (so the notebook-override failure of Run 13 cannot happen), then dumps a few samples.
#   python experiments/run_cfg.py NAME '{"lr_backbone_a": 1e-5}'
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

name, over = sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
base = {
    "force_strategy": None, "random_policy": False, "freeze_agent_a": False,
    "ppl_brake": 1e9, "min_unique_ratio": 0.0, "b_mode": "random",
    "num_episodes": 3000, "log_every": 25, "save_every": 10 ** 9,
    "checkpoint_dir": f"runs/exp_{name}/ckpt", "log_path": f"runs/exp_{name}/training_log.csv",
    "plot_path": f"runs/exp_{name}/curves.png",
}
CONFIG.update(base); CONFIG.update(over)
print("EXPERIMENT", name, {k: CONFIG[k] for k in sorted(set(base) | set(over))}, flush=True)

from train import Trainer
tr = Trainer(CONFIG); tr.run()
with open(f"runs/exp_{name}/samples.txt", "w") as f:
    for p in ["The history of the", "In 1998 , the", "The city is located"]:
        for _ in range(2):
            f.write(f"### {p}\n{tr.agent_a.generate(p)[0]}\n\n")
