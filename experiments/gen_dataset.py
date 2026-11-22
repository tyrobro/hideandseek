# experiments/gen_dataset.py
# Q1 / Q5: build a labelled dataset with the REAL watermark code and a frozen GPT-2.
# Classes: token, syntactic, semantic, positional, none (no watermark).
# Run from the repo root:  python experiments/gen_dataset.py --n 5000 --out data/wm_dataset.jsonl
# Resumable: re-running appends until --n rows exist.

import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from config import CONFIG
from agents.agent_a import AgentA, WatermarkPolicyHead, WatermarkLogitsProcessor
from rewards.reward_functions import compute_perplexity

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--out", default="data/wm_dataset.jsonl")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--top_k", type=int, default=None,
                help="None = whatever the model's generation_config does (the training default); 0 disables top-k")
ap.add_argument("--strength", type=float, default=None, help="wm_strength override (v2 watermarks)")
ap.add_argument("--wm_version", default=None, help="v1 (old) or v2; default = config.py")
args = ap.parse_args()
if args.strength is not None:
    CONFIG["wm_strength"] = args.strength
if args.wm_version is not None:
    CONFIG["wm_version"] = args.wm_version

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
done = 0
if os.path.exists(args.out):
    with open(args.out) as f:
        done = sum(1 for _ in f)
if done >= args.n:
    print(f"{args.out} already has {done} rows"); sys.exit(0)

rng = random.Random(args.seed + done)
random.seed(args.seed + done)          # the token watermark uses the global `random` module
torch.manual_seed(args.seed + done)

classes = list(CONFIG["watermark_types"]) + ["none"]
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
prompts = [x["text"] for x in dataset if len(x["text"].strip()) > 100]

agent = AgentA(dict(CONFIG), args.device)
agent.model.eval()
tok, eos = agent.tokenizer, agent.tokenizer.eos_token_id
max_new = CONFIG["max_gen_length"]

gc = agent.model.generation_config
meta = {"top_k_default": gc.top_k, "top_p_default": gc.top_p, "temperature_default": gc.temperature,
        "top_k_override": args.top_k, "max_new": max_new}
with open(args.out + ".meta.json", "w") as f:
    json.dump(meta, f)
print("generation settings:", meta)

with open(args.out, "a") as fout:
    for i in range(done, args.n):
        label = classes[i % len(classes)]                    # balanced
        prompt = prompts[rng.randrange(len(prompts))][:100]
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(args.device)
        ii, pi = rng.randrange(3), rng.randrange(3)
        intensity = WatermarkPolicyHead.INTENSITY_VALUES[ii]
        pos_frac = WatermarkPolicyHead.POSITION_FRACTIONS[pi]
        wm_start = int(pos_frac * max_new)
        wm_end = min(wm_start + max(5, max_new // 5), max_new - 1)

        procs = []
        if label != "none":
            procs = [WatermarkLogitsProcessor(label, wm_start, wm_end, intensity, agent, max_new)]
        kw = dict(max_new_tokens=max_new, do_sample=True, temperature=1.0,
                  logits_processor=procs, pad_token_id=eos, use_cache=True)
        if args.top_k is not None:
            kw["top_k"] = args.top_k
        with torch.no_grad():
            gen = agent.model.generate(input_ids=inputs["input_ids"],
                                       attention_mask=inputs["attention_mask"], **kw)
        new_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()
        hit_eos = eos in new_ids
        if hit_eos:
            new_ids = new_ids[:new_ids.index(eos)]
        text = tok.decode(new_ids, skip_special_tokens=True)
        n_new = len(new_ids)
        s_tok, e_tok = min(wm_start, n_new), min(wm_end + 1, n_new)
        c0 = len(tok.decode(new_ids[:s_tok], skip_special_tokens=True))
        c1 = len(tok.decode(new_ids[:e_tok], skip_special_tokens=True))
        ppl = compute_perplexity(agent.model, tok, text, device=args.device) if text.strip() else None

        fout.write(json.dumps({
            "i": i, "label": label, "intensity_idx": ii, "intensity": intensity,
            "pos_idx": pi, "pos_frac": pos_frac, "wm_start": wm_start, "wm_end": wm_end,
            "n_new": n_new, "hit_eos": hit_eos, "ppl": ppl, "text": text, "ids": new_ids,
            "char_span": [min(c0, len(text)), min(c1, len(text))],
        }) + "\n")
        if (i + 1) % 100 == 0:
            fout.flush(); print(f"{i + 1}/{args.n}")
print("done")
