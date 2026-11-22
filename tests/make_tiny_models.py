# tests/make_tiny_models.py - offline stand-ins (random tiny GPT-2 + DistilBERT with local tokenizers)
# so the whole Trainer can be smoke-tested without downloading anything.
#   python tests/make_tiny_models.py /path/to/outdir
import json, os, random, sys
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DistilBertConfig,
                          DistilBertModel, DistilBertTokenizerFast)

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.watermark_v2 import FUNCTION_WORDS

def make(out):
    rng = random.Random(0)
    letters = "abcdefghijklmnopqrstuvwxyz"
    content = sorted({"".join(rng.choice(letters) for _ in range(rng.randint(4, 8))) for _ in range(500)})
    words = FUNCTION_WORDS + content

    # ---- GPT-2 style byte-level BPE with whole-word tokens "Ġword" ----
    b2u = bytes_to_unicode()
    vocab = {}
    for b in range(256):
        vocab[b2u[b]] = len(vocab)
    merges = []
    for w in words:
        parts = ["Ġ"] + list(w)
        cur = parts[0]
        for ch in parts[1:]:
            merges.append((cur, ch)); cur = cur + ch
            if cur not in vocab:
                vocab[cur] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    g = os.path.join(out, "gpt2"); os.makedirs(g, exist_ok=True)
    json.dump(vocab, open(os.path.join(g, "vocab.json"), "w"))
    seen, lines = set(), ["#version: 0.2"]
    for a, b in merges:
        if (a, b) not in seen:
            seen.add((a, b)); lines.append(f"{a} {b}")
    open(os.path.join(g, "merges.txt"), "w").write("\n".join(lines) + "\n")
    tok = GPT2Tokenizer(os.path.join(g, "vocab.json"), os.path.join(g, "merges.txt"))
    tok.save_pretrained(g)
    cfg = GPT2Config(vocab_size=len(vocab), n_positions=256, n_embd=64, n_layer=2, n_head=2,
                     bos_token_id=vocab["<|endoftext|>"], eos_token_id=vocab["<|endoftext|>"])
    GPT2LMHeadModel(cfg).save_pretrained(g)

    # ---- DistilBERT ----
    d = os.path.join(out, "distil"); os.makedirs(d, exist_ok=True)
    wp = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(letters) + ["##" + c for c in letters] \
         + list(".,;:'\"-!?0123456789") + [w for w in words]
    open(os.path.join(d, "vocab.txt"), "w").write("\n".join(dict.fromkeys(wp)) + "\n")
    dtok = DistilBertTokenizerFast(os.path.join(d, "vocab.txt"), do_lower_case=True)
    dtok.save_pretrained(d)
    DistilBertModel(DistilBertConfig(vocab_size=len(dict.fromkeys(wp)), dim=64, n_layers=2, n_heads=2,
                                     hidden_dim=128, max_position_embeddings=512)).save_pretrained(d)
    return g, d

if __name__ == "__main__":
    print(make(sys.argv[1]))
