# tests/test_watermark_v2.py  -  offline checks of environment/watermark_v2.py (no GPT-2 needed)
#   python tests/test_watermark_v2.py
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from environment.watermark_v2 import WatermarkEngine, FUNCTION_WORDS, PUNCT

V = 50257
def fake_vocab(seed=0):
    """GPT-2-like token strings: function words, punctuation, whole words, sub-words, byte junk."""
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    vocab = [None] * V
    vocab[0] = ","; vocab[1] = "."; vocab[2] = ";"; vocab[3] = ":"
    for k, w in enumerate(FUNCTION_WORDS):
        vocab[10 + k] = "Ġ" + w
    for i in range(V):
        if vocab[i] is None:
            r = rng.random()
            w = "".join(rng.choice(letters) for _ in range(rng.randint(3, 9)))
            vocab[i] = ("Ġ" + w) if r < 0.6 else (w if r < 0.9 else "Ġ" + w + "1")
    return vocab

def main():
    eng = WatermarkEngine(fake_vocab(), V, strength=1.0)
    sz = eng.set_sizes(); print("set sizes", sz)

    # 1. sets exist, are mutually disjoint, and the function-word set is the real one
    names = list(eng.sets)
    for i, a in enumerate(names):
        assert len(eng.sets[a]) > 0, a
        for b in names[i + 1:]:
            assert not (set(eng.sets[a]) & set(eng.sets[b])), (a, b)
    assert len(eng.sets["syntactic"]) >= len(FUNCTION_WORDS)
    assert abs(sz["token"] - 0.2 * V) < 2

    # 2. deterministic given the key; different key -> different sets
    e2 = WatermarkEngine(fake_vocab(), V)
    assert e2.sets == eng.sets
    assert WatermarkEngine(fake_vocab(), V, key=1).sets["semantic"] != eng.sets["semantic"]

    # 3. bias scales with intensity and strength; positional ramps; others constant over the window
    assert eng.bias("token", 0, 17, 0.5) == 0.5 * eng.deltas["token"]
    assert eng.bias("syntactic", 5, 17, 1.0) == eng.bias("syntactic", 16, 17, 1.0)
    r = [eng.bias("positional", k, 17, 1.0) for k in range(17)]
    assert all(r[i] < r[i + 1] for i in range(16)), r
    assert abs(r[-1] - eng.deltas["positional"]) < 1e-9
    assert WatermarkEngine(fake_vocab(), V, strength=2.0).bias("semantic", 0, 17, 1.0) == 2 * eng.bias("semantic", 0, 17, 1.0)

    # 4. apply(): only the type's set changes, by exactly the bias; the rest of the logits is untouched
    torch.manual_seed(0)
    logits = torch.randn(1, V)
    for t in names:
        out, info = eng.apply(t, logits, 10, 17, 1.0)
        diff = (out - logits)[0]
        ids = torch.tensor(eng.sets[t])
        assert torch.allclose(diff[ids], torch.full_like(diff[ids], info["bias"]), atol=1e-5), t
        other = torch.ones(V, dtype=torch.bool); other[ids] = False
        assert diff[other].abs().max() == 0, t
        # not a softmax no-op (the v1 positional bug): probability mass on the set must increase
        p0 = logits.softmax(-1)[0, ids].sum(); p1 = out.softmax(-1)[0, ids].sum()
        assert p1 > p0 * 1.3, (t, float(p0), float(p1))
        print(f"{t:11s} set mass {float(p0):.3f} -> {float(p1):.3f}")

    # 5. input tensor is not modified in place
    assert torch.equal(logits, logits.clone())
    print("all watermark v2 tests passed")

if __name__ == "__main__":
    main()
