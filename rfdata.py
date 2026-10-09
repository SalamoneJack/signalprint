"""I/Q data loading and leakage-safe splitting for the OSU LoRa dataset subset.

LEAKAGE GUARANTEE
-----------------
The unit of splitting is the *recording* (one IQ_k.dat file = one independent 20 s capture of one
device on one day). Splits are decided on recording IDs *before* any windowing, and every window is
cut from inside a single recording, so no recording (and no sample) can contribute to two splits.

  * Days 1-4 ("in-day" pool): for every (day, device) pair its 10 recordings are shuffled with a fixed
    seed and assigned 6 -> train, 2 -> val, 2 -> test. Every device/day appears in each split (balanced),
    but each individual recording lives in exactly one split.
  * Day 5 is held out entirely ("cross-day test"). Nothing captured on that day is used for training
    or model selection. This is the stricter, more realistic test: a different session on a different day.

Normalisation is per window (unit RMS), so no statistics are fitted across splits.
`build_splits` asserts the recording sets are pairwise disjoint and writes the split to splits.json.
"""
import json
import os

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "osu_lora_setup1")
N_DEV, N_DAYS, N_TX = 25, 5, 10


def rec_id(day, dev, tx):
    return "D%d_dev%02d_tx%02d" % (day, dev, tx)


def build_splits(seed=0, out_path=None):
    rng = np.random.RandomState(seed)
    splits = {"train": [], "val": [], "test": [], "test_crossday": []}
    for day in range(1, N_DAYS + 1):
        for dev in range(1, N_DEV + 1):
            txs = list(range(1, N_TX + 1))
            if day == 5:
                splits["test_crossday"] += [(day, dev, t) for t in txs]
                continue
            rng.shuffle(txs)
            splits["train"] += [(day, dev, t) for t in txs[:6]]
            splits["val"] += [(day, dev, t) for t in txs[6:8]]
            splits["test"] += [(day, dev, t) for t in txs[8:]]
    # Drop recordings that are not on disk (e.g. Day2/Device9 returns HTTP 403 on the OSU server).
    # Assignment above is unaffected, so the split stays deterministic regardless of what is missing.
    missing = [rec_id(*r) for v in splits.values() for r in v if not os.path.exists(rec_path(*r))]
    for k in splits:
        splits[k] = [r for r in splits[k] if os.path.exists(rec_path(*r))]
    # --- leakage checks ---
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = {rec_id(*r) for r in splits[names[i]]}
            b = {rec_id(*r) for r in splits[names[j]]}
            assert not (a & b), "LEAKAGE: %s and %s share recordings %s" % (names[i], names[j], a & b)
    assert sum(len(v) for v in splits.values()) + len(missing) == N_DAYS * N_DEV * N_TX
    assert all(r[0] == 5 for r in splits["test_crossday"])
    assert all(r[0] != 5 for k in ("train", "val", "test") for r in splits[k])
    if out_path:
        with open(out_path, "w") as f:
            out = {k: [rec_id(*r) for r in v] for k, v in splits.items()}
            out["missing_not_downloadable"] = missing
            json.dump(out, f, indent=1)
    return splits


def rec_path(day, dev, tx, root=ROOT):
    return os.path.join(root, "Day%d" % day, "Device%d" % dev, "IQ_%d.dat" % tx)


def load_recordings(recs, root=ROOT):
    """Return (X, y): X float16 tensor [n_rec, T, 2] (I,Q), y int64 device labels 0..24."""
    arrs, labels = [], []
    for day, dev, tx in recs:
        a = np.fromfile(rec_path(day, dev, tx, root), dtype=np.float32).reshape(-1, 2)
        arrs.append(a)
        labels.append(dev - 1)
    T = min(len(a) for a in arrs)
    X = torch.from_numpy(np.stack([a[:T] for a in arrs]).astype(np.float16))
    return X, torch.tensor(labels, dtype=torch.long)


def _norm(w):
    # w: [B, 2, L] float32. Unit RMS per window (no cross-split statistics).
    rms = w.pow(2).sum(1, keepdim=True).mean(2, keepdim=True).sqrt()
    return w / (rms + 1e-8)


def random_windows(X, y, n, L, gen=None):
    """Sample n windows at random recordings/offsets (training). Returns [n,2,L] float32, labels, rec index."""
    R, T, _ = X.shape
    ri = torch.randint(0, R, (n,), generator=gen)
    off = torch.randint(0, T - L, (n,), generator=gen)
    idx = off[:, None] + torch.arange(L)[None]
    w = X[ri[:, None], idx].float().permute(0, 2, 1)
    return _norm(w), y[ri], ri


def fixed_windows(X, y, L, per_rec):
    """Deterministic, non-overlapping windows evenly spread over each recording (val/test)."""
    R, T, _ = X.shape
    starts = np.linspace(0, T - L, per_rec).astype(np.int64)
    if per_rec * L <= T:  # make them non-overlapping
        starts = np.arange(per_rec) * (T // per_rec)
    idx = torch.from_numpy(starts)[:, None] + torch.arange(L)[None]  # [P, L]
    w = X[:, idx]                       # [R, P, L, 2]
    w = w.reshape(R * per_rec, L, 2).permute(0, 2, 1)
    rec = torch.arange(R).repeat_interleave(per_rec)
    return w, y[rec], rec               # kept fp16; normalised per batch at eval time


def normalise(w):
    return _norm(w.float())
