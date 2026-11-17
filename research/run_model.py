"""Leave-one-match-out frame classifier on degraded DFL data. Saves per-frame probabilities.

    python research/run_model.py <train_profile> <test_profile> [feature_set] [folds]

feature_set: all | noball (player-only features, to measure what survives a ball outage)
Probabilities are symmetrised: the model is run on the observation and on its team-swapped mirror
and the two answers are averaged, so H and A are treated identically by construction.
"""
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402

CACHE = os.path.join(B.ROOT, "data", "external", "cache")
STEP = 5          # train on every 5th frame; predict on all
SWAP_LABEL = np.array([B.A, B.H, B.D])


def obs_for(match, profile):
    halves = B.load_halves(match)
    mi = B.MATCHES.index(match)
    return halves, [B.degrade(h, profile, 1000 + 10 * mi + hi) for hi, h in enumerate(halves)]


def column_mask(names, feature_set):
    if feature_set == "all":
        return np.ones(len(names), dtype=bool)
    ball_words = ("ball", "ctrl", "near", "held", "_d1", "_d2", "_n3", "_n8", "seen", "vis")
    return np.array([not any(w in n for w in ball_words) for n in names])


def run(train_profile, test_profile, feature_set="all", folds=None):
    os.makedirs(CACHE, exist_ok=True)
    feats = {}
    for profile in {train_profile, test_profile}:
        for match in B.MATCHES:
            halves, obs = obs_for(match, profile)
            for hi, (half, o) in enumerate(zip(halves, obs)):
                f0, names = B.features(o, swap=False)
                f1, _ = B.features(o, swap=True)
                feats[(profile, match, hi)] = (f0, f1, half["lab"])
    keep = column_mask(names, feature_set)
    results = []
    for fold, test in enumerate(B.MATCHES):
        if folds is not None and fold not in folds:
            continue
        t0 = time.time()
        xs, ys = [], []
        for match in B.MATCHES:
            if match == test:
                continue
            for hi in (0, 1):
                f0, f1, lab = feats[(train_profile, match, hi)]
                sel = np.arange((fold + hi) % STEP, len(lab), STEP)
                xs += [f0[sel][:, keep], f1[sel][:, keep]]
                ys += [lab[sel], SWAP_LABEL[lab[sel]]]
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
                                               min_samples_leaf=200, l2_regularization=1.0,
                                               early_stopping=False, random_state=0)
        model.fit(np.concatenate(xs), np.concatenate(ys))
        out = {}
        for hi in (0, 1):
            f0, f1, lab = feats[(test_profile, test, hi)]
            p0 = model.predict_proba(f0[:, keep])
            p1 = model.predict_proba(f1[:, keep])[:, SWAP_LABEL]
            out["p_%d" % hi] = ((p0 + p1) / 2).astype(np.float32)
            out["lab_%d" % hi] = lab
            out["vis_%d" % hi] = f0[:, names.index("ball_vis")] > 0.5
        np.savez_compressed(os.path.join(CACHE, "probs_%s_%s_%s_%s.npz" % (
            train_profile, test_profile, feature_set, test)), **out)
        pred = [out["p_%d" % hi].argmax(1) for hi in (0, 1)]
        res = B.score([out["lab_0"], out["lab_1"]], pred)
        lab = np.concatenate([out["lab_0"], out["lab_1"]])
        vis = np.concatenate([out["vis_0"], out["vis_1"]])
        hard = np.concatenate(pred)
        res["acc_seen"] = float((hard == lab)[vis].mean())
        res["acc_unseen"] = float((hard == lab)[~vis].mean()) if (~vis).any() else float("nan")
        results.append(res)
        print("fold %s  %s  (%.0fs)" % (test, {k: round(float(v), 4) for k, v in res.items()},
                                         time.time() - t0), flush=True)
    print("MEAN train=%s test=%s feats=%s argmax: %s" % (
        train_profile, test_profile, feature_set,
        {k: round(float(np.mean([r[k] for r in results])), 4) for k in results[0]}), flush=True)


if __name__ == "__main__":
    folds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "all", folds)
