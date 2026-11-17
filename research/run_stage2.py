"""Refinement stage (the MS-TCN idea with a tree model): a second classifier reads a wide window of
stage-1 probabilities and re-predicts each frame. Stage-1 inputs are held-out predictions.

    python research/run_stage2.py target_target_all
"""
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402

CACHE = os.path.join(B.ROOT, "data", "external", "cache")
OFFSETS = (-250, -125, -75, -50, -25, -12, -6, -3, 0, 3, 6, 12, 25, 50, 75, 125, 250)
SWAP = np.array([1, 0, 2])


def window_features(p):
    T = len(p)
    idx = np.arange(T)
    cols = [p[np.clip(idx + o, 0, T - 1)] for o in OFFSETS]
    cs = np.concatenate((np.zeros((1, 3)), np.cumsum(p, 0)))
    for w in (12, 50, 125):
        lo, hi = np.maximum(idx - w, 0), np.minimum(idx + w + 1, T)
        cols.append((cs[idx] - cs[lo]) / np.maximum(idx - lo, 1)[:, None])        # past mean
        cols.append((cs[hi] - cs[idx + 1]) / np.maximum(hi - idx - 1, 1)[:, None])  # future mean
    return np.concatenate(cols, 1).astype(np.float32)


def main(tag):
    data = {}
    for m in B.MATCHES:
        z = np.load(os.path.join(CACHE, "probs_%s_%s.npz" % (tag, m)))
        data[m] = {k: z[k] for k in z.files}
    for fold, test in enumerate(B.MATCHES):
        xs, ys = [], []
        for m in B.MATCHES:
            if m == test:
                continue
            for hi in (0, 1):
                p, lab = data[m]["p_%d" % hi], data[m]["lab_%d" % hi]
                sel = np.arange((fold + hi) % 3, len(lab), 3)
                xs += [window_features(p)[sel], window_features(p[:, SWAP])[sel]]
                ys += [lab[sel], SWAP[lab[sel]]]
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                               min_samples_leaf=200, early_stopping=False,
                                               random_state=0)
        model.fit(np.concatenate(xs), np.concatenate(ys))
        out = dict(data[test])
        for hi in (0, 1):
            p = data[test]["p_%d" % hi]
            q = (model.predict_proba(window_features(p))
                 + model.predict_proba(window_features(p[:, SWAP]))[:, SWAP]) / 2
            out["p_%d" % hi] = q.astype(np.float32)
        np.savez_compressed(os.path.join(CACHE, "probs_%s_stage2_%s.npz" % (tag, test)), **out)
        res = B.score([out["lab_0"], out["lab_1"]], [out["p_0"].argmax(1), out["p_1"].argmax(1)])
        print("stage2 fold", test, {k: round(float(v), 4) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
