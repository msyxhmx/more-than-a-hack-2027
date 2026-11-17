"""A hand-written rule system on the same degraded data, as the no-training comparator.

Possession: the team whose player is within 1.5 m of the ball takes over only after holding it for
`hold` consecutive frames (touches and deflections are ignored); with no ball the last team keeps it.
Dead ball: ball tracked outside the pitch (latched until it is back inside), or most players walking
for a sustained stretch. Settings are chosen on the other six matches (cross-fit).
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402
import decoders as dec  # noqa: E402


def rules(F, names, hold, slow_thr):
    col = {n: F[:, i] for i, n in enumerate(names)}
    held, margin, vis = col["held"], col["ball_margin"], col["ball_vis"] > 0.5
    T = len(held)
    team = np.zeros(T, dtype=np.int8)
    cur, cnt = 0, 0
    first = np.flatnonzero(np.nan_to_num(held) != 0)
    cur = 0 if not len(first) or held[first[0]] > 0 else 1
    for t in range(T):
        h = held[t]
        want = -1 if (np.isnan(h) or h == 0) else (0 if h > 0 else 1)
        if want == -1 or want == cur:
            cnt = 0
        else:
            cnt += 1
            if cnt >= hold:
                cur = want
                team[t - hold + 1:t] = cur
                cnt = 0
        team[t] = cur
    dead = np.zeros(T, dtype=bool)
    latched = False
    for t in range(T):
        if vis[t]:
            if margin[t] < -0.5:
                latched = True
            elif margin[t] > 0.5 and latched and col["ball_speed"][t] > 2.0:
                latched = False
        dead[t] = latched
    slow = (np.nan_to_num(col["all_slow_past50"], nan=0) + np.nan_to_num(col["all_slow_next50"], nan=0)) / 2
    dead |= slow > slow_thr
    return np.where(dead, B.D, team).astype(np.int8)


if __name__ == "__main__":
    grid = list(itertools.product((8, 15, 25), (0.6, 0.7, 0.8), (0, 25)))
    scores, full = {}, {}
    for mi, m in enumerate(B.MATCHES):
        halves = B.load_halves(m)
        labs = [h["lab"] for h in halves]
        feats = [B.features(B.degrade(h, "target", 1000 + 10 * mi + hi)) for hi, h in enumerate(halves)]
        for hold, thr, vote in grid:
            pred = [rules(F, names, hold, thr) for F, names in feats]
            if vote:
                pred = [dec.smooth_vote(np.eye(3)[p], vote) for p in pred]
            scores[((hold, thr, vote), m)] = dec.fast_spell_f1(labs, pred)
            full[((hold, thr, vote), m)] = B.score(labs, pred)
    vals, picks = [], []
    for m in B.MATCHES:
        best = max(grid, key=lambda k: np.mean([scores[(k, o)] for o in B.MATCHES if o != m]))
        vals.append(full[(best, m)])
        picks.append(best)
    print("rules, cross-fit:", {k: round(float(np.mean([v[k] for v in vals])), 4) for k in vals[0]})
    print("picks:", picks)
