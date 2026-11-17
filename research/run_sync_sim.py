"""Synthetic test of cross-camera sync by player-configuration matching (other matches only).

A fake "broadcast" stream is cut from a DFL half: only players inside a window that follows the
ball are seen, with independent noise, an unknown time offset, invalid stretches (close-ups) and
replays that show footage from earlier. The estimator never sees the offset or the replay flags.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402


def chamfer(bx, tx):
    """Mean distance from each seen broadcast player to the nearest tactical player (same team)."""
    total, n = 0.0, 0
    for team in ("home", "away"):
        b, t = bx[team], tx[team]
        ok = ~np.isnan(b[:, 0])
        if not ok.any():
            continue
        tt = t[~np.isnan(t[:, 0])]
        if not len(tt):
            continue
        d = np.sqrt(((b[ok][:, None] - tt[None]) ** 2).sum(-1)).min(1)
        total += d.sum()
        n += len(d)
    return total / n if n else np.nan


def run(match, hi, true_offset, seed):
    rng = np.random.RandomState(seed)
    half = B.load_halves(match)[hi]
    tact = B.degrade(half, "target", seed)                      # what the tactical ATD would give
    T = len(half["lab"])
    n_b = T - 4000
    src = np.arange(n_b) + true_offset                          # tactical time shown at broadcast frame j
    kind = np.zeros(n_b, dtype=np.int8)                         # 0 live, 1 invalid, 2 replay
    j = int(rng.exponential(20 * 25))
    while j < n_b:
        length = 25 + int(rng.exponential(8 * 25))
        if rng.random_sample() < 0.5:
            kind[j:j + length] = 1
        else:
            kind[j:j + length] = 2
            back = int(rng.uniform(8, 30) * 25)
            src[j:j + length] = np.clip(src[j:j + length] - back, 0, T - 1)
        j += length + int(rng.exponential(20 * 25))
    ball = half["ball"][:, :2]
    cx = np.convolve(ball[:, 0], np.ones(25) / 25, mode="same")
    cy = np.convolve(ball[:, 1], np.ones(25) / 25, mode="same")
    bro = {}
    for team in ("home", "away"):
        p = half[team][src] + rng.normal(0, 0.4, half[team][src].shape)
        seen = (np.abs(p[..., 0] - cx[src][:, None]) < 22) & (np.abs(p[..., 1] - cy[src][:, None]) < 18)
        p[~seen] = np.nan
        p[kind == 1] = np.nan
        bro[team] = p
    usable = np.flatnonzero(kind != 1)
    probe = rng.choice(usable, 150, replace=False)
    offsets = np.arange(0, 4000)
    cost = np.full(len(offsets), np.nan)
    for oi, o in enumerate(offsets):
        vals = [chamfer({k: bro[k][jj] for k in bro}, {k: tact[k][jj + o] for k in tact}) for jj in probe[:40]]
        cost[oi] = np.nanmedian(vals)
    est = int(offsets[np.nanargmin(cost)])
    per_frame = np.array([chamfer({k: bro[k][jj] for k in bro}, {k: tact[k][jj + est] for k in tact})
                          for jj in usable[::10]])
    is_replay = kind[usable[::10]] == 2
    flagged = per_frame > 1.5
    prec = (flagged & is_replay).sum() / max(flagged.sum(), 1)
    rec = (flagged & is_replay).sum() / max(is_replay.sum(), 1)
    sorted_cost = np.sort(cost[~np.isnan(cost)])
    print("%s h%d true offset %d  estimated %d  (error %d frames) | cost at best %.2f m, next-best "
          "offset outside +-5 frames %.2f m | replay flag precision %.3f recall %.3f | live share %.2f"
          % (match, hi, true_offset, est, est - true_offset, sorted_cost[0],
             np.nanmin(np.where(np.abs(offsets - est) > 5, cost, np.inf)), prec, rec,
             (kind == 0).mean()), flush=True)


if __name__ == "__main__":
    for i, (m, hi, off) in enumerate((("J03WMX", 0, 1234), ("J03WOH", 1, 2871), ("J03WR9", 0, 377))):
        run(m, hi, off, 50 + i)
