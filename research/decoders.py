"""Turn per-frame class probabilities into a label sequence. Everything is deterministic.

  smooth_vote      majority vote on the argmax (what the organisers' baseline does)
  viterbi          HMM decode with a switching cost
  viterbi_min_dur  same, plus a minimum length per segment (a small explicit-duration model)
  sample_paths     posterior samples from the same chain (forward filter, backward sample)
  fast_spell_f1    the official spellF1, re-implemented with a sweep so it can be called thousands
                   of times; checked against evaluation/possession_metrics.py in __main__
  mbr_select       pick, from a set of candidate decodes, the one with the highest EXPECTED
                   spellF1 under the posterior samples -- needs no labels
"""
import numpy as np

H, A, D = 0, 1, 2
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def smooth_vote(probs, window):
    hard = probs.argmax(1)
    if window <= 0:
        return hard.astype(np.int8)
    idx = np.arange(len(hard))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(hard), idx + window + 1)
    counts = []
    for k in range(3):
        c = np.concatenate(([0], np.cumsum(hard == k)))
        counts.append(c[hi] - c[lo])
    return np.argmax(np.stack(counts, 1), 1).astype(np.int8)


def _emissions(probs, alpha):
    return alpha * np.log(np.clip(probs.astype(np.float64), 1e-6, 1.0))


def _switch_matrix(lam_team, lam_dead):
    cost = np.array([[0.0, lam_team, lam_dead],
                     [lam_team, 0.0, lam_dead],
                     [lam_dead, lam_dead, 0.0]])
    return -cost


def viterbi(probs, lam_team, lam_dead=None, alpha=1.0):
    e = _emissions(probs, alpha)
    trans = _switch_matrix(lam_team, lam_team if lam_dead is None else lam_dead)
    T = len(e)
    back = np.zeros((T, 3), dtype=np.int8)
    score = e[0].copy()
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_min_dur(probs, lam_team, lam_dead, min_alive, min_dead, alpha=1.0):
    """Best path in which every alive segment lasts >= min_alive frames and every dead one
    >= min_dead (segments cut by the start or end of the half are exempt)."""
    e = _emissions(probs, alpha)
    trans = _switch_matrix(lam_team, lam_dead)
    T = len(e)
    dmin = np.array([min_alive, min_alive, min_dead])
    cum = np.concatenate((np.zeros((1, 3)), np.cumsum(e, 0)))
    NEG = -1e18
    F = np.full((T, 3), NEG)            # best score ending at t in k, k free to end here
    enter = np.full((T, 3), NEG)        # best score of entering k at t (excluding e[t])
    enter_from = np.zeros((T, 3), dtype=np.int8)
    how = np.zeros((T, 3), dtype=np.int8)   # 0 = continued, 1 = just completed the minimum
    enter[0] = 0.0
    enter_from[0] = -1
    for t in range(T):
        if t > 0:
            cand = F[t - 1][:, None] + trans
            np.fill_diagonal(cand, NEG)
            enter[t] = cand.max(0)
            enter_from[t] = cand.argmax(0)
        for k in range(3):
            stay = F[t - 1, k] + e[t, k] if t > 0 else NEG
            s = t - dmin[k] + 1
            if s < 0:
                s = 0 if enter_from[0, k] == -1 else None   # head of the half is exempt
            fresh = NEG
            if s is not None:
                if s == 0:
                    fresh = cum[t + 1, k] - cum[0, k]
                elif enter[s, k] > NEG / 2:
                    fresh = enter[s, k] + cum[t + 1, k] - cum[s, k]
            if stay >= fresh:
                F[t, k], how[t, k] = stay, 0
            else:
                F[t, k], how[t, k] = fresh, 1
    # tail exemption: allow the final segment to be shorter than its minimum
    best, best_k, best_start = NEG, 0, None
    for k in range(3):
        if F[T - 1, k] > best:
            best, best_k, best_start = F[T - 1, k], k, None
        for s in range(max(1, T - dmin[k] + 1), T):
            if enter[s, k] > NEG / 2:
                val = enter[s, k] + cum[T, k] - cum[s, k]
                if val > best:
                    best, best_k, best_start = val, k, s
    path = np.empty(T, dtype=np.int8)
    t, k = T - 1, best_k
    if best_start is not None:
        path[best_start:] = k
        t, k = best_start - 1, enter_from[best_start, k]
    while t >= 0:
        if how[t, k] == 0 and t > 0:
            path[t] = k
            t -= 1
            continue
        s = max(0, t - dmin[k] + 1)
        path[s:t + 1] = k
        if s == 0:
            break
        k = enter_from[s, k]
        t = s - 1
    return path


def sample_paths(probs, lam_team, lam_dead, alpha, n, seed):
    """n posterior samples from the chain whose MAP is viterbi(probs, lam_team, lam_dead, alpha)."""
    rng = np.random.RandomState(seed)
    e = _emissions(probs, alpha)
    trans = _switch_matrix(lam_team, lam_dead)
    T = len(e)
    fwd = np.empty((T, 3))
    fwd[0] = e[0]
    for t in range(1, T):
        m = fwd[t - 1][:, None] + trans
        mx = m.max(0)
        fwd[t] = mx + np.log(np.exp(m - mx).sum(0)) + e[t]
    paths = np.empty((n, T), dtype=np.int8)

    def draw(logits):
        p = np.exp(logits - logits.max(1, keepdims=True))
        c = np.cumsum(p / p.sum(1, keepdims=True), 1)
        return (rng.random_sample((len(p), 1)) > c).sum(1).clip(0, 2).astype(np.int8)

    paths[:, -1] = draw(np.repeat(fwd[-1][None], n, 0))
    for t in range(T - 2, -1, -1):
        paths[:, t] = draw(fwd[t][None, :] + trans[:, paths[:, t + 1]].T)
    return paths


def spells(path):
    """[(start, stop, team)] maximal H or A runs of one half."""
    breaks = np.flatnonzero(path[1:] != path[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(path)]))
    keep = path[starts] != D
    return np.stack((starts[keep], stops[keep], path[starts][keep]), 1)


def _matches(gt, pred):
    """Matched-pair counts at each IoU threshold: greedy by IoU, one-to-one, same team."""
    pairs = []
    j0 = 0
    for i in range(len(gt)):
        gs, ge, gk = gt[i]
        while j0 < len(pred) and pred[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(pred) and pred[j][0] < ge:
            ps, pe, pk = pred[j]
            if pk == gk and pe > gs:
                inter = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                pairs.append((-inter / float(union), i, j))
            j += 1
    pairs.sort()
    out = []
    for thr in IOUS:
        used_g, used_p, tp = set(), set(), 0
        for neg_iou, i, j in pairs:
            if -neg_iou < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            tp += 1
        out.append(tp)
    return np.array(out, dtype=float)


def fast_spell_f1(gt_paths, pred_paths):
    """spellF1 of a match given per-half label arrays (lists of equal length)."""
    tp = np.zeros(len(IOUS))
    n_gt = n_pred = 0
    for g, p in zip(gt_paths, pred_paths):
        gs, ps = spells(g), spells(p)
        n_gt += len(gs)
        n_pred += len(ps)
        tp += _matches(gs, ps)
    if n_gt + n_pred == 0:
        return 0.0
    return float(np.mean(2 * tp / (n_gt + n_pred)))


def mbr_select(candidates, samples):
    """candidates: list of per-match decodes (each a list of per-half arrays).
    samples: list of posterior samples in the same shape. Returns (best index, expected scores)."""
    expected = np.array([np.mean([fast_spell_f1(s, c) for s in samples]) for c in candidates])
    return int(expected.argmax()), expected


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "evaluation"))
    import possession_metrics as pm
    rng = np.random.RandomState(0)
    worst = 0.0
    for trial in range(200):
        T = 3000
        def rand_path():
            out, t = np.empty(T, dtype=np.int8), 0
            while t < T:
                n = 1 + int(rng.exponential(rng.choice([5, 40, 150])))
                out[t:t + n] = rng.randint(0, 3)
                t += n
            return out
        g, p = rand_path(), rand_path()
        if trial % 2:
            p = g.copy()
            flips = rng.randint(0, T, 30)
            for f in flips:
                p[f:f + rng.randint(1, 60)] = rng.randint(0, 3)
        official = pm.spell_scores(g, p, [(0, T)])["mean_f1"]
        worst = max(worst, abs(official - fast_spell_f1([g], [p])))
        probs = rng.dirichlet([0.3, 0.3, 0.3], T)
        a = viterbi_min_dur(probs, 2.0, 3.0, 1, 1)
        b = viterbi(probs, 2.0, 3.0)
        assert np.array_equal(a, b), "min-duration decode with minimum 1 must equal plain Viterbi"
        c = viterbi_min_dur(probs, 1.0, 1.0, 12, 20)
        br = np.flatnonzero(c[1:] != c[:-1]) + 1
        seg = np.diff(np.concatenate(([0], br, [T])))
        kinds = c[np.concatenate(([0], br))]
        inner = slice(1, len(seg) - 1)
        assert all(n >= (20 if k == D else 12) for n, k in zip(seg[inner], kinds[inner])), "min dur"
    print("fast_spell_f1 max |difference| vs official over 200 random cases: %.2e" % worst)
    print("viterbi_min_dur checks passed")
