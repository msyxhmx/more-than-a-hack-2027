"""Compare decoders on saved held-out probabilities (from run_model.py).

    python research/run_decode.py <tag>        e.g. target_target_all

Protocol: every decoder setting is scored on all 7 held-out matches. For each test match the
setting is chosen on the OTHER six matches ("cross-fit"), so no number below was tuned on the match
it is reported on. "oracle" picks the best setting per match with its labels and is only an upper
bound. MBR picks per match WITHOUT labels, by expected spellF1 under posterior samples.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402
import decoders as dec  # noqa: E402

CACHE = os.path.join(B.ROOT, "data", "external", "cache")


def load(tag):
    out = {}
    for m in B.MATCHES:
        z = np.load(os.path.join(CACHE, "probs_%s_%s.npz" % (tag, m)))
        out[m] = ([z["p_0"], z["p_1"]], [z["lab_0"], z["lab_1"]])
    return out


def grid():
    g = {}
    for w in (0, 5, 12, 25, 38, 50, 75, 125):
        g[("vote", w)] = lambda p, w=w: dec.smooth_vote(p, w)
    for alpha, lam in itertools.product((0.1, 0.3, 1.0), (0.5, 1, 2, 5, 10, 20, 40)):
        g[("viterbi", alpha, lam)] = lambda p, a=alpha, l=lam: dec.viterbi(p, l, l, a)
    for lam, ma, md in itertools.product((2, 5, 10), (12, 25, 50), (50, 125)):
        g[("mindur", lam, ma, md)] = lambda p, l=lam, a=ma, d=md: dec.viterbi_min_dur(p, l, l, a, d, 1.0)
    return g


def main(tag):
    data = load(tag)
    g = grid()
    decodes, scores = {}, {}
    t0 = time.time()
    for key, fn in g.items():
        for m, (probs, labs) in data.items():
            decodes[(key, m)] = [fn(p) for p in probs]
            scores[(key, m)] = dec.fast_spell_f1(labs, decodes[(key, m)])
    print("decoded %d settings in %.0fs" % (len(g), time.time() - t0), flush=True)

    def crossfit(keys):
        picked, vals = [], []
        for m in B.MATCHES:
            best = max(keys, key=lambda k: np.mean([scores[(k, o)] for o in B.MATCHES if o != m]))
            picked.append(best)
            vals.append(scores[(best, m)])
        return float(np.mean(vals)), vals, picked

    report = {}
    for family in ("vote", "viterbi", "mindur"):
        keys = [k for k in g if k[0] == family]
        mean, vals, picked = crossfit(keys)
        oracle = float(np.mean([max(scores[(k, m)] for k in keys) for m in B.MATCHES]))
        best_fixed = max(keys, key=lambda k: np.mean([scores[(k, m)] for m in B.MATCHES]))
        report[family] = dict(crossfit=mean, per_match=vals, oracle_per_match=oracle,
                              most_picked=str(max(set(picked), key=picked.count)),
                              best_fixed=str(best_fixed),
                              n_pred=float(np.mean([sum(len(dec.spells(h)) for h in decodes[(best_fixed, m)])
                                                    for m in B.MATCHES])),
                              n_gt=float(np.mean([sum(len(dec.spells(h)) for h in data[m][1])
                                                  for m in B.MATCHES])))
        print(family, json.dumps(report[family]), flush=True)
    mean_all, vals_all, _ = crossfit(list(g))
    print("all families, cross-fit: %.4f" % mean_all, flush=True)

    # MBR: label-free choice among all candidate decodes, per match.
    keys = list(g)
    best_overall = None
    if len(sys.argv) > 2 and sys.argv[2] == "nombr":
        return
    for alpha_s, lam_s in ((0.3, 5), (1.0, 10)):
        chosen, expected_gap = {}, []
        for mi, (m, (probs, labs)) in enumerate(data.items()):
            per_half = [dec.sample_paths(p, lam_s, lam_s, alpha_s, 16, 7 + mi * 2 + hi)
                        for hi, p in enumerate(probs)]
            samples = [[per_half[0][i], per_half[1][i]] for i in range(16)]
            idx, expected = dec.mbr_select([decodes[(k, m)] for k in keys], samples)
            chosen[m] = (keys[idx], scores[(keys[idx], m)], float(expected[idx]))
        mean = float(np.mean([chosen[m][1] for m in B.MATCHES]))
        print("MBR sampler alpha=%.1f lam=%d: actual spellF1 of label-free pick %.4f | "
              "expected %.4f | picks %s" % (alpha_s, lam_s, mean,
                                            np.mean([chosen[m][2] for m in B.MATCHES]),
                                            sorted(set(str(chosen[m][0]) for m in B.MATCHES))),
              flush=True)
        if best_overall is None or mean > best_overall[0]:
            best_overall = (mean, alpha_s, lam_s)
    print("MBR best sampler setting (chosen with labels, so optimistic): %s" % (best_overall,))
    json.dump({k: v for k, v in report.items()},
              open(os.path.join(CACHE, "decode_report_%s.json" % tag), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
