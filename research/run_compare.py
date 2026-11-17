"""Score saved probabilities from several runs with ONE fixed decoder (min-duration Viterbi,
switch cost 5, min alive 25 frames, min dead 125 frames) on the folds they share."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B  # noqa: E402
import decoders as dec  # noqa: E402

CACHE = os.path.join(B.ROOT, "data", "external", "cache")
FOLDS = ("J03WMX", "J03WOH", "J03WPY", "J03WR9")

for tag in sys.argv[1:]:
    rows = []
    for m in FOLDS:
        path = os.path.join(CACHE, "probs_%s_%s.npz" % (tag, m))
        if not os.path.exists(path):
            continue
        z = np.load(path)
        labs = [z["lab_0"], z["lab_1"]]
        pred = [dec.viterbi_min_dur(z["p_%d" % h], 5, 5, 25, 125, 1.0) for h in (0, 1)]
        res = B.score(labs, pred)
        hard = np.concatenate([z["p_0"].argmax(1), z["p_1"].argmax(1)])
        lab = np.concatenate(labs)
        vis = np.concatenate([z["vis_0"], z["vis_1"]])
        res["frame_acc_seen"] = float((hard == lab)[vis].mean())
        res["frame_acc_unseen"] = float((hard == lab)[~vis].mean()) if (~vis).any() else float("nan")
        rows.append(res)
    if rows:
        print("%-28s n=%d %s" % (tag, len(rows), {k: round(float(np.mean([r[k] for r in rows])), 4)
                                                 for k in rows[0]}), flush=True)
