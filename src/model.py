"""Tree model + minimum-duration Viterbi (task M01) -> outputs/model/dead.csv + team.csv

    python src/run.py --team-name Lawan --method model     # the normal way, via the pipeline
    python src/model.py                                    # this lane alone

What it does
------------
A. TRAIN on the 7 open DFL matches (other games, CC-BY 4.0; see src/dfl.py). Their clean tracking
   is first degraded to look like an automatic tracker: the ball goes missing in long blocks with
   the gap lengths measured from the organisers' public baseline file, and players get jitter and
   dropouts ("target" profile, research/bench.py). A gradient-boosted tree classifier learns
   H / A / D per frame from ~190 features that work with or without a ball.
B. READ the hackathon match with the team's loader (possession.load_tracking).
C. TRANSLATE it into the training data's shape: metres with the origin at the centre spot (the
   loader measures from a corner), players grouped by team, one array per half.
D. PREDICT per-frame odds for H / A / D, averaged with the team-swapped mirror so neither side is
   favoured.
E. CLEAN UP with a minimum-duration Viterbi decode (switch cost 5, alive spells >= 25 frames, dead
   spells >= 125 frames). That setting was chosen by leave-one-match-out tests on the DFL matches
   and is fixed here; nothing is tuned on the hackathon match or its public 5 minutes.
F. WRITE the two lane files with lanes.write_lanes. Frames outside the two graded halves are X.

Every step is deterministic (fixed seeds, no early stopping), so the same inputs give the same
file. Rules: this file only reads the match data locally; no match data goes anywhere else.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (HERE, os.path.join(ROOT, "research")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import bench as B  # noqa: E402  (research/bench.py: noise simulator + features)
import decoders as dec  # noqa: E402  (research/decoders.py: Viterbi decoders)

TRAIN_PROFILE = "target"
STEP = 5                                   # train on every 5th frame of each half
SWAP_LABEL = np.array([B.A, B.H, B.D])     # label of the same frame with the teams exchanged
HGB_PARAMS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
DECODER = dict(lam_team=5.0, lam_dead=5.0, min_alive=25, min_dead=125, alpha=1.0)
LETTERS = np.array(["H", "A", "D"], dtype=object)


# -- A. train ------------------------------------------------------------------------------------

def train(verbose=True):
    """Fit the frame classifier on all 7 DFL matches under the target noise profile."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    import dfl
    dfl.ensure()
    t0 = time.time()
    xs, ys = [], []
    for mi, match in enumerate(B.MATCHES):
        for hi, half in enumerate(B.load_halves(match)):
            obs = B.degrade(half, TRAIN_PROFILE, 1000 + 10 * mi + hi)   # same seeds as research
            f0, _ = B.features(obs, swap=False)
            f1, _ = B.features(obs, swap=True)
            lab = half["lab"]
            sel = np.arange(hi % STEP, len(lab), STEP)
            xs += [f0[sel], f1[sel]]
            ys += [lab[sel], SWAP_LABEL[lab[sel]]]
        if verbose:
            print("  features for training match {}/{} ({:.0f}s)".format(
                mi + 1, len(B.MATCHES), time.time() - t0), flush=True)
    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(np.concatenate(xs), np.concatenate(ys))
    if verbose:
        print("  model trained ({:.0f}s)".format(time.time() - t0), flush=True)
    return model


# -- C. translate --------------------------------------------------------------------------------

def to_obs(players, ball, slot_team, pitch, start, stop):
    """One half of loader output -> the dict bench.features expects.

    players (T, slots, 2) and ball (T, 2) are in metres from a corner of the pitch; the DFL
    training data is in metres from the centre spot, so shift by half the pitch. slot_team says
    which team each tracker slot belongs to (0 home, 1 away).
    """
    centre = np.array([pitch[0] / 2.0, pitch[1] / 2.0])
    seg_players = players[start:stop].astype(np.float64) - centre
    return dict(ball=ball[start:stop].astype(np.float64) - centre,
                home=seg_players[:, slot_team == 0],
                away=seg_players[:, slot_team == 1])


# -- D + E. predict and clean up -----------------------------------------------------------------

def predict_proba(model, obs):
    """(T, 3) odds for H / A / D, averaged with the team-swapped mirror."""
    f0, _ = B.features(obs, swap=False)
    f1, _ = B.features(obs, swap=True)
    p0 = model.predict_proba(f0)
    p1 = model.predict_proba(f1)[:, SWAP_LABEL]
    return (p0 + p1) / 2.0


def decode(probs):
    """Per-frame odds -> H / A / D codes with the fixed minimum-duration Viterbi."""
    return dec.viterbi_min_dur(probs, DECODER["lam_team"], DECODER["lam_dead"],
                               DECODER["min_alive"], DECODER["min_dead"], DECODER["alpha"])


def label_match(model, players, ball, slot_team, pitch, halves, n_frames):
    """Full-length array of 'H' / 'A' / 'D' inside the given halves and 'X' elsewhere."""
    labels = np.full(n_frames, "X", dtype=object)
    for start, stop in halves:
        obs = to_obs(players, ball, slot_team, pitch, start, stop)
        labels[start:stop] = LETTERS[decode(predict_proba(model, obs))]
    return labels


# -- F. the pipeline entry point -----------------------------------------------------------------

def main(out_dir=None):
    import lanes
    import possession
    from games import GAME, GAMES
    out_dir = out_dir or os.path.join(lanes.OUT, "model")
    print("tree model: training on the 7 DFL matches (a few minutes) ...", flush=True)
    model = train()
    print("tree model: reading the match tracking ...", flush=True)
    players, ball, slot_team = possession.load_tracking(None)   # no cache: always from raw data
    labels = label_match(model, players, ball, slot_team, GAMES[GAME]["pitch"], lanes.GRADED,
                         lanes.N_FRAMES)
    lanes.write_lanes(labels, out_dir)
    print("tree model: wrote {}".format(os.path.relpath(out_dir, ROOT)), flush=True)
    return out_dir


if __name__ == "__main__":
    main()
