"""Sanity checks on the match tracking and the tree model's output. Prints PASS / FAIL only.

    python src/check_tracking.py              # after running the model lane at least once

Run it yourself, locally, in a normal terminal. It deliberately prints no numbers, so its output
can be shared with teammates or an AI coding assistant without sharing anything about the match.
A FAIL names the check; investigate it locally.

Every threshold below is a broad football plausibility bound fixed in advance (pitch size, players
on a pitch, ...). None of them is tuned to this match.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lanes  # noqa: E402
import model as M  # noqa: E402
import possession  # noqa: E402
from games import GAME, GAMES  # noqa: E402

RESULTS = []


def check(name, ok):
    RESULTS.append(bool(ok))
    print("{}  {}".format("PASS" if ok else "FAIL", name), flush=True)


def tracking_checks():
    players, ball, slot_team = possession.load_tracking(None)
    pitch = GAMES[GAME]["pitch"]
    check("frame count matches the tactical video", len(ball) == lanes.N_FRAMES)
    check("every tracker slot has a team, and both teams have slots",
          set(np.unique(slot_team)) == {0, 1})
    pts = players.reshape(-1, 2)
    pts = pts[~np.isnan(pts[:, 0])]
    inside = ((pts[:, 0] > -5) & (pts[:, 0] < pitch[0] + 5) &
              (pts[:, 1] > -5) & (pts[:, 1] < pitch[1] + 5)).mean()
    check("player positions lie on the pitch (metres from a corner, before centring)", inside > 0.99)
    sides = []
    for hi, (start, stop) in enumerate(lanes.GRADED):
        obs = M.to_obs(players, ball, slot_team, pitch, start, stop)
        allp = np.concatenate((obs["home"], obs["away"]), axis=1)
        check("half {}: players centred on the centre spot after the shift".format(hi + 1),
              abs(np.nanmean(allp[..., 0])) < 10 and abs(np.nanmean(allp[..., 1])) < 10)
        n = (~np.isnan(allp[..., 0])).sum(1)
        check("half {}: typical number of tracked players is 15-26".format(hi + 1),
              15 <= np.median(n) <= 26)
        seen = (~np.isnan(obs["ball"][:, 0])).mean()
        check("half {}: ball seen on 20-90% of frames".format(hi + 1), 0.2 <= seen <= 0.9)
        hx, ax = np.nanmean(obs["home"][..., 0]), np.nanmean(obs["away"][..., 0])
        check("half {}: the two teams sit on opposite halves on average".format(hi + 1),
              np.sign(hx) != np.sign(ax))
        sides.append(np.sign(hx))
    check("teams change ends at half-time", sides[0] != sides[1])


def output_checks():
    out_dir = os.path.join(lanes.OUT, "model")
    dead, why_d = lanes.read_dead(os.path.join(out_dir, "dead.csv"))
    team, why_t = lanes.read_team(os.path.join(out_dir, "team.csv"))
    check("model lanes exist and meet the contract", why_d is None and why_t is None)
    if why_d is not None or why_t is not None:
        return
    labels = lanes.merge(dead, team)
    for hi, (start, stop) in enumerate(lanes.GRADED):
        seg = labels[start:stop]
        alive = np.isin(seg, ["H", "A"])
        share_d = (seg == "D").mean()
        share_h = (seg[alive] == "H").mean() if alive.any() else 0.0
        check("half {}: dead ball on 5-60% of frames".format(hi + 1), 0.05 <= share_d <= 0.6)
        check("half {}: each team holds 20-80% of live frames".format(hi + 1),
              0.2 <= share_h <= 0.8)
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], change))
        n_poss = int(np.isin(seg[starts], ["H", "A"]).sum())
        check("half {}: 50-600 possessions".format(hi + 1), 50 <= n_poss <= 600)


if __name__ == "__main__":
    tracking_checks()
    output_checks()
    print("ALL PASS" if all(RESULTS) else "SOME CHECKS FAILED")
