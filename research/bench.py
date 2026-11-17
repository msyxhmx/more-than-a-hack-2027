"""Offline test bench on OTHER matches (open DFL data). Nothing here reads the hackathon match.

Pieces:
  load_halves     clean tracking + H/A/D labels for one DFL match
  degrade         turn clean tracking into "ATD-like" observations (ball outages, noisy anonymous
                  players) under a named noise profile
  features        per-frame, team-swap-symmetric features that work with or without a ball
  baseline_labels the organisers' baseline algorithm (nearest player + majority vote)
  score           official spellF1 / macroF1 via evaluation/possession_metrics.py
"""
import os
import sys

import numpy as np
from scipy.signal import lfilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

NPZ_DIR = os.path.join(ROOT, "data", "external", "dfl_npz")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
H, A, D, X = 0, 1, 2, 3
FPS = 25

# Run-length quantiles of the ball track being present / absent, measured label-free from the
# organisers' PUBLIC baseline_raw.csv (D there == ball not tracked). Percentile -> frames.
GAP_Q = ((0, 1), (8, 4), (10, 67), (25, 154), (50, 268), (75, 529), (90, 882), (95, 1274),
         (99, 2688), (100, 5537))
VIS_Q = ((0, 12), (10, 53), (25, 135), (50, 343), (75, 762), (90, 1622), (95, 2292), (99, 3223),
         (100, 3667))

PROFILES = {
    # name: ball-outage model, onset bias toward contested frames, player noise
    "clean": dict(ball="none", bias=0.0, player=False),
    "iid": dict(ball="iid", bias=0.0, player=True),          # right rate, wrong structure
    "blocks": dict(ball="blocks", bias=0.0, player=True),    # right gap lengths, unbiased onset
    "target": dict(ball="blocks", bias=0.7, player=True),    # gap lengths + contested-biased onset
}
MISSING_RATE = 0.41


def min_touch_filter(lab, tau=6):
    """Absorb alive spells shorter than tau frames into the previous team's spell.

    The challenge text says a touch without control is not a change of possession and that the
    last team keeps the ball while it is loose. tau is fixed a priori, not tuned.
    """
    lab = lab.copy()
    breaks = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [len(lab)]))
    for i in range(1, len(starts)):
        k, prev = lab[starts[i]], lab[starts[i] - 1]
        if k != D and prev != D and prev != k and ends[i] - starts[i] < tau:
            lab[starts[i]:ends[i]] = prev
    return lab


def load_halves(match, tau=6):
    data = np.load(os.path.join(NPZ_DIR, match + ".npz"))
    halves = []
    for h in (0, 1):
        lab = np.where(data["status_%d" % h] == 0, D,
                       np.where(data["poss_%d" % h] == 1, H, A)).astype(np.int8)
        halves.append(dict(ball=data["ball_%d" % h].astype(np.float64),
                           home=data["home_%d" % h].astype(np.float64),
                           away=data["away_%d" % h].astype(np.float64),
                           lab=min_touch_filter(lab, tau)))
    return halves


def _quantile_sampler(table):
    pct = np.array([p for p, _ in table], dtype=float) / 100.0
    logv = np.log(np.array([v for _, v in table], dtype=float))
    return lambda rng: int(round(float(np.exp(np.interp(rng.random_sample(), pct, logv)))))


def _contested(half):
    """Frames where an automatic ball tracker plausibly loses the ball: aerial or crowded."""
    ball = half["ball"]
    everyone = np.concatenate((half["home"], half["away"]), axis=1)
    dist = np.sqrt(((everyone - ball[:, None, :2]) ** 2).sum(-1))
    close_h = np.nanmin(np.where(np.isnan(dist[:, :half["home"].shape[1]]), 99,
                                 dist[:, :half["home"].shape[1]]), axis=1)
    close_a = np.nanmin(np.where(np.isnan(dist[:, half["home"].shape[1]:]), 99,
                                 dist[:, half["home"].shape[1]:]), axis=1)
    crowd = (np.nan_to_num(dist, nan=99.0) < 2.5).sum(1)
    return (ball[:, 2] > 2.0) | (crowd >= 3) | ((close_h < 1.5) & (close_a < 1.5))


def degrade(half, profile, seed):
    """Observations under a noise profile. Returns dict(ball[T,2], home, away) with NaN = unseen."""
    cfg = PROFILES[profile]
    rng = np.random.RandomState(seed)
    T = len(half["lab"])
    ball = half["ball"][:, :2].copy()
    home, away = half["home"].copy(), half["away"].copy()

    # The DFL files freeze the ball (speed exactly 0) on every dead frame -- a provider artefact
    # that would let a model read "dead" straight off the ball. A real automatic tracker keeps
    # following the physical ball while someone fetches and places it, so replace the frozen
    # stretch: hidden for 1 in 5 stoppages, otherwise carried by the player nearest to where it
    # went dead and then resting at the restart spot for the last 2 s. Applied to every profile.
    everyone = np.concatenate((home, away), axis=1)
    dead = np.concatenate(([0], (half["lab"] == D).astype(int), [0]))
    for s, e in zip(np.flatnonzero(np.diff(dead) == 1), np.flatnonzero(np.diff(dead) == -1)):
        if rng.random_sample() < 0.2:
            ball[s:e] = np.nan
            continue
        d = ((everyone[s] - ball[s]) ** 2).sum(-1)
        if np.all(np.isnan(d)):
            continue
        carrier = everyone[s:e, int(np.nanargmin(d))]
        carried = np.where(np.isnan(carrier), ball[s], carrier) + rng.normal(0, 0.3, (e - s, 2))
        rest = max(s, e - 2 * FPS)
        ball[s:rest] = carried[:rest - s]
        ball[rest:e] = ball[e] if e < T else ball[s]

    if cfg["player"]:
        for arr in (home, away):
            noise = rng.normal(0.0, 0.2 * np.sqrt(1 - 0.9 ** 2), arr.shape)
            arr += lfilter([1.0], [1.0, -0.9], noise, axis=0)            # AR(1), sigma 0.2 m
            for p in range(arr.shape[1]):                                # bursty tracklet losses
                t = int(rng.exponential(41 * FPS))
                while t < T:
                    gap = 1 + int(rng.exponential(3 * FPS))
                    arr[t:t + gap, p] = np.nan
                    t += gap + 1 + int(rng.exponential(41 * FPS))

    if cfg["ball"] == "iid":
        ball[rng.random_sample(T) < MISSING_RATE] = np.nan
        ball += rng.normal(0.0, 0.3, ball.shape)
    elif cfg["ball"] == "blocks":
        draw_gap, draw_vis = _quantile_sampler(GAP_Q), _quantile_sampler(VIS_Q)
        contested = np.flatnonzero(_contested(half)) if cfg["bias"] > 0 else np.array([], int)
        missing = np.zeros(T, dtype=bool)
        t = 0 if rng.random_sample() < MISSING_RATE else draw_vis(rng)
        while t < T:
            if len(contested) and rng.random_sample() < cfg["bias"]:
                j = np.searchsorted(contested, t)
                near = [contested[i] for i in (j - 1, j) if 0 <= i < len(contested)]
                near = [c for c in near if abs(c - t) <= 2 * FPS]
                if near:
                    t = int(min(near, key=lambda c: abs(c - t)))
            gap = draw_gap(rng)
            missing[t:t + gap] = True
            t += gap + draw_vis(rng)
        ball[missing] = np.nan
        ball += rng.normal(0.0, 0.3, ball.shape)
    return dict(ball=ball, home=home, away=away)


# -- features ----------------------------------------------------------------------------------

WINDOWS = (12, 25, 50, 125, 250)


def _roll(values, w, future):
    """NaN-aware mean over the w frames strictly before (or after) each frame."""
    ok = ~np.isnan(values)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, values, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(len(values))
    if future:
        lo, hi = np.minimum(idx + 1, len(values)), np.minimum(idx + 1 + w, len(values))
    else:
        lo, hi = np.maximum(idx - w, 0), idx
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)


def _velocity(pos, k=5):
    v = np.full(pos.shape, np.nan)
    v[k:-k] = (pos[2 * k:] - pos[:-2 * k]) / (2.0 * k / FPS)
    return v


def _fill(values, forward):
    idx = np.arange(len(values))
    ok = ~np.isnan(values)
    if forward:
        last = np.maximum.accumulate(np.where(ok, idx, -1))
        return np.where(last >= 0, values[np.clip(last, 0, None)], np.nan)
    nxt = np.minimum.accumulate(np.where(ok, idx, len(values))[::-1])[::-1]
    return np.where(nxt < len(values), values[np.clip(nxt, None, len(values) - 1)], np.nan)


def features(obs, swap=False):
    """[T, F] feature matrix in a canonical frame: team P attacks +x. swap=True makes away P.

    Every feature is computed from the observation only. With swap the same function sees the two
    teams exchanged and the pitch mirrored, which is what makes swap-augmentation and
    test-time symmetrisation exact.
    """
    P, Q = (obs["away"], obs["home"]) if swap else (obs["home"], obs["away"])
    ball = obs["ball"]
    sign = 1.0 if np.nanmean(P[..., 0]) < np.nanmean(Q[..., 0]) else -1.0
    flip = np.array([sign, 1.0])
    P, Q, ball = P * flip, Q * flip, ball * flip
    T = len(ball)
    cols, names = [], []

    def add(name, values):
        names.append(name)
        cols.append(np.asarray(values, dtype=np.float32))

    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    team = {}
    for tag, pos in (("P", P), ("Q", Q)):
        vel = _velocity(pos)
        speed = np.sqrt((vel ** 2).sum(-1))
        team[tag] = (pos, vel, speed)
        xs = np.sort(np.where(np.isnan(pos[..., 0]), np.inf, pos[..., 0]), axis=1)
        n = (~np.isnan(pos[..., 0])).sum(1)
        add(tag + "_cx", np.nanmean(pos[..., 0], 1))
        add(tag + "_cy_abs", np.abs(np.nanmean(pos[..., 1], 1)))
        add(tag + "_sx", np.nanstd(pos[..., 0], 1))
        add(tag + "_sy", np.nanstd(pos[..., 1], 1))
        add(tag + "_cvx", np.nanmean(vel[..., 0], 1))
        add(tag + "_cvy_abs", np.abs(np.nanmean(vel[..., 1], 1)))
        add(tag + "_speed", np.nanmean(speed, 1))
        add(tag + "_speed_max", np.nanmax(np.where(np.isnan(speed), -1, speed), 1))
        add(tag + "_slow", np.nanmean(np.where(np.isnan(speed), np.nan, speed < 1.5), 1))
        add(tag + "_x_second_lo", xs[:, 1])
        add(tag + "_x_lo", xs[:, 0])
        hi = xs[np.arange(T), np.clip(n - 1, 0, None)]
        add(tag + "_x_hi", np.where(np.isinf(hi), np.nan, hi))
        add(tag + "_n", n)
    allpos = np.concatenate((P, Q), axis=1)
    allspeed = np.concatenate((team["P"][2], team["Q"][2]), axis=1)
    allvx = np.concatenate((team["P"][1][..., 0], team["Q"][1][..., 0]), axis=1)
    slow_all = np.nanmean(np.where(np.isnan(allspeed), np.nan, allspeed < 1.5), 1)
    speed_all = np.nanmean(allspeed, 1)
    flow_x = np.nanmean(allvx, 1)
    add("all_slow", slow_all)
    add("all_speed", speed_all)
    add("all_flow_x", flow_x)
    add("cx_gap", cols[names.index("Q_cx")] - cols[names.index("P_cx")])
    add("sx_ratio", cols[names.index("P_sx")] / (cols[names.index("Q_sx")] + 1e-3))
    add("sy_ratio", cols[names.index("P_sy")] / (cols[names.index("Q_sy")] + 1e-3))

    vis = ~np.isnan(ball[:, 0])
    bvel = _velocity(ball, 3)
    bspeed = np.sqrt((bvel ** 2).sum(-1))
    dist = {}
    for tag in ("P", "Q"):
        d = np.sqrt(((team[tag][0] - ball[:, None, :]) ** 2).sum(-1))
        d = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
        dist[tag] = d
        add(tag + "_d1", np.where(vis & np.isfinite(d[:, 0]), d[:, 0], np.nan))
        add(tag + "_d2", np.where(vis & np.isfinite(d[:, 1]), d[:, 1], np.nan))
        add(tag + "_n3", np.where(vis, (d < 3.0).sum(1), np.nan))
        add(tag + "_n8", np.where(vis, (d < 8.0).sum(1), np.nan))
    d_p, d_q = cols[names.index("P_d1")], cols[names.index("Q_d1")]
    ctrl = np.tanh((d_q - d_p) / 2.0)                       # +1 = P clearly nearer, NaN unseen
    near = np.minimum(d_p, d_q)
    margin = np.minimum(52.5 - np.abs(ball[:, 0]), 34.0 - np.abs(ball[:, 1]))
    held = np.where(near < 1.5, np.sign(d_q - d_p), np.where(vis, 0.0, np.nan))
    add("ball_vis", vis)
    add("ball_x", ball[:, 0])
    add("ball_y_abs", np.abs(ball[:, 1]))
    add("ball_margin", margin)
    add("ball_speed", bspeed)
    add("ball_vx", bvel[:, 0])
    add("ctrl", ctrl)
    add("near", near)
    add("held", held)
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(vis, idx, -10 ** 6))
    nxt = np.minimum.accumulate(np.where(vis, idx, 10 ** 7)[::-1])[::-1]
    since = np.minimum(idx - last, 3000)
    until = np.minimum(nxt - idx, 3000)
    add("since_seen", since)
    add("until_seen", until)
    for name, values in (("ctrl", ctrl), ("near", near), ("ball_x", ball[:, 0]),
                         ("ball_margin", margin), ("ball_speed", bspeed)):
        add("last_" + name, _fill(np.asarray(values, float), True))
        add("next_" + name, _fill(np.asarray(values, float), False))
    for w in WINDOWS:
        for name, values in (("ctrl", ctrl), ("held", held), ("near", near), ("vis", vis.astype(float)),
                             ("ball_speed", bspeed), ("ball_margin", margin),
                             ("all_slow", slow_all), ("all_speed", speed_all), ("flow_x", flow_x),
                             ("P_cvx", cols[names.index("P_cvx")]),
                             ("Q_cvx", cols[names.index("Q_cvx")]),
                             ("cx_gap", cols[names.index("cx_gap")]),
                             ("sx_ratio", cols[names.index("sx_ratio")])):
            values = np.asarray(values, float)
            add("%s_past%d" % (name, w), _roll(values, w, False))
            add("%s_next%d" % (name, w), _roll(values, w, True))
    return np.stack(cols, axis=1), names


# -- organisers' baseline, re-implemented on the observation -----------------------------------

def baseline_labels(obs, window=50):
    ball = obs["ball"]
    dh = np.nanmin(np.where(np.isnan(obs["home"][..., 0]), np.inf,
                            ((obs["home"] - ball[:, None]) ** 2).sum(-1)), axis=1)
    da = np.nanmin(np.where(np.isnan(obs["away"][..., 0]), np.inf,
                            ((obs["away"] - ball[:, None]) ** 2).sum(-1)), axis=1)
    codes = np.where(np.isnan(ball[:, 0]), -1, np.where(dh <= da, 0, 1))
    if window <= 0:
        return np.where(codes < 0, D, codes).astype(np.int8)
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    idx = np.arange(len(codes))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(codes), idx + window + 1)
    home, away = hc[hi] - hc[lo], ac[hi] - ac[lo]
    return np.where(home + away == 0, D, np.where(home >= away, H, A)).astype(np.int8)


# -- scoring -----------------------------------------------------------------------------------

def score(gt_halves, pred_halves):
    """Official metrics for one match. Halves are joined with an X gap so nothing straddles."""
    pad = np.full(100, X, dtype=np.int8)
    gt = np.concatenate([np.concatenate((g, pad)) for g in gt_halves])
    pred = np.concatenate([np.concatenate((p.astype(np.int8), pad)) for p in pred_halves])
    res = pm.evaluate(gt, pred)
    return dict(spellF1=res["spell_f1_mean"], macroF1=res["macro_f1"],
                n_gt=res["n_gt_spells"], n_pred=res["n_pred_spells"],
                D_recall=res["per_class"]["D"]["recall"], D_prec=res["per_class"]["D"]["precision"],
                team_acc=res["team_acc_both_alive"])
