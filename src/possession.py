"""Possession model (owner: Syahmi) -> outputs/team.csv

Contract: columns `frame,team`, 199,319 rows, frame 0..199318 in order, team = H, A or X.

    python src/possession.py                          # tracking + Smart Tagging set pieces (v2)
    python src/possession.py --no-set-pieces          # v1: tracking only
    python src/possession.py --dead outputs/dead.csv  # also use Ilham's stoppages (better restarts)

Idea
----
The baseline gives the ball to the nearest player on every frame and then smooths. That flickers on
every deflection and every loose ball, and the smoothing then blurs the real hand-overs by a second
or more. spellF1 punishes both: invented possessions are false positives, and late transitions fail
the IoU thresholds.

This model is a small state machine instead:

1. Ball. Short tracking gaps (<= BALL_GAP_FILL frames) are filled by linear interpolation. Longer
   gaps stay missing, and missing frames are neutral: they neither confirm nor challenge anyone.
2. Evidence. On a frame with a ball, a team has "control evidence" when its nearest player is
   within CONTROL_RADIUS metres of the ball and closer than the nearest opponent by at least
   CONTEST_MARGIN. Contested frames (both close) are neutral.
3. Switch. The team in possession keeps the ball until the other team shows controlled possession:
   either HOLD_FRAMES evidence frames inside a CHALLENGE_WINDOW without the holder touching the
   ball in between, or evidence from >= 2 different players of that team (a completed pass), each
   with at least TOUCH_FRAMES frames. A single touch or deflection never switches.
4. Timing. A switch is back-dated to the first evidence frame of the challenge, so the transition
   lands where the opponent first got the ball, not HOLD_FRAMES later.
5. Restarts (only with --dead). When a stoppage ends, the state is cleared; the first team to show
   TOUCH_FRAMES of evidence takes the ball, back-dated to the restart frame.
6. Set pieces (v2, task P04). Smart Tagging (Metrica's automatic tagging) marks each set piece as
   a ~5 s window tagged with the team taking it. Inside that window the tagged team holds the ball
   and the other team's evidence is ignored, so the restarting team is right even when an opponent
   stands next to the ball (throw-ins, free kicks). After a --dead stoppage, the set piece that ends
   it names the new holder.
7. Clean-up. A possession shorter than MIN_SPELL_FRAMES sandwiched between two spells of the other
   team is absorbed into them (A-H-A -> A-A-A).

Missing-ball gaps: by default the holder's label is kept. Under spellF1 an X is filled with the
previous label anyway, so holding and X score the same there, and holding can only help macroF1.
--x-on-gaps writes X on gaps longer than X_GAP_FRAMES instead, if you prefer to be explicit.

Data rule: this file only reads the data locally. Nothing from the match goes anywhere else.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402

# Params -- set from football reasoning, NOT tuned on the public 5 minutes (rules).
FPS = 25
BALL_GAP_FILL = 12        # frames (0.5 s): interpolate ball gaps up to this long
CONTROL_RADIUS = 2.0      # m: a player this close to the ball can be controlling it
CONTEST_MARGIN = 0.75     # m: nearest player must beat the nearest opponent by this much
HOLD_FRAMES = 10          # frames (0.4 s) of evidence for the other team to take the ball
TOUCH_FRAMES = 3          # frames: minimum evidence per player for the "completed pass" rule
CHALLENGE_WINDOW = 75     # frames (3 s): evidence older than this no longer counts
MIN_SPELL_FRAMES = 25     # frames (1 s): shorter sandwiched possessions are absorbed
SET_PIECE_HOLD = 125      # frames (5 s): protect the restarting team for at most this long
X_GAP_FRAMES = 50         # frames (2 s): with --x-on-gaps, longer ball gaps are written as X

GRADED = ((11185, 90652), (110808, 190198))  # half-open [start, stop), from DATA.md
HOME, AWAY, NONE = 0, 1, -1
OUT_DEFAULT = os.path.join(ROOT, "outputs", "team.csv")
CACHE_DEFAULT = os.path.join(ROOT, "outputs", "atd_cache.npz")


# -- reading -------------------------------------------------------------------------------------

def _xy(group):
    """(x, y) of one 'x,y,speed' group, or (nan, nan)."""
    if not group or group[0] == "N":
        return np.nan, np.nan
    parts = group.split(",")
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return np.nan, np.nan


def read_atd(raw_path, n_slots, n_frames, pitch):
    """players (n_frames, n_slots, 2) and ball (n_frames, 2) in metres, NaN where missing."""
    length, width = pitch
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD frame_id counts from 1
            if not 0 <= frame < n_frames:
                continue
            groups = parts[1].split(";")
            for slot in range(min(len(groups), n_slots)):
                players[frame, slot] = _xy(groups[slot])
            ball[frame] = _xy(parts[2])
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return players, ball


def load_tracking(cache_path=CACHE_DEFAULT, refresh=False):
    """Tracking arrays plus slot teams, cached locally in outputs/ (gitignored)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    slot_team = team_of_each_track(meta_path)
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path))], dtype=np.int64)
    if cache_path and os.path.isfile(cache_path) and not refresh:
        cached = np.load(cache_path)
        if "stamp" in cached and np.array_equal(cached["stamp"], stamp):
            return cached["players"], cached["ball"], slot_team
        print("cache is stale (raw ATD changed), re-reading")
    n_frames = source_frames("tactical")
    print("reading {} (first run only, ~1-2 min) ...".format(os.path.basename(raw_path)))
    players, ball = read_atd(raw_path, len(slot_team), n_frames, GAMES[GAME]["pitch"])
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, players=players, ball=ball, stamp=stamp)
    return players, ball, slot_team


def read_dead(path, n_frames):
    """Boolean dead mask from Ilham's dead.csv, or None if the file is missing."""
    if not path:
        return None
    if not os.path.isfile(path):
        print("warning: {} not found, running without stoppages".format(path))
        return None
    import pandas as pd
    table = pd.read_csv(path)
    table.columns = [c.strip().lower() for c in table.columns]
    if len(table) != n_frames or list(table.columns[:2]) != ["frame", "dead"]:
        raise ValueError("{} does not meet the contract (frame,dead x {})".format(path, n_frames))
    return table["dead"].values.astype(int) == 1


# -- the model -----------------------------------------------------------------------------------

def fill_ball_gaps(ball, max_gap):
    """Linear interpolation over missing-ball runs of at most max_gap frames, bounded both sides."""
    ball = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    idx = np.flatnonzero(seen)
    if len(idx) < 2:
        return ball, seen
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            t = (np.arange(a + 1, b) - a) / float(b - a)
            ball[a + 1:b] = ball[a] + (ball[b] - ball[a]) * t[:, None]
    return ball, ~np.isnan(ball[:, 0])


def control_evidence(players, ball, slot_team):
    """Per frame: team with control evidence (HOME/AWAY/NONE) and the slot of that player."""
    n = len(ball)
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)  # NaN where either is missing
    dist = np.where(np.isnan(dist), np.inf, dist)
    home_cols = np.flatnonzero(slot_team == HOME)
    away_cols = np.flatnonzero(slot_team == AWAY)
    dh, da = dist[:, home_cols], dist[:, away_cols]
    nh, na = dh.min(axis=1), da.min(axis=1)
    sh = home_cols[dh.argmin(axis=1)]
    sa = away_cols[da.argmin(axis=1)]

    team = np.full(n, NONE, dtype=np.int8)
    slot = np.full(n, -1, dtype=np.int16)
    home_ctrl = (nh <= CONTROL_RADIUS) & (nh + CONTEST_MARGIN <= na)
    away_ctrl = (na <= CONTROL_RADIUS) & (na + CONTEST_MARGIN <= nh)
    team[home_ctrl], slot[home_ctrl] = HOME, sh[home_ctrl]
    team[away_ctrl], slot[away_ctrl] = AWAY, sa[away_ctrl]
    return team, slot


def set_piece_owner(n_frames, home, away):
    """Per frame: the team taking a set piece (HOME/AWAY), NONE elsewhere. From Smart Tagging.

    Smart Tagging is Metrica's AUTOMATIC annotation (DATA.md), not human data about the game. Each
    SET PIECES instance (throw-in, corner, goal kick, free kick, kick-off) is a ~5 s window around
    a restart, tagged with the team taking it. Instances tagged N/A are skipped. Only the first
    SET_PIECE_HOLD frames of a window are protected, so a quick turnover after the restart still
    switches through the normal challenge rules.
    """
    from load_smart_tagging import load as load_tagging
    owner = np.full(n_frames, NONE, dtype=np.int8)
    names = {home.strip().lower(): HOME, away.strip().lower(): AWAY}
    for event in load_tagging():
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        team = names.get((event["tags"].get("Team") or "").strip().lower())
        if team is None:
            continue
        a = max(0, event["start_frame"])
        b = min(n_frames, max(event["end_frame"], a + 1), a + SET_PIECE_HOLD)
        owner[a:b] = team
    return owner


def run_state_machine(ev_team, ev_slot, start, stop, dead=None, restart_owner=None):
    """Possession per frame in [start, stop): HOME/AWAY, NONE before anyone is known.

    restart_owner (v2): inside a set-piece window the tagged team holds the ball and the other
    team's evidence is ignored; normal challenge rules resume when the window ends.
    """
    out = np.full(stop - start, NONE, dtype=np.int8)
    holder = NONE
    ch_team, ch_first, ch_last, ch_frames, ch_slots = NONE, 0, 0, 0, {}
    restart_from = start  # back-date target for the first owner of a phase

    def reset_challenge():
        return NONE, 0, 0, 0, {}

    f = start
    while f < stop:
        if dead is not None and dead[f]:
            # Stoppage: freeze, then clear the holder when play resumes.
            j = f
            while j < stop and dead[j]:
                j += 1
            out[f - start:j - start] = holder
            holder = NONE
            if restart_owner is not None:
                tagged = restart_owner[f:j][restart_owner[f:j] != NONE]
                if len(tagged):
                    holder = int(tagged[-1])  # the set piece that ends this stoppage
            ch_team, ch_first, ch_last, ch_frames, ch_slots = reset_challenge()
            restart_from = j
            f = j
            continue

        if restart_owner is not None and restart_owner[f] != NONE:
            if holder != restart_owner[f]:
                holder = int(restart_owner[f])
                restart_from = f
            ch_team, ch_first, ch_last, ch_frames, ch_slots = reset_challenge()
            out[f - start] = holder
            f += 1
            continue

        t = ev_team[f]
        if t != NONE:
            if t == holder:
                ch_team, ch_first, ch_last, ch_frames, ch_slots = reset_challenge()
            else:
                if ch_team != t or f - ch_first > CHALLENGE_WINDOW:
                    ch_team, ch_first, ch_last, ch_frames, ch_slots = t, f, f, 0, {}
                ch_frames += 1
                ch_last = f
                s = int(ev_slot[f])
                ch_slots[s] = ch_slots.get(s, 0) + 1
                passed = sum(1 for c in ch_slots.values() if c >= TOUCH_FRAMES) >= 2
                needed = TOUCH_FRAMES if holder == NONE else HOLD_FRAMES
                if ch_frames >= needed or passed:
                    since = restart_from if holder == NONE else ch_first
                    out[since - start:f - start + 1] = t
                    holder = t
                    restart_from = f + 1
                    ch_team, ch_first, ch_last, ch_frames, ch_slots = reset_challenge()
                    f += 1
                    continue
        out[f - start] = holder
        f += 1

    # Before the first owner of the half: back-fill with that owner.
    known = np.flatnonzero(out != NONE)
    if len(known):
        out[:known[0]] = out[known[0]]
    return out


def absorb_short_spells(codes, min_len, dead=None, offset=0):
    """A-H-A with a short H -> A-A-A. Only spells of H/A; NONE and dead frames act as walls."""
    codes = codes.copy()
    live = codes != NONE
    if dead is not None:
        live &= ~dead[offset:offset + len(codes)]
    changed = True
    while changed:
        changed = False
        bounds = np.flatnonzero(np.diff(np.concatenate(([-9], np.where(live, codes, -9), [-9]))))
        runs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        for k in range(1, len(runs) - 1):
            a, b = runs[k]
            if not live[a] or b - a >= min_len:
                continue
            pa, pb = runs[k - 1]
            na, nb = runs[k + 1]
            if live[pa] and live[na] and codes[pa] == codes[na] != codes[a] \
                    and pb == a and na == b:
                codes[a:b] = codes[pa]
                changed = True
                break
    return codes


def build(dead=None, x_on_gaps=False, players=None, ball=None, slot_team=None, n_frames=None,
          restart_owner=None):
    """Full-length team labels as an array of 'H'/'A'/'X'."""
    if players is None:
        players, ball, slot_team = load_tracking()
    n_frames = n_frames or len(ball)
    ball_filled, seen = fill_ball_gaps(ball, BALL_GAP_FILL)
    ev_team, ev_slot = control_evidence(players, ball_filled, slot_team)

    codes = np.full(n_frames, NONE, dtype=np.int8)
    for start, stop in GRADED:
        stop = min(stop, n_frames)
        half = run_state_machine(ev_team, ev_slot, start, stop, dead, restart_owner)
        codes[start:stop] = absorb_short_spells(half, MIN_SPELL_FRAMES, dead, start)

    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "X")).astype("<U1")
    if x_on_gaps:
        missing = ~seen
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a > X_GAP_FRAMES:
                labels[a:b] = "X"
    return labels, ev_team


def write_team_csv(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,team\n")
        handle.writelines("{},{}\n".format(i, v) for i, v in enumerate(labels))


def describe(labels, ev_team):
    for start, stop in GRADED:
        seg = labels[start:stop]
        change = np.flatnonzero(seg[1:] != seg[:-1])
        spells = 1 + int(np.sum([(seg[i] in "HA") and (seg[i + 1] in "HA") for i in change]))
        print("  half {}-{}: H {:>6}  A {:>6}  X {:>6}   possessions ~{}   evidence frames {:.1%}"
              .format(start, stop - 1, int(np.sum(seg == "H")), int(np.sum(seg == "A")),
                      int(np.sum(seg == "X")), spells, float(np.mean(ev_team[start:stop] >= 0))))


def main():
    parser = argparse.ArgumentParser(description="Possession model -> outputs/team.csv")
    parser.add_argument("--dead", help="Ilham's dead.csv, to clear possession at each stoppage")
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--x-on-gaps", action="store_true",
                        help="write X on missing-ball gaps longer than {} frames".format(X_GAP_FRAMES))
    parser.add_argument("--refresh-cache", action="store_true", help="re-read the raw ATD")
    parser.add_argument("--cache", default=CACHE_DEFAULT, help="parsed-tracking cache file")
    parser.add_argument("--no-set-pieces", action="store_true",
                        help="v1 behaviour: ignore Smart Tagging set pieces at restarts")
    args = parser.parse_args()

    t0 = time.time()
    home, away = team_names(require(["atd_meta"])[0])
    print("H = {}, A = {}".format(home, away))
    players, ball, slot_team = load_tracking(args.cache, refresh=args.refresh_cache)
    dead = read_dead(args.dead, len(ball))
    restart_owner = None
    if not args.no_set_pieces:
        restart_owner = set_piece_owner(len(ball), home, away)
        print("set pieces: {} frames tagged H, {} tagged A".format(
            int(np.sum(restart_owner == HOME)), int(np.sum(restart_owner == AWAY))))
    labels, ev_team = build(dead, args.x_on_gaps, players, ball, slot_team, restart_owner=restart_owner)
    write_team_csv(args.out, labels)
    describe(labels, ev_team)
    print("wrote {} in {:.0f}s".format(os.path.relpath(args.out, ROOT), time.time() - t0))
    print("next: python src/score.py --team {}".format(os.path.relpath(args.out, ROOT)))


if __name__ == "__main__":
    main()
