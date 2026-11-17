"""End-to-end check of src/model.py on Metrica's PUBLIC Sample Game 3 (not the hackathon match).

    python research/check_metrica.py

Why: Sample Game 3 comes from the same provider and format family as the hackathon tracking
(EPTS text + XML metadata, coordinates 0..1 from a corner). Running the exact translate / predict
/ decode path on it catches format bugs -- wrong origin, swapped teams, wrong attack direction --
on data we are allowed to look at.

The reference labels are built here from the sample's public event data, so they are rough: they
follow events, not Metrica's corrected tracking. Use the score to compare against the baseline on
the same reference and to spot gross bugs, not as an estimate of the hackathon score.
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import model as M  # noqa: E402
import bench as B  # noqa: E402
import decoders as dec  # noqa: E402

GAME_DIR = os.path.join(ROOT, "data", "external", "metrica-sample", "data", "Sample_Game_3")
PITCH = (105.0, 68.0)
ALIVE_TEAM_EVENTS = {"PASS", "CARRY", "SET PIECE", "SHOT", "RECOVERY"}


def read_tracking():
    """players (T, n_players, 2) in metres from a corner, ball (T, 2), team per player column.

    Sample Game 3 changes its channel layout at substitutions, so each DataFormatSpecification is
    mapped to player ids separately (the hackathon file has one fixed layout of 46 slots).
    """
    root = ET.parse(os.path.join(GAME_DIR, "Sample_Game_3_metadata.xml")).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    column = {pid: i for i, (pid, _) in enumerate(players)}
    team = np.array([0 if tid == teams[0] else 1 for _, tid in players], dtype=np.int8)
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [column[channel_player[r]] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    lines = open(os.path.join(GAME_DIR, "Sample_Game_3_tracking.txt")).read().splitlines()
    T = len(lines)
    pos = np.full((T, len(players), 2), np.nan)
    ball = np.full((T, 2), np.nan)
    for line in lines:
        frame_s, body, ball_s = line.split(":")
        frame = int(frame_s)
        order = next(o for s, e, o in specs if s <= frame <= e)
        for col, group in zip(order, body.split(";")):
            x, y = group.split(",")[:2]
            pos[frame - 1, col] = (float(x), float(y))
        bx, by = ball_s.split(",")[:2]
        ball[frame - 1] = (float(bx), float(by))
    pos *= PITCH
    ball *= PITCH
    return pos, ball, team, teams


def reference_labels(n_frames, home_id):
    """Rough H/A/D per frame from the public events, plus the frame range of each period."""
    events = json.load(open(os.path.join(GAME_DIR, "Sample_Game_3_events.json")))["data"]
    events.sort(key=lambda e: e["start"]["frame"])
    lab = np.full(n_frames, B.X, dtype=np.int8)
    halves = []
    for period in (1, 2):
        evs = [e for e in events if e["period"] == period]
        start = evs[0]["start"]["frame"] - 1
        stop = max(e["end"]["frame"] for e in evs)
        halves.append((start, stop))
        state = B.H if evs[0]["team"]["id"] == home_id else B.A
        dead_until_restart = False
        marks = []
        for e in evs:
            kind, f = e["type"]["name"], e["start"]["frame"] - 1
            team = B.H if e["team"]["id"] == home_id else B.A
            if kind in ("BALL OUT", "FAULT RECEIVED"):
                marks.append((e["end"]["frame"] - 1 if kind == "BALL OUT" else f, B.D))
                dead_until_restart = True
            elif kind == "SET PIECE":
                marks.append((f, team))
                dead_until_restart = False
            elif kind in ALIVE_TEAM_EVENTS and not dead_until_restart:
                marks.append((f, team))
        cur = state
        marks.append((stop, None))
        lab[start:marks[0][0]] = cur
        for (f, k), (nxt, _) in zip(marks[:-1], marks[1:]):
            cur = k
            lab[max(f, start):max(nxt, start)] = cur
    return lab, halves


def main():
    t0 = time.time()
    pos, ball, team, teams = read_tracking()
    lab, halves = reference_labels(len(ball), teams[0])
    print("sample game 3: {} frames, {} player columns, halves {}".format(len(ball), pos.shape[1],
                                                                        halves), flush=True)
    model = M.train()
    # bench.score expects H/A/D codes per half; X outside halves is ignored there.
    gt, ours, base = [], [], []
    for start, stop in halves:
        obs = M.to_obs(pos, ball, team, PITCH, start, stop)
        ours.append(M.decode(M.predict_proba(model, obs)))
        base.append(B.baseline_labels(obs, 50))
        gt.append(lab[start:stop])
    for name, pred in (("baseline (nearest player, +-50 vote)", base), ("tree model + decode", ours)):
        res = B.score(gt, pred)
        n = sum(len(dec.spells(p)) for p in pred)
        print("{:40s} spellF1 {:.3f}  macroF1 {:.3f}  possessions {}".format(
            name, res["spellF1"], res["macroF1"], n), flush=True)
    n_gt = sum(len(dec.spells(g)) for g in gt)
    print("reference possessions (from events): {}".format(n_gt))
    # direction check: does the model put the right team in possession more often than not?
    alive = [(g == B.H) | (g == B.A) for g in gt]
    agree = np.mean(np.concatenate([(o == g)[a] for o, g, a in zip(ours, gt, alive)]))
    other = [np.where(g == B.H, B.A, B.H) for g in gt]
    flipped = np.mean(np.concatenate([(o == x)[a] for o, x, a in zip(ours, other, alive)]))
    print("on alive frames: same team as reference {:.3f}, opposite team {:.3f}".format(
        agree, flipped))
    print("done in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
