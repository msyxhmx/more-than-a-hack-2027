"""Convert the open DFL matches (Bassek et al. 2025, CC-BY 4.0) into compact .npz files.

These are OTHER matches (Bundesliga 2022/23), used only to test methods offline. Each match gives,
at 25 fps: player x/y per team, ball x/y/z, and two per-frame fields from the provider --
BallPossession (1 = home, 2 = away) and BallStatus (1 = alive, 0 = dead). Those two fields map
straight onto the hackathon's H / A / D labels.

    python research/dfl_load.py data/external/dfl data/external/dfl_npz
"""
import glob
import os
import re
import sys

import numpy as np

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]+" TeamId="([^"]+)" PersonId="([^"]+)"')
N_ATTR = re.compile(r' N="(\d+)"')
X_ATTR = re.compile(r' X="([-\d.]+)"')
Y_ATTR = re.compile(r' Y="([-\d.]+)"')
Z_ATTR = re.compile(r' Z="([-\d.]+)"')
POSS = re.compile(r' BallPossession="(\d)"')
STAT = re.compile(r' BallStatus="(\d)"')
TEAM = re.compile(r'<Team TeamId="([^"]+)"[^>]*Role="(\w+)"')
HALVES = ("firstHalf", "secondHalf")


def parse_match(info_path, pos_path):
    roles = dict(TEAM.findall(open(info_path, encoding="utf-8").read()))  # TeamId -> home/guest
    sets = {}  # (half, team, person) -> list of rows
    current = None
    with open(pos_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if current is None:
                    continue
                n = int(N_ATTR.search(line).group(1))
                x = float(X_ATTR.search(line).group(1))
                y = float(Y_ATTR.search(line).group(1))
                if current[1] == "BALL":
                    z = float(Z_ATTR.search(line).group(1))
                    sets[current].append((n, x, y, z, int(POSS.search(line).group(1)),
                                          int(STAT.search(line).group(1))))
                else:
                    sets[current].append((n, x, y))
            elif line.startswith("<FrameSet"):
                half, team, person = FRAMESET.search(line).groups()
                if half in HALVES and (team == "BALL" or team in roles):
                    current = (half, team, person)
                    sets[current] = []
                else:
                    current = None  # referees, extra sections
    out = {}
    for hi, half in enumerate(HALVES):
        ball = np.array(sets[[k for k in sets if k[0] == half and k[1] == "BALL"][0]], dtype=np.float64)
        n0, n1 = int(ball[0, 0]), int(ball[-1, 0])
        T = n1 - n0 + 1
        b = np.full((T, 3), np.nan, dtype=np.float32)
        poss = np.zeros(T, dtype=np.int8)
        stat = np.zeros(T, dtype=np.int8)
        idx = ball[:, 0].astype(int) - n0
        b[idx] = ball[:, 1:4]
        poss[idx] = ball[:, 4]
        stat[idx] = ball[:, 5]
        out["ball_%d" % hi] = b
        out["poss_%d" % hi] = poss
        out["status_%d" % hi] = stat
        for role, name in (("home", "home"), ("guest", "away")):
            keys = sorted(k for k in sets if k[0] == half and roles.get(k[1]) == role)
            arr = np.full((T, len(keys), 2), np.nan, dtype=np.float32)
            for pi, key in enumerate(keys):
                rows = np.array(sets[key], dtype=np.float64)
                if not len(rows):
                    continue
                ii = rows[:, 0].astype(int) - n0
                ok = (ii >= 0) & (ii < T)
                arr[ii[ok], pi] = rows[ok, 1:3]
            out["%s_%d" % (name, hi)] = arr
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for pos_path in sorted(glob.glob(os.path.join(src, "*positions_raw_observed*.xml"))):
        match = pos_path.rsplit("DFL-MAT-", 1)[1][:-4]
        info = glob.glob(os.path.join(src, "*matchinformation*%s.xml" % match))[0]
        target = os.path.join(dst, match + ".npz")
        if os.path.exists(target):
            continue
        data = parse_match(info, pos_path)
        np.savez_compressed(target, **data)
        print(match, {k: v.shape for k, v in data.items() if k.startswith(("ball", "home", "away"))},
              flush=True)
