"""The lane-file contract in one place (see src/README.md).

Every producer and consumer of outputs/ goes through these helpers, so the contract cannot drift
between files:

- a model that predicts H/A/D per frame (Ilham's tree model + Viterbi) calls `write_lanes` once and
  gets a contract-correct dead.csv and team.csv;
- possession.py (rules) writes team.csv only and keeps working as before;
- score.py and merge.py read lanes with `check_lane`, fall back to the baseline with
  `fallback_lanes`, and combine them with `merge`.

Round trip: for any label array, merge(*split_labels(labels)) equals labels inside the graded
halves and X outside them. `self_test()` checks that on random labels; no match data is involved.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
GRADED = ((11185, 90652), (110808, 190198))  # half-open, from DATA.md / src/README.md
OUT = os.path.join(ROOT, "outputs")
FALLBACK = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")


# -- reading -------------------------------------------------------------------------------------

def check_lane(path, column, allowed):
    """(values, None) if the file meets the contract, else (None, reason)."""
    if not os.path.isfile(path):
        return None, "missing"
    try:
        table = pd.read_csv(path, dtype={column: "object"})
    except Exception as error:
        return None, "unreadable: {}".format(error)
    table.columns = [str(c).strip().lower() for c in table.columns]
    if list(table.columns) != ["frame", column]:
        return None, "columns {} != ['frame', '{}']".format(list(table.columns), column)
    if len(table) != N_FRAMES:
        return None, "{} rows != {}".format(len(table), N_FRAMES)
    if not np.array_equal(table["frame"].values, np.arange(N_FRAMES)):
        return None, "frame column is not 0..{} in order".format(N_FRAMES - 1)
    values = table[column]
    if values.isna().any():
        return None, "{} blank cells".format(int(values.isna().sum()))
    values = np.array([str(v).strip().upper() for v in values], dtype=object)
    bad = sorted(set(values) - set(allowed))
    if bad:
        return None, "values outside {}: {}".format(sorted(allowed), bad[:5])
    return values, None


def read_dead(path):
    return check_lane(path, "dead", {"0", "1"})


def read_team(path):
    return check_lane(path, "team", {"H", "A", "X"})


def fallback_lanes():
    """The baseline (baseline_smooth50) split into the two lanes."""
    labels = pm.read_label_csv(FALLBACK)["label"].values.astype(str)
    return split_labels(labels)


# -- combining -----------------------------------------------------------------------------------

def split_labels(labels):
    """H/A/D/X labels -> (dead, team) lane values. D -> dead 1 / team X; H, A -> dead 0 / team H, A."""
    labels = np.array([str(v).strip().upper() for v in labels], dtype=object)
    if len(labels) != N_FRAMES:
        raise ValueError("{} labels != {}".format(len(labels), N_FRAMES))
    bad = sorted(set(labels) - {"H", "A", "D", "X"})
    if bad:
        raise ValueError("labels outside H/A/D/X: {}".format(bad[:5]))
    dead = np.where(labels == "D", "1", "0").astype(object)
    team = np.where(np.isin(labels, ["H", "A"]), labels, "X").astype(object)
    return dead, team


def merge(dead, team):
    """The contract's merge rule, as label strings."""
    labels = np.full(N_FRAMES, "X", dtype=object)
    for start, stop in GRADED:
        labels[start:stop] = team[start:stop]
        labels[start:stop][dead[start:stop] == "1"] = "D"
    return labels


# -- writing -------------------------------------------------------------------------------------

def _write(path, column, values):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame({"frame": np.arange(N_FRAMES), column: values}).to_csv(path, index=False)


def write_lanes(labels, out_dir=OUT):
    """Write dead.csv and team.csv for a model that predicts one H/A/D/X label per frame.

    Checks the files it wrote against the contract and returns their paths.
    """
    dead, team = split_labels(labels)
    dead_path = os.path.join(out_dir, "dead.csv")
    team_path = os.path.join(out_dir, "team.csv")
    _write(dead_path, "dead", dead)
    _write(team_path, "team", team)
    for path, reader in ((dead_path, read_dead), (team_path, read_team)):
        _, why = reader(path)
        if why is not None:
            raise RuntimeError("{} breaks the contract: {}".format(path, why))
    return dead_path, team_path


def write_submission(labels, path):
    """Write the frame,label submission and run the official validator logic on it."""
    _write(path, "label", labels)
    reason = pm.validate_submission(pm.read_label_csv(path), N_FRAMES)
    if reason is not None:
        raise RuntimeError("{} fails validation: {}".format(path, reason))
    return path


# -- self test (synthetic labels only) -----------------------------------------------------------

def self_test(seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(np.array(["H", "A", "D", "X"], dtype=object), N_FRAMES)
    merged = merge(*split_labels(labels))
    inside = np.zeros(N_FRAMES, dtype=bool)
    for start, stop in GRADED:
        inside[start:stop] = True
    assert np.array_equal(merged[inside], labels[inside]), "round trip changed graded labels"
    assert (merged[~inside] == "X").all(), "frames outside the graded halves must be X"
    return True


if __name__ == "__main__":
    self_test()
    print("lanes.py self test OK (synthetic labels)")
