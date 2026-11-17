"""One command: match data in data/ -> outputs/<team_name>.csv (task I06).

    python src/run.py --team-name Lawan                  # the method in DEFAULT_METHOD
    python src/run.py --team-name Lawan --method rules   # rule model + dead-ball lane
    python src/run.py --team-name Lawan --method model   # tree model (needs src/model.py)

Methods
-------
model     Ilham's tree model + Viterbi. Contract: src/model.py has main(out_dir) that writes
          out_dir/dead.csv and out_dir/team.csv via lanes.write_lanes(). Lanes go to outputs/model/.
rules     possession.py v2 -> outputs/team.csv, and dead_ball.py -> outputs/dead.csv when it is
          implemented; until then the dead-ball lane falls back to the baseline (merge rule).
baseline  both lanes from evaluation/results/baseline_smooth50.csv (the insurance file).

The method is chosen up front, never by its public 5-minute score (RULES.md). Every step is
deterministic; the script ends by printing the file's SHA-256 so two runs can be compared.
"""

import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lanes  # noqa: E402

DEFAULT_METHOD = "model"  # the submitted file; chosen on held-out DFL matches, never on this match


def run_rules():
    import possession
    import numpy as np
    from games import require
    from baseline import team_names

    home, away = team_names(require(["atd_meta"])[0])
    players, ball, slot_team = possession.load_tracking(None)  # no cache: always from raw data
    try:
        import dead_ball
        dead_ball.main()
        dead_path = os.path.join(lanes.OUT, "dead.csv")
    except NotImplementedError:
        dead_path = None
        stale = os.path.join(lanes.OUT, "dead.csv")
        if os.path.isfile(stale):
            os.remove(stale)  # never merge a dead.csv this run did not produce
        print("dead_ball.py not implemented yet: the dead-ball lane falls back to the baseline")
    dead = possession.read_dead(dead_path, len(ball)) if dead_path else None
    owner = possession.set_piece_owner(len(ball), home, away)
    labels, _ = possession.build(dead, False, players, ball, slot_team, restart_owner=owner)
    possession.write_team_csv(os.path.join(lanes.OUT, "team.csv"), labels)
    return lanes.OUT


def run_model():
    if not os.path.isfile(os.path.join(HERE, "model.py")):
        sys.exit("src/model.py is not in the repository yet (task M01); use --method rules")
    import model
    out_dir = os.path.join(lanes.OUT, "model")
    model.main(out_dir)
    return out_dir


def run_baseline():
    out_dir = os.path.join(lanes.OUT, "baseline_lanes")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    return out_dir  # empty: merge falls back to the baseline for both lanes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--method", choices=["model", "rules", "baseline"], default=DEFAULT_METHOD)
    args = parser.parse_args()

    lanes_dir = {"model": run_model, "rules": run_rules, "baseline": run_baseline}[args.method]()

    base_dead, base_team = lanes.fallback_lanes()
    parts = []
    for name, reader, fallback in (("dead.csv", lanes.read_dead, base_dead),
                                   ("team.csv", lanes.read_team, base_team)):
        values, why = reader(os.path.join(lanes_dir, name))
        if why == "missing":
            print("{}: baseline".format(name))
            values = fallback
        elif why is not None:
            sys.exit("{} breaks the contract: {}".format(name, why))
        else:
            print("{}: {}".format(name, os.path.relpath(os.path.join(lanes_dir, name), lanes.ROOT)))
        parts.append(values)

    out = os.path.join(lanes.OUT, "{}.csv".format(args.team_name))
    lanes.write_submission(lanes.merge(*parts), out)
    with open(out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print("wrote {} (method {}, validator passed)".format(os.path.relpath(out, lanes.ROOT),
                                                          args.method))
    print("sha256 {}".format(digest))


if __name__ == "__main__":
    main()
