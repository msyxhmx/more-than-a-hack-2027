"""Merge the two lanes into the submission (contract: src/README.md).

    python src/merge.py --team-name <team_name>                        # outputs/dead.csv + team.csv
    python src/merge.py --team-name <team_name> --lanes outputs/model  # the tree model's lanes

Writes outputs/<team_name>.csv and checks it with the official validator logic. A MISSING lane
falls back to the baseline (as the contract says) with a loud warning; a lane that exists but breaks
the contract stops the merge, because silently submitting the baseline is worse than failing.

Then run:  python evaluation/validate_submission.py outputs/<team_name>.csv
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lanes import OUT, ROOT, fallback_lanes, merge, read_dead, read_team, write_submission  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team-name", required=True, help="output file is outputs/<name>.csv")
    parser.add_argument("--lanes", default=OUT, metavar="DIR",
                        help="folder holding dead.csv and team.csv (default outputs/)")
    parser.add_argument("--out", help="override the output path")
    args = parser.parse_args()

    base_dead, base_team = fallback_lanes()
    lanes = []
    for name, reader, fallback in (("dead.csv", read_dead, base_dead),
                                   ("team.csv", read_team, base_team)):
        path = os.path.join(args.lanes, name)
        values, why = reader(path)
        if why == "missing":
            print("WARNING  {} missing -- using the BASELINE for this lane".format(
                os.path.relpath(path, ROOT)))
            values = fallback
        elif why is not None:
            sys.exit("FAIL  {} breaks the contract: {}".format(os.path.relpath(path, ROOT), why))
        else:
            print("OK       {}".format(os.path.relpath(path, ROOT)))
        lanes.append(values)

    out = args.out or os.path.join(OUT, "{}.csv".format(args.team_name))
    write_submission(merge(*lanes), out)
    print("OK       wrote {} (validator passed)".format(os.path.relpath(out, ROOT)))


if __name__ == "__main__":
    main()
