"""One command for every check (owner: Syahmi, task P02).

    python src/score.py                                   # outputs/dead.csv + outputs/team.csv
    python src/score.py --sub outputs/<team_name>.csv     # also validate the real submission file
    python src/score.py --log "possession v1"             # append the scores to outputs/score_log.csv

What it prints
--------------
1. Contract checks on dead.csv and team.csv (columns, 199,319 rows, frame order, allowed values).
   A missing lane file falls back to baseline_smooth50, exactly like the merge rule.
2. The merged file, built in memory with the contract's merge rule, run through the official
   validator logic, with its possession count.
3. Public 5-minute scores of the merged file: spellF1 (ranking), macroF1, possessions vs the truth.
4. Each lane on its own, so one person's bug does not hide behind the other's:
   - dead lane: D precision / recall / F1, and spellF1 if possession were perfect;
   - team lane: H/A accuracy on live frames, X rate, and spellF1 if stoppages were perfect.

The public window is 17 possessions. Use these numbers to see that the pipeline is CORRECT, never
to choose between two versions or tune a parameter -- that is against the rules and, as the
baselines show, picks the wrong version anyway.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402

N_FRAMES = GAMES[GAME]["n_video_frames"]
GRADED = ((11185, 90652), (110808, 190198))  # half-open, from DATA.md / src/README.md
OUT = os.path.join(ROOT, "outputs")
FALLBACK = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")


# -- lane files ----------------------------------------------------------------------------------

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


def fallback_lanes():
    labels = pm.read_label_csv(FALLBACK)["label"].values.astype(str)
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


# -- scoring helpers -----------------------------------------------------------------------------

def public_gt():
    paths = glob.glob(os.path.join(ROOT, "evaluation", "gt", "*_possession_gt_public.csv"))
    if not paths:
        sys.exit("public ground truth not found in evaluation/gt/")
    return pm.label_codes(pm.read_label_csv(paths[0]))


def spell_f1(gt, pred):
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)


def dead_lane_report(gt, dead_codes):
    graded = gt != pm.X
    truth = gt[graded] == pm.D
    pred = dead_codes[graded] == "1"
    tp = int(np.sum(truth & pred))
    precision = tp / float(pred.sum()) if pred.sum() else 0.0
    recall = tp / float(truth.sum()) if truth.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # If possession were perfect: GT team on live frames, carried through frames the lane misses.
    combo = gt.copy()
    team_ffill = pd.Series(np.where(np.isin(gt, [pm.H, pm.A]), gt, np.nan)).ffill().bfill()
    team_ffill = team_ffill.fillna(pm.H).values.astype(np.int8)
    combo[graded] = np.where(dead_codes[graded] == "1", pm.D, team_ffill[graded])
    spells = spell_f1(gt, combo)

    gt_stops = len([s for s in pm.segments(gt, pm.valid_runs(gt)) if s[2] == pm.D])
    pred_stops = len([s for s in pm.segments(combo, pm.valid_runs(gt)) if s[2] == pm.D])
    return precision, recall, f1, spells["mean_f1"], gt_stops, pred_stops


def team_lane_report(gt, team_codes):
    live = np.isin(gt, [pm.H, pm.A])
    answered = live & (team_codes != "X")
    letters = np.array(pm.LABELS)[gt]
    accuracy = float(np.mean(team_codes[answered] == letters[answered])) if answered.any() else 0.0
    x_rate = float(np.mean(team_codes[live] == "X")) if live.any() else 0.0

    # If stoppages were perfect: GT D wherever the truth is dead, the lane everywhere else.
    combo = pm.label_codes(np.where(gt == pm.D, "D", np.where(gt == pm.X, "X", team_codes)))
    spells = spell_f1(gt, combo)
    return accuracy, x_rate, spells["mean_f1"], spells["n_pred"], spells["n_gt"]


# -- main ----------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", default=os.path.join(OUT, "dead.csv"))
    parser.add_argument("--team", default=os.path.join(OUT, "team.csv"))
    parser.add_argument("--sub", help="the real submission file to validate as well")
    parser.add_argument("--log", metavar="VERSION", help="append scores to outputs/score_log.csv")
    args = parser.parse_args()
    ok = True

    print("1. lane files (contract)")
    base_dead, base_team = fallback_lanes()
    dead, why = check_lane(args.dead, "dead", {"0", "1"})
    print("   dead.csv  {}".format("OK" if why is None else "FALLBACK to baseline -- " + why))
    if why is None:
        dead_src = "dead.csv"
    else:
        dead, dead_src = base_dead, "baseline"
        ok &= why == "missing"
    team, why = check_lane(args.team, "team", {"H", "A", "X"})
    print("   team.csv  {}".format("OK" if why is None else "FALLBACK to baseline -- " + why))
    if why is None:
        team_src = "team.csv"
    else:
        team, team_src = base_team, "baseline"
        ok &= why == "missing"

    print("\n2. merged submission (in memory)")
    merged = merge(dead, team)
    reason = pm.validate_submission(
        pd.DataFrame({"frame": np.arange(N_FRAMES), "label": merged}), N_FRAMES)
    codes = pm.label_codes(merged)
    runs = [(a, b) for a, b in GRADED]
    stats = pm.possession_stats(pm.fill_abstentions(codes, runs), runs)
    print("   validator {}".format("OK" if reason is None else "FAIL -- " + reason))
    print("   full match: {} possessions, mean {:.1f} s, home share {:.0%}".format(
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
    ok &= reason is None

    if args.sub:
        try:
            sub = pm.read_label_csv(args.sub)
            sub_reason = pm.validate_submission(sub, N_FRAMES)
        except Exception as error:
            sub_reason = "unreadable: {}".format(error)
        print("   {}: {}".format(os.path.basename(args.sub),
                                 "OK" if sub_reason is None else "FAIL -- " + sub_reason))
        if sub_reason is None:
            same = np.array_equal(pm.label_codes(sub), codes)
            print("   matches the in-memory merge: {}".format("yes" if same else
                  "NO -- merge.py and the contract disagree"))
            ok &= same
        ok &= sub_reason is None

    gt = public_gt()
    result = pm.evaluate(gt, codes)
    print("\n3. public 5 minutes  (correctness check only -- never tune on this)")
    print("   spellF1 {:.4f}   macroF1 {:.4f}   possessions {} vs truth {}   mean {:.1f} s vs {:.1f} s"
          .format(result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
                  result["n_gt_spells"], result["mean_spell_duration_s"],
                  result["gt_mean_spell_duration_s"]))
    print("   reference: baseline_smooth50 spellF1 0.3256 / macroF1 0.6207 here, 0.2433 full match")

    print("\n4. each lane alone")
    p, r, f, s_dead, gt_stops, pred_stops = dead_lane_report(gt, dead)
    print("   dead ({:<8}) D precision {:.2f}  recall {:.2f}  F1 {:.2f}   stoppages {} vs truth {}"
          "   spellF1 if possession were perfect {:.4f}".format(
              dead_src, p, r, f, pred_stops, gt_stops, s_dead))
    acc, x_rate, s_team, n_pred, n_gt = team_lane_report(gt, team)
    print("   team ({:<8}) H/A accuracy on live frames {:.2%}  X rate {:.1%}   possessions {} vs "
          "truth {}   spellF1 if stoppages were perfect {:.4f}".format(
              team_src, acc, x_rate, n_pred, n_gt, s_team))

    if args.log:
        path = os.path.join(OUT, "score_log.csv")
        new = not os.path.isfile(path)
        os.makedirs(OUT, exist_ok=True)
        with open(path, "a") as handle:
            if new:
                handle.write("version,dead_src,team_src,spellF1,macroF1,poss,gt_poss,"
                             "dead_F1,team_acc,spell_dead_only,spell_team_only,full_match_poss\n")
            handle.write("{},{},{},{:.4f},{:.4f},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{}\n".format(
                args.log.replace(",", " "), dead_src, team_src, result["spell_f1_mean"],
                result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"], f, acc,
                s_dead, s_team, stats["n_spells"]))
        print("\nlogged to {}".format(os.path.relpath(path, ROOT)))

    print("\n{}".format("ALL CHECKS PASSED" if ok else "SOMETHING FAILED -- see above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
