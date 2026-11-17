# Team Lawan: technical document

More than a Hack 2027, ball-possession challenge (FC Barcelona Innovation Hub and Metrica Sports).
Code: https://github.com/msyxhmx/more-than-a-hack-2027. The submission is `Lawan.csv`.

## 1. Summary

We train a frame classifier on **other** matches and use it to label this one. The classifier is a
gradient-boosted tree that predicts home / away / dead for every frame. A **minimum-duration
Viterbi** decoder then turns those per-frame predictions into clean possessions. The training data
is the 7 open DFL Bundesliga matches. Before training we degrade their clean tracking so that it
looks like an automatic tracker's output, with the ball missing in long blocks. Nothing in the
pipeline is fitted, tuned or chosen on the provided match or its public 5 minutes.

On the 7 DFL matches, tested leave-one-match-out under simulated tracker noise, this scores
**spellF1 0.61**. The organisers' baseline scores **0.20** on the same simulated data.

## 2. What the problem rewards, and why we built this

spellF1 matches whole possessions by overlap, so the score depends on how many possessions we
predict and where their boundaries fall. Per-frame accuracy matters much less. A one-frame flicker
to the wrong team splits a long possession into three pieces, and usually none of them matches.
The organisers' four baselines show this: macroF1 moves by 0.04 across them while spellF1 moves
fivefold. So we need a good per-frame signal and a decoder that refuses to flicker.

The tracking also shapes the design. In the organisers' public `baseline_raw.csv` (where D marks
frames with no ball), the ball goes missing in **blocks**: a median of about 11 seconds, and almost
no gaps between 1 and 5 seconds long. The ball doesn't flicker in and out, it vanishes for whole spells.
Filling gaps by interpolation can't solve that. For long stretches the model has to read
possession from how the players move.

## 3. The approach

### 3.1 Training data: degraded DFL matches

- **Source.** The 7 matches in the open DFL dataset (Bassek et al. 2025, *Scientific Data*;
  https://doi.org/10.6084/m9.figshare.28196177), 2022/23 Bundesliga and 2. Bundesliga, CC-BY 4.0.
  Each match has 25 fps positions for all players and the ball. Each frame also has the provider's
  `BallPossession` (home / away) and `BallStatus` (alive / dead) fields, which map straight onto H, A
  and D. About 2.7 GB in total. `src/dfl.py` downloads the files from a mirror pinned to one commit
  and checks each one against a SHA-256 hash.
- **Who labelled it.** The Deutsche Fußball Liga's data provider, as published with the
  dataset (the labels are provider fields, not hand annotations by us). We added no hand labels anywhere, on any match.
- **Degradation ("degradation twin").** Clean tracking would teach the model things the real data
  never shows. `research/bench.py` therefore passes every DFL half through a noise simulator:
    - **Ball outages.** The ball goes missing in blocks. Their lengths are drawn from the
      run-length distribution of present and missing ball in the organisers' public
      `baseline_raw.csv`. Measuring that needs no labels.
    - **Where outages start.** 70% of outages start near a contested moment (ball in the air, or
      a crowd around the ball).
    - **Players.** Positions get 0.2 m correlated jitter, and tracks drop out in bursts, so on
      average about 20.5 of 22 players are visible.
    - **Dead-ball freeze.** The DFL files freeze the ball on every dead frame, which would let a
      model read "dead" straight off the ball. We replace that stretch with a ball carried by the
      nearest player and then placed at the restart spot, hidden in 1 stoppage out of 5.
- **Team-swap augmentation.** Every training frame is also used with the two teams exchanged
  (labels H and A swapped), so the model has no home/away bias.

On the bench, the noise model is what matters most. With the same test data, a model trained on
clean tracking scores 0.36 spellF1, and one trained on random per-frame ball dropout at the right
overall rate scores 0.46. Training on block outages scores 0.62 to 0.63.

### 3.2 Features and classifier

There are about 190 per-frame features (`research/bench.py: features`). They are computed in a
canonical orientation, with the team being scored attacking towards +x:

- **Team shape and motion:** centroid, spread, mean and maximum speed, share of slow players,
  last-defender line, and number of players seen.
- **Ball, when visible:** each team's nearest and second-nearest distance to the ball, players
  within 3 m and 8 m, a soft "control" score, ball speed, and distance to the touchline.
- **Time context:** frames since and until the ball was last and next seen, and the last and next
  known values of the ball features.
- **Rolling means** of the key signals over past and future windows of 12, 25, 50, 125 and 250
  frames.

When the ball is not seen, its features are NaN, which `HistGradientBoostingClassifier` handles
natively. The classifier uses scikit-learn 1.9.1 with 200 iterations, learning rate 0.1, 63 leaves,
min 200 samples per leaf, L2 1.0, no early stopping and `random_state=0`. It trains on every 5th
frame of every half.

At prediction time the model runs on the match and on its team-swapped mirror. The two outputs
are averaged, so H and A are treated identically.

### 3.3 Decoder: minimum-duration Viterbi

The decoder (`research/decoders.py: viterbi_min_dur`) finds the best H / A / D path in which every
possession lasts at least **25 frames (1 s)** and every stoppage at least **125 frames (5 s)**, with
a switch cost of 5. Segments cut by the start or end of a half are exempt.

These settings were picked on held-out DFL matches and are fixed in `src/model.py`. On the bench,
the same classifier scores 0.25 with raw per-frame argmax (1,458 predicted possessions against
420 true) and 0.61 once decoded (326 predicted).

### 3.4 Output

Frames outside the two graded halves (11,185–90,651 and 110,808–190,197, from DATA.md) are written
as X. Inside them, every frame gets H, A or D from the decoder. `src/lanes.py` checks the file
against the format contract, and `src/run.py` runs the organisers' `validate_submission.py` checks
before printing the file's SHA-256.

## 4. Possession rules we implemented

Our definition of possession is the DFL provider's per-frame labels, with one change, applied
consistently in training. The decoder's minimum durations then shape the output:

| situation | what we do | where |
|---|---|---|
| **Ball out of play / restarts** | D, from DFL `BallStatus`. A stoppage lasts at least 5 s in the output, so a throw-in or free kick is one D spell. Possession after the restart goes to whichever team the model sees in control. | training labels, decoder |
| **Touches without control, deflections, challenges** | An opposing touch shorter than 6 frames (0.24 s) is absorbed into the previous team's possession, and in the output no possession is shorter than 1 s. A tackle that only knocks the ball loose does not change possession. A won challenge does, once the winning team keeps the ball. | `min_touch_filter` (tau = 6), decoder |
| **Loose balls** | The last team in possession keeps it until the other team controls the ball (the challenge text's rule). | DFL label convention, decoder |
| **Ball not tracked** | We still predict H / A / D from the players' movement. We never write X inside a graded half. | features, decoder |
| **Before and after each half** | X, outside the graded windows. | `src/lanes.py` |

**Disclosure on the 6-frame threshold.** We based tau = 6 on the challenge text ("a touch without
control" is not a change of possession) and fixed it before testing; it was not tuned. While
choosing it we had read the organisers' published lateness table in EVALUATION.md, which is
derived from the ground truth. It suggests this match has fewer very short possessions than DFL.
We used it only as a description, never to fit or select anything.

## 5. Assets: used and dropped

| asset | used? | how |
|---|---|---|
| Tactical ATD (`_atd.xml`, `_atd.txt`) | **Yes** | Player and ball positions and each track's team: the model's only input from the match |
| Organisers' `baseline_raw.csv` (public) | **Yes, label-free** | Run lengths of the ball being present or missing, used for the noise simulator |
| Smart Tagging XML | Built, not in the final file | Used by our rule-based fallback (set-piece owner). It could serve as a restart prior or a label-free cross-check, but this is untested |
| Tactical video | No | The ball is a few pixels wide; no ball detector was tried |
| Homography | No | |
| Broadcast ATD / video | No | Syncing the two cameras in pitch space worked in simulation (0–2 frame error), but the real gain was untested by the deadline |
| Metrica sample data (open Sample Games) | Checks only | Compared label definitions, and ran the exact translate / predict / decode path on Sample Game 3 to catch format bugs (`research/check_metrica.py`): spellF1 0.71 against 0.37 for the baseline, on a rough event-based reference. Not used for training or tuning |
| Public 5-minute ground truth | Correctness check only | Confirms the pipeline runs end to end and the file is aligned. Never used to tune or choose between versions |

**Methods tried and dropped** (all evaluated on DFL, leave-one-match-out):

| method | result |
|---|---|
| Hand-written rules (nearest player with hold time, dead-ball rules), on the bench (`research/run_rules.py`) | 0.30 spellF1 against 0.61. A fuller rule model (`src/possession.py`, state machine + Smart Tagging set pieces) is kept as a fallback (`--method rules`) |
| A second refinement stage (MS-TCN idea) on top of the classifier | 0.609 → 0.610, inside noise |
| Majority-vote and plain Viterbi smoothing | 0.59 and 0.60, a tie with min-duration Viterbi |
| Label-free decoder selection by expected spellF1 (posterior sampling) | No gain; the model's posterior is overconfident |
| Player-only model (no ball features) | 0.52, kept as evidence that the model survives long ball outages, not as the submission |

## 6. Limitations and what we'd do with more time

- **Simulated vs real noise.** All scores above are on DFL matches with *simulated* tracker
  noise. The real ATD may have faults we did not simulate: identity swaps, duplicate tracks,
  wrong team labels, false ball detections, and whatever the ball does while play is stopped. We
  expect a lower score on the real match, but well above the baseline.
- **Different definitions of possession.** The DFL provider switches possession more readily than
  Metrica's "controlled possession" appears to. A stricter relabelling of the training data would
  predict fewer, longer possessions. We had no legal way to set how strict without touching this
  match.
- **Short possessions.** On DFL, possessions under 2 s are mostly missed (7% found under 1 s, 28%
  at 1–2 s), while those over 10 s are found 90% of the time. The boundaries we do find are
  accurate (mean error 3 frames).
- **Second-half kick-off.** The graded second half starts about 2 minutes before its kick-off, and
  the FAQ says that stretch is all D. We let the model label it like any other stretch (players
  standing still is a strong dead-ball signal) instead of hard-coding it. Detecting the kick-off
  from Smart Tagging by code would make this certain.
- **Only 7 training matches, from one league.** More labelled tracking with the same definition of
  possession would likely help more than a bigger model.
- **Next steps.**
    1. Recover the ball from the broadcast feed by syncing the cameras in pitch space. Perfect
       tracking is worth at most about +0.12 on the bench.
    2. Use Smart Tagging restarts as a decoder prior.
    3. Calibrate the posterior so that label-free model selection becomes usable.

## 7. Reproduction

Python 3.12, macOS or Linux. On Windows use `.venv\Scripts\python`. The match data bundle goes in
`data/` (or point `MTAH_DATA_DIR` at it).

```bash
git clone https://github.com/msyxhmx/more-than-a-hack-2027.git && cd more-than-a-hack-2027
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lawan.txt
.venv/bin/python src/run.py --team-name Lawan
python evaluation/validate_submission.py outputs/Lawan.csv
```

`run.py` downloads and verifies the DFL matches on the first run, trains the model (a few minutes
on a laptop CPU), labels the match, writes `outputs/Lawan.csv`, validates it and prints its SHA-256.
No model file is stored: training is part of every run, so the file is rebuilt from data and code
alone.

**Determinism.** The DFL files are pinned by SHA-256, every random step has a fixed seed, early
stopping is off, and package versions are pinned in `requirements-lawan.txt`. Two complete runs on
the same machine give bit-identical output. We also checked the parts that could differ between
machines, using DFL matches only (`research/py312_bench.log`, and a check on 17 Nov):

- **Python version.** The pinned packages give bit-identical probabilities on Python 3.11 and 3.12.
- **CPU thread count.** Training with 8 or 2 threads gives the same probabilities and labels, bit
  for bit.

A different CPU architecture or operating system has not been tested. If a rebuild there ever
differs, the change would be tiny floating-point noise in the probabilities, which the decoder
almost always absorbs.

Data rule: the match data is not in this repository (`data/` and `outputs/` are gitignored), and
no match data was ever sent to a hosted AI model. AI assistants were used to write code only.
