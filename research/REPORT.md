# Possession research report — More than a Hack 2027

Written 16 Nov 2026. Submission deadline is **17 Nov 2026** ([OVERVIEW.md](../OVERVIEW.md)).

**Tags.** `[VERIFIED]` = I read it in the source or ran it myself. `[REPORTED]` = stated by a
source I could not check in full. `[HYPOTHESIS]` = my own reasoning, not tested.

**What I did and did not touch.** Rule 4 bans sending this match's data to a cloud model, and I am
one. So I never read `data/tactical/*` or any video. Everything about this match below comes from
files in the public repo: the docs, the scorer, and the organisers' committed
`evaluation/results/baseline_raw.csv`. All experiments ran on **other matches** (7 open Bundesliga
games), on this laptop. Code is in [research/](.), numbers are reproducible with the commands in
section 7.

---

## 1. Executive summary

1. **Build this:** train a per-frame H/A/D classifier on other matches that have possession labels,
   after degrading their clean tracking to look like this match's ATD; then decode with a
   minimum-duration Viterbi. `[VERIFIED]` on held-out matches: spellF1 **0.61** vs **0.20** for the
   organisers' baseline on the same simulated data.
2. **The ball does not flicker, it vanishes in blocks.** In the public baseline file there are only
   146 ball gaps in play, median 268 frames (10.7 s), mean 439 frames. `[VERIFIED]` So gap-filling
   is the wrong tool; the model must read possession from player movement alone for long stretches.
3. **Players alone are enough to stay useful.** A model with every ball feature removed still
   reaches spellF1 **0.52**, dead-ball recall 0.94, team accuracy 0.88. `[VERIFIED]`
4. **Decoding is where the score is.** The same classifier scores 0.25 with raw argmax (1,458
   predicted possessions vs 420 true) and 0.59–0.61 once decoded. `[VERIFIED]`
5. **The noise model must match.** Train on clean tracking or on random per-frame ball dropout and
   the model collapses on block-outage data (spellF1 0.36 and 0.46 vs 0.63). `[VERIFIED]`
6. **Fancier temporal models did not pay.** A refinement stage (the MS-TCN idea) moved 0.609 to
   0.610, far inside the organisers' ±0.04 noise band. `[VERIFIED]`
7. **Open data exists and is the right kind.** 7 DFL matches carry per-frame possession and
   ball-status fields at 25 fps under CC-BY 4.0. `[VERIFIED]`
8. **Biggest untested upside:** recover the ball from the broadcast feed by syncing the two cameras
   in pitch space. Sync worked in simulation (0–2 frame error). Perfect tracking would add at most
   about +0.12 on the bench `[VERIFIED]`; the real gain is `[HYPOTHESIS]`.
9. **Biggest risks:** the real ATD's noise differs from my simulator, and Metrica's "controlled
   possession" is stricter than the DFL label. Both are unmeasured on this match. `[HYPOTHESIS]`
10. **Honest expectation for the real match:** well above the 0.2433 floor, below the 0.61 seen in
    simulation. I cannot put a verified number on it without touching the match.

---

## 2. What the metric and data force

Read from [evaluation/possession_metrics.py](../evaluation/possession_metrics.py) and the docs.

- `[VERIFIED]` A possession is a maximal run of `H` or of `A`. A `D` run splits one. So `H D H` is
  two possessions: missing a short stoppage merges two true possessions into one wrong one.
- `[VERIFIED]` Matching is one-to-one, same team, greedy by IoU, averaged over IoU 0.3–0.7. A
  flicker inside a long possession costs one false positive, and can also break the long match.
- `[VERIFIED]` `X` is forward-filled inside a half before spells are built, so it means "repeat my
  last label".
- `[VERIFIED]` The organisers state gaps below ~0.04 spellF1 are noise on the full match
  ([EVALUATION.md](../EVALUATION.md)). I treat any difference below that as a tie.
- `[VERIFIED]` Ground truth is "derived from corrected tracking": `D` where that tracking marks the
  ball dead, else the team it names as owning the ball ([CHALLENGE.md](../CHALLENGE.md)).

### The ball-gap structure (label-free, from the public baseline file)

`baseline_raw.csv` says `D` exactly where the ATD has no ball
([starter/baseline.py](../starter/baseline.py)), so its `D` runs are the ball outages. `[VERIFIED]`
Measured over the in-play windows:

| | count | median | mean | 10th–90th pct |
|---|---|---|---|---|
| ball-missing runs | 146 | 268 frames (10.7 s) | 439 frames (17.6 s) | 67–882 frames |
| ball-visible runs | 145 | 343 frames (13.7 s) | 633 frames (25.3 s) | 53–1,622 frames |

Almost no gap is between 25 and 125 frames long. `[VERIFIED]` That pattern suggests the tracker
already bridges short gaps and gives up on long ones. `[HYPOTHESIS]`

---

## 3. The test bench (other matches only)

**Data.** 7 matches from the open DFL dataset
([Bassek et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/),
[figshare](https://doi.org/10.6084/m9.figshare.28196177),
[mirror](https://huggingface.co/datasets/pysport/idsse-data)). `[VERIFIED]` Each has 25 fps
positions for all players and the ball, plus per-frame `BallPossession` (1 = home, 2 = away) and
`BallStatus` (0 = inactive, 1 = active). I map these to H/A/D directly.

**Do the labels mean the same thing?** Partly.

- `[VERIFIED]` DFL: about 437 possessions per match, mean 7.2 s, 45% of frames dead.
- `[VERIFIED]` The public 5 minutes here: 17 possessions, mean 14.2 s, 19.6% dead
  ([DATA.md](../DATA.md)). Five minutes is too small to compare shares.
- `[VERIFIED]` Diagnostic only: I shifted the DFL truth late by 1/3/8/25 frames and scored it
  against itself, as the organisers did for their table. DFL gives 0.991/0.960/0.862/0.557; the
  published table for this match is 1.000/0.982/0.893/0.654. So this match has fewer very short
  possessions than DFL.
- **Disclosure.** That published table is derived from the private ground truth. I used it as a
  description, and it influenced one small choice: the bench absorbs possessions shorter than 6
  frames into the previous team's possession. If you want to be strictly safe under rule 2, justify
  that filter from the challenge text ("a simple touch without control" is not a change) or drop
  it, and ask the organisers before calibrating anything to that table.

**Simulated ATD noise** ([bench.py](bench.py)). `[VERIFIED]` as code; realism is `[HYPOTHESIS]`.

- Ball outages drawn from the run-length quantiles in section 2, with 70% of onsets moved to a
  nearby "contested" frame (ball above 2 m, or 3+ players within 2.5 m).
- Players: anonymous, 0.2 m correlated jitter, bursty losses giving about 20.5 of 22 seen (the
  figure in [DATA.md](../DATA.md)).
- One trap found and fixed: DFL freezes the ball (speed exactly 0) on 100% of dead frames.
  `[VERIFIED]` A model reads "dead" straight off that. I replace the frozen stretch with a ball
  carried by the nearest player, then resting at the restart spot, hidden for 1 stoppage in 5.
  That replacement is my guess at what a real tracker shows. `[HYPOTHESIS]`

**Is the simulation believable?** `[VERIFIED]` The organisers' own baseline, re-implemented and
run on the simulated data, lands close to its published scores on the real match, with the same
ordering. I did not tune toward these.

| baseline window | simulated DFL spellF1 | published, real match |
|---|---|---|
| raw | 0.035 | 0.0486 |
| ±25 | 0.185 | 0.1996 |
| ±50 | 0.203 | 0.2433 |
| ±125 | 0.182 | 0.2242 |

**Protocol.** Leave one match out, train on the other six, score the held-out match with the
official scorer. Decoder settings for a match are chosen on the other six. All seeds fixed.

---

## 4. Results

### 4.1 Main table (mean of 7 held-out matches, 420 true possessions per match)

| system | spellF1 | macroF1 | predicted possessions |
|---|---|---|---|
| organisers' baseline, ±50 vote | 0.203 | 0.498 | 436 |
| hand-written rules (hold time + dead-ball rules) | 0.297 | 0.712 | 259 |
| frame model, raw argmax | 0.250 | 0.897 | 1,458 |
| frame model + majority vote ±25 | 0.589 | — | 330 |
| frame model + Viterbi | 0.602 | — | 330 |
| **frame model + min-duration Viterbi** | **0.609** | 0.909 (4 folds) | 326 |
| frame model + refinement stage + Viterbi | 0.610 | — | 300 |
| player-only model (no ball features) + min-duration Viterbi | 0.516 | — | 287 |
| label-free decoder pick (section 6.2), good sampler | 0.605 | — | — |

All `[VERIFIED]`. The four decoded rows from 0.589 to 0.610 are a tie under the ±0.04 noise band.

Frame-level detail of the frame model: dead-ball recall 0.961, precision 0.955; team accuracy when
both agree the ball is alive 0.900; accuracy 0.935 with the ball seen and 0.880 with it unseen.
`[VERIFIED]`

### 4.2 Where the remaining error is

Recall of true possessions at IoU 0.5, best decode, by true length: `[VERIFIED]`

| true length | share of possessions | found |
|---|---|---|
| under 1 s | 10% | 7% |
| 1–2 s | 17% | 28% |
| 2–4 s | 23% | 47% |
| 4–10 s | 27% | 69% |
| over 10 s | 22% | 90% |

Boundaries that are found are found on time: mean absolute error 3 frames. `[VERIFIED]` The loss
is missed and invented short possessions, not late boundaries. Since this match appears to have
fewer very short possessions than DFL (section 3), this error source may be smaller here.
`[HYPOTHESIS]`

### 4.3 Noise matching (4 held-out matches, one fixed decoder)

Every model is tested on the same target-like data; only the noise used in training changes.

| training noise | spellF1 | macroF1 | frame accuracy, ball seen | frame accuracy, ball unseen |
|---|---|---|---|---|
| none (clean tracking) | 0.358 | 0.558 | 0.601 | 0.501 |
| random per-frame ball dropout at 41% | 0.459 | 0.732 | 0.940 | 0.505 |
| block outages with the measured lengths | 0.616 | 0.910 | 0.941 | 0.891 |
| block outages + onsets biased to contested moments (same as test) | 0.626 | 0.909 | 0.940 | 0.890 |
| for reference: player-only model, matched noise | 0.516 | 0.876 | 0.880 | 0.896 |

All `[VERIFIED]`. Getting the block structure right is worth +0.16 spellF1 over random dropout and
+0.26 over clean training. Knowing *where* outages start added nothing measurable (0.616 vs 0.626
is a tie). That is good news: the part that matters is the part that can be measured from this
match without labels.

---

## 5. Ranked approaches

Effect sizes are on the simulated bench unless stated. Rule check refers to the five hard rules.

| # | approach | effect on spellF1 | confidence and basis | needs | rules |
|---|---|---|---|---|---|
| 1 | Supervised frame model on other matches + matched noise + symmetric features | 0.20 → 0.61 with decoding | High on the bench `[VERIFIED]`; medium for the real match `[HYPOTHESIS]` | DFL data, scikit-learn, laptop CPU | OK. External data is other matches, CC-BY, open to organisers |
| 2 | Structured decoding (vote, Viterbi, min-duration) | 0.25 → 0.59–0.61 | High `[VERIFIED]` | nothing extra | OK. Tune on other matches only |
| 3 | Player-only modelling for ball outages | keeps 0.52 with no ball | High `[VERIFIED]`; agrees with published 84–86% team accuracy and 0.92 in-play accuracy from players alone | same as 1 | OK |
| 4 | Hand-written rules only | 0.20 → 0.30 | Medium `[VERIFIED]` for my rule set; a better rule set could do more | none | OK |
| 5 | Ball recovery from the broadcast feed | at most about +0.12 (0.63 → 0.75) if every outage were filled | Sync `[VERIFIED]` in simulation; real gain `[HYPOTHESIS]` | broadcast ATD + homography | OK, all automatic |
| 6 | Smart Tagging as prior and as a label-free check | unknown | `[HYPOTHESIS]`. 59 set pieces mark restarts; phases carry team tags | provided XML | OK. It is automatic output |
| 7 | Deep temporal segmentation models (MS-TCN, ASFormer, DiffAct, ASRF) | about 0 over row 2 here | Refinement stage gave +0.001 `[VERIFIED]`; full neural versions untested | PyTorch | OK |
| 8 | Learned ball-trajectory imputation (set transformer + Bi-LSTM) | unknown, likely small over row 3 | `[HYPOTHESIS]`. Row 3 uses the same signal more directly | PyTorch, more data | OK |
| 9 | Re-detect the ball in the tactical video | unknown | `[HYPOTHESIS]`. Ball is a few pixels wide in a wide shot | video, GPU, open ball detector | OK if run locally |
| 10 | Self-training on this match with pseudo-labels | unknown | `[HYPOTHESIS]` | same as 1 | OK. Pseudo-labels from code are allowed |
| 11 | Full explicit-duration model (HSMM) | +0.007 over Viterbi, a tie | Min-duration version `[VERIFIED]`; full duration prior untested | nothing extra | OK |
| 12 | Video action spotting (T-DEED style) | unknown | `[HYPOTHESIS]`. Trained on broadcast, not tactical wide shots | video, GPU, SoccerNet labels | Doubtful: SoccerNet videos need an NDA, so "accessible to organisers" is unclear |
| 13 | Whistle / audio cues | unknown | Whether the video has audio is unknown to me | video | OK if local |
| — | Tune anything on the public 5 minutes | — | — | — | **Breaks rule 2. Dropped** |
| — | Send frames or tracking to a hosted model | — | — | — | **Breaks rule 4. Dropped** |
| — | Change-point detection (PELT) | — | Same penalised-cost family as Viterbi, without class structure. Not tested | — | Dropped as dominated |

### Evidence from the literature

**Temporal action segmentation.**
- `[VERIFIED]` The segmental F1@k score was proposed because it penalises over-segmentation and
  ignores small temporal shifts ([Lea et al. 2017](https://arxiv.org/abs/1611.05267)). spellF1 is
  the same idea with one-to-one IoU matching.
- `[VERIFIED]` MS-TCN stacks stages that refine the previous stage's prediction and adds a
  smoothing loss against over-segmentation
  ([Abu Farha & Gall 2019](https://arxiv.org/abs/1903.01945),
  [MS-TCN++](https://arxiv.org/abs/2006.09220)).
- `[VERIFIED]` ASRF adds a boundary-regression branch whose boundaries refine frame-wise output
  ([Ishikawa et al. 2021](https://arxiv.org/abs/2007.06866)).
- `[VERIFIED]` ASFormer and DiffAct are transformer and diffusion variants of the same iterative
  refinement ([ASFormer](https://arxiv.org/abs/2110.08568),
  [DiffAct](https://arxiv.org/abs/2303.17959)); survey:
  [Ding et al. 2022](https://arxiv.org/abs/2210.10352).
- `[VERIFIED]` Explicit length modelling in Viterbi decoding gave large gains in weakly supervised
  segmentation ([Richard et al. 2018](https://arxiv.org/abs/1805.06875)).
- `[VERIFIED]` SSTDA adapts a segmentation model to unlabeled target videos with self-supervised
  domain prediction ([Chen et al. 2020](https://arxiv.org/abs/2003.02824)).
- **Fit to this problem.** These models were built for 10–50 classes with learned video features.
  Here there are 3 classes and 7 training matches, and my tree-based refinement stage added
  nothing. `[VERIFIED]` I expect little from the heavier models without far more labelled data.
  `[HYPOTHESIS]`

**Possession and events from tracking.**
- `[VERIFIED]` A rule-based system assigns possession when the ball is inside a player's
  "possession zone" and detects set pieces from player configurations while the ball is dead
  ([Vidal-Codina et al. 2022](https://arxiv.org/abs/2202.00804)). The paper notes providers signal
  dead ball differently: a flag (Tracab) or missing ball data.
- `[VERIFIED]` Individual possession from position data reached F-score 0.88 for ball actions on
  Bundesliga matches ([Link & Hoernig 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/)).
- `[REPORTED]` ETSY syncs events to tracking with rules
  ([blog](https://dtai.cs.kuleuven.be/sports/blog/etsy:-a-rule-based-approach-to-event-and-tracking-data-synchronization),
  [code](https://github.com/ML-KULeuven/ETSY)).
- `[VERIFIED]` ELASTIC syncs using tracking-only features and detects the end of pass-like events
  ([Kim et al. 2025](https://arxiv.org/abs/2508.09238)).
- **Fit.** Sync methods need event data for this match, which rule 3 forbids. Their touch-detection
  features (ball acceleration, player–ball distance) are reusable; the pipelines are not.

**Ball missing.**
- `[VERIFIED]` With the ball fully hidden, a set-transformer + Bi-LSTM model predicted the
  possessing team with 83.96% accuracy and the ball within 3.7 m on average, trained partly on
  Metrica sample games ([Kim et al. 2023](https://arxiv.org/abs/2306.08206)).
- `[VERIFIED]` TranSPORTmer imputes hidden agents and classifies states including possession and
  out-of-play with one transformer ([Capellera et al. 2024](https://arxiv.org/abs/2410.17785)).
- `[VERIFIED]` In-play vs stopped was classified from player positions alone with accuracy 0.92
  and F1 0.93 on 47 test matches
  ([Lang et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/)).
- My player-only numbers (team accuracy 0.878, dead-ball recall 0.94) agree with both. `[VERIFIED]`

**Video.**
- `[VERIFIED]` T-DEED won SoccerNet 2024 Ball Action Spotting with 73.39 mAP@1
  ([results paper](https://arxiv.org/abs/2409.10587), [model](https://arxiv.org/abs/2404.05392)).
- `[VERIFIED]` SoccerNet 2025 added Team Ball Action Spotting, which assigns each action to a team
  ([results paper](https://arxiv.org/abs/2508.19182)).
- `[VERIFIED]` SoccerNet-GSR is 200 clips of 30 s for tracking and identification on a minimap
  ([Somers et al. 2024](https://arxiv.org/abs/2404.11335)). It has no possession labels that I
  found; unknown.
- `[VERIFIED]` WASB is an open ball detector covering soccer
  ([paper](https://arxiv.org/abs/2311.05237), [code](https://github.com/nttcom/WASB-SBDT)).

**Decoding and transfer.**
- `[REPORTED]` Hidden semi-Markov models give each state an explicit duration distribution
  ([Yu 2010](https://doi.org/10.1016/j.artint.2009.11.011)).
- `[REPORTED]` Minimum Bayes-risk decoding picks the output with the best expected score under the
  model's posterior ([Kumar & Byrne 2004](https://aclanthology.org/N04-1022/)).
- `[VERIFIED]` Domain randomisation: train on varied simulated corruption so the real domain looks
  like one more variation ([Tobin et al. 2017](https://arxiv.org/abs/1703.06907)).
- `[VERIFIED]` Self-training with a noised student improved robustness on shifted test sets
  ([Xie et al. 2019](https://arxiv.org/abs/1911.04252)).

---

## 6. Invented methods

I searched for each and did not find it in the literature. That is weaker than proving it is new.

### 6.1 Degradation twin: fit the noise, then train through it — TESTED

- **Mechanism.** Measure this match's ATD noise with code and no labels: ball-present and
  ball-missing run-length distributions, players seen per frame. Push clean, labelled tracking from
  other matches through a simulator with those statistics. Train on the result.
- **Builds on.** Domain randomisation (Tobin et al., above); the gap table in section 2.
- **Why it should win here.** The outages are 10-second blocks. A model that never trained without
  a ball for 10 seconds has nothing to fall back on.
- **Result.** `[VERIFIED]` See section 4.3. Random per-frame dropout at the correct 41% rate is not
  enough; the block structure is what matters.
- **How it could fail.** Real ATD noise has parts I did not simulate: identity swaps, duplicate
  tracks, wrong team labels, false ball detections, and whatever the ball does while play is
  stopped.
- **Falsifying test.** Done on other matches (section 4.3). On this match it can only be checked
  indirectly, by the Smart Tagging agreement in 6.4.

### 6.2 Label-free decoder selection by expected spellF1 — TESTED, mixed

- **Mechanism.** Draw posterior label sequences from the decoder's chain (forward filter, backward
  sample). Score each candidate decode against the samples with the real spellF1 code. Keep the
  candidate with the highest average. No labels involved, so it can run on this match.
- **Builds on.** Minimum Bayes-risk decoding (Kumar & Byrne, above). I found no use of it with a
  segment-IoU metric in action segmentation. `[HYPOTHESIS]` on novelty.
- **Why it should help.** You are not allowed to tune smoothing on this match. This picks a
  smoothing strength from the match's own probabilities.
- **Result.** `[VERIFIED]` With sensible sampler settings the label-free pick scored 0.594–0.605,
  level with settings tuned on labelled other matches (0.609). With a badly set sampler it picked
  the raw argmax and scored 0.25. The model's expected score was 0.93 against an actual 0.60, so
  the posterior is badly overconfident.
- **Verdict.** No gain over tuning on other matches. Useful only as a cross-check. Not in the
  recommended pipeline.

### 6.3 Cross-camera ball recovery by pitch-space sync — sync TESTED in simulation, gain UNTESTED

- **Mechanism.** Both recordings give player positions in the same normalised pitch coordinates.
  For a candidate time offset, measure the mean distance from each broadcast player to the nearest
  same-team tactical player. The true offset minimises it. Then, frame by frame, keep the broadcast
  ball only where that distance is small, which rejects replays, and use it to fill tactical
  outages.
- **Builds on.** [DATA.md](../DATA.md): shared coordinates, broadcast ball visible on 50.5% of its
  frames, broadcast homography valid on 65.8%.
- **Why it should help.** A broadcast camera follows the ball; a wide tactical camera loses it.
  Their outages should overlap less than chance. `[HYPOTHESIS]`
- **Result.** `[VERIFIED]` in simulation ([run_sync_sim.py](run_sync_sim.py)): with a hidden
  offset, a partial moving view, 0.4 m noise, close-ups and replays, the offset was recovered with
  errors of 1, 2 and 0 frames in three trials. Replays were flagged with recall 0.93–1.00 and
  precision 0.44–0.55, so about half the flagged frames were good frames thrown away.
- **Size of the prize.** `[VERIFIED]` The same pipeline trained and tested with no outages and no
  player noise scores spellF1 0.745, against 0.626 with target-like noise (same 4 held-out
  matches, same decoder). So perfect tracking is worth at most about +0.12, and ball recovery can
  only claim part of that. The 0.745 is slightly flattering: in that setting an unseen ball only
  ever happens during a stoppage.
- **How it could fail.** The broadcast file may not be one continuous real-time recording, in which
  case the offset changes at every edit. Broadcast positions may be too noisy. The two ball tracks
  may fail at the same moments.
- **Falsifying test on this match, within the rules.** Run the offset search per half. A single
  sharp minimum that stays put across several sub-windows confirms a constant offset. No labels
  needed.

### 6.4 Smart Tagging agreement as a rule-safe validation signal — UNTESTED

- **Mechanism.** Phases such as Build Up, Progression and Attacking Transition carry a team tag;
  set pieces mark restarts. Compute (a) the share of frames inside attacking phases where the
  predicted team equals the tag, and (b) the share of set-piece starts that fall within a second of
  a predicted `D`→alive change. Use those two numbers to compare pipeline versions.
- **Builds on.** [DATA.md](../DATA.md): 638 automatic intervals, team tags on 602 of them.
- **Why it matters.** It is the only measurement on this match that is neither hand-made nor the
  ground truth, so it is the only legal way to choose between versions here. `[HYPOTHESIS]`
- **How it could fail.** The tags may come from the same ball track and share its errors, which
  would reward copying the tracker. Throw-ins have no team tag.
- **Falsifying test.** Not possible on DFL, which has no such tags. `[HYPOTHESIS]`, untested.

### 6.5 Team-swap symmetrisation — in the pipeline, not ablated

- **Mechanism.** Features are computed in a frame where one team attacks to the right. The model
  is run on the observation and on its mirror with teams exchanged; the two outputs are averaged.
  H and A are then treated identically by construction, and training data doubles.
- **Status.** `[VERIFIED]` it runs; its separate contribution is not measured. `[HYPOTHESIS]`

---

## 7. Usable resources

### Data from other matches

| dataset | what it contains | licence |
|---|---|---|
| [DFL open data](https://doi.org/10.6084/m9.figshare.28196177) ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)) | 7 Bundesliga / 2. Bundesliga matches 2022/23. 25 fps positions, ball height, per-frame possession and ball status, synced events | CC-BY 4.0 `[VERIFIED]` |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) | 3 anonymised matches, 25 fps tracking + events, same vendor and coordinate convention as this match. No per-frame possession field; I derived one from events (385 and 374 possessions, 39–41% dead in games 1 and 2) | No formal licence. README asks for responsible use and acknowledgement `[VERIFIED]` |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) | 10 A-League matches 2024/25, broadcast tracking at 10 fps with a per-frame possession field (player and team), plus dynamic events | MIT `[VERIFIED]` |
| [PFF FC World Cup 2022](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data) | 64 matches, broadcast tracking + events | Access by form; terms unknown `[REPORTED]` |
| [FOOTPASS](https://arxiv.org/abs/2511.16183) | Play-by-play action spotting over full matches with tactical context | Unknown |
| [Google Research Football](https://github.com/google-research/football) | Simulator. Observations include `ball_owned_team` and `game_mode` (throw-in, kick-off…), so unlimited synthetic labels | Apache-2.0 `[VERIFIED]` |
| [SoccerNet](https://github.com/SoccerNet/sn-gamestate) | Broadcast video tasks | Videos behind an NDA `[REPORTED]` |

I used only the DFL matches for training and testing. Metrica samples were used once, for the
label comparison in section 3.

### Code and models

| item | use | licence |
|---|---|---|
| [kloppy](https://github.com/PySport/kloppy) | loads Metrica EPTS, Sportec/DFL, SkillCorner | BSD-3 `[VERIFIED]` |
| [scikit-learn HistGradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) | the frame model; accepts NaN inputs, which is how "ball unseen" is passed in | BSD-3 |
| [MS-TCN](https://github.com/yabufarha/ms-tcn) | reference temporal model | licence not stated in GitHub metadata |
| [ASFormer](https://github.com/ChinaYi/ASFormer), [DiffAct](https://github.com/Finspire13/DiffAct), [ASRF](https://github.com/yiskw713/asrf), [SSTDA](https://github.com/cmhungsteve/SSTDA) | reference temporal models | MIT `[VERIFIED]` |
| [T-DEED](https://github.com/arturxe2/T-DEED) | video event spotting | GPL-3.0 `[VERIFIED]` |
| [WASB](https://github.com/nttcom/WASB-SBDT) | ball detection with soccer weights | MIT `[VERIFIED]` |
| [ETSY](https://github.com/ML-KULeuven/ETSY) | event–tracking sync rules | licence not stated in GitHub metadata |

### This repo's research code

```bash
research/.venv/bin/python -I research/dfl_load.py data/external/dfl data/external/dfl_npz
research/.venv/bin/python -I research/run_model.py target target all     # frame model, 7 folds
research/.venv/bin/python -I research/run_decode.py target_target_all    # decoder comparison
research/.venv/bin/python -I research/run_stage2.py target_target_all    # refinement stage
research/.venv/bin/python -I research/run_model.py iid target all 0,2,4,6    # wrong-noise variants
research/.venv/bin/python -I research/run_compare.py target_target_all iid_target_all
research/.venv/bin/python -I research/run_rules.py                       # rule comparator
research/.venv/bin/python -I research/run_sync_sim.py                    # camera sync simulation
```

`data/` is gitignored, so the downloaded matches stay out of commits.

---

## 8. Implementation sketches

### A. Degradation-twin model + min-duration decode (recommended, tested)

- **Inputs.** Tactical ATD (positions, team of each slot, ball). DFL matches.
- **Steps.**
  1. Parse the ATD into per-half arrays in metres, pitch centre at the origin. Nothing uses the
     sign of `y`, so the top-left origin convention does not matter.
  2. By code, measure the ball run-length quantiles and the players-seen distribution. Write them
     to a small JSON that the simulator reads, replacing the constants in [bench.py](bench.py).
  3. Degrade the DFL matches with that profile (fixed seeds), build features in both orientations,
     fit `HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=63, random_state=0)`.
  4. Predict each graded half, average the two orientations.
  5. Decode: min-duration Viterbi, switch cost 5, minimum 25 alive frames, 125 dead frames. These
     were the most-picked settings on held-out DFL matches.
  6. Outside the graded halves write `X`. For the graded stretch before the second-half kick-off,
     write `D` up to the start of the Smart Tagging "Kick Off (Start)" set piece of the second half,
     which is found by code. Hard-coding frame 113,742 from the FAQ is a constant tied to a frame;
     ask before relying on it.
  7. Write the CSV, run `evaluation/validate_submission.py`.
- **Output.** `outputs/<team>.csv`.
- **Determinism.** Fixed seeds, pinned package versions, DFL files pinned by hash. `[VERIFIED]` Two complete runs
  (simulate noise, build features, train, predict) on this laptop gave bitwise-identical
  probabilities. Across machines or library versions it is untested, so also
  commit the fitted model file and its SHA-256 so the organisers can regenerate without retraining.

### B. Broadcast ball recovery (untested on real data)

- **Inputs.** Both ATDs, broadcast homography `valid` flag.
- **Steps.**
  1. Per half, search the time offset that minimises the player-matching distance on a few hundred
     valid broadcast frames. Check it is stable across sub-windows.
  2. Per broadcast frame, compute the same distance at the chosen offset. Keep the frame if it is
     under a threshold set on DFL simulation.
  3. Where the tactical ball is missing and a kept broadcast frame has a ball, copy it across.
  4. Re-measure the noise profile on the merged ball track, then run pipeline A unchanged.
- **Output.** A merged ball track; then the same CSV.
- **Determinism.** Exhaustive integer search, no randomness beyond a seeded choice of probe frames.

### C. Smart Tagging as prior and referee (untested)

- **Inputs.** The Smart Tagging XML, the frame probabilities from A.
- **Steps.**
  1. Convert set pieces to restart frames (`round(seconds * 25)`).
  2. Add a bonus in the decoder for a `D`→alive change near a restart frame, and for `D` in the two
     seconds before it. Set the bonus on DFL using DFL's own set-piece events as stand-ins.
  3. Compute the two agreement numbers from 6.4 for every pipeline version. Use them, never the
     public ground truth, to choose.
- **Output.** A decoded label file plus a two-number report per version.
- **Determinism.** Pure arithmetic.

---

## 9. Open questions and unsolved problems

1. **What does the ATD ball do while play is stopped?** My simulator guesses. The player-only
   result (0.52) does not depend on that guess; the 0.61 partly does.
2. **How strict is "controlled possession"?** DFL switches possession more readily than this match
   appears to. A stricter relabelling of the training data would predict fewer, longer possessions.
   Choosing how strict, without tuning on this match, is unsolved.
3. **May the published lateness table be used for calibration?** It is derived from the private
   ground truth. Ask the organisers.
4. **Short possessions.** Half of DFL possessions last under 4 seconds and most of those are
   missed. Nothing I tried fixed this.
5. **Overconfident posterior.** Expected spellF1 0.93 vs actual 0.60. A calibrated segment-level
   posterior would make 6.2 worth using.
6. **Unsimulated tracker faults.** Identity swaps, duplicates, wrong team labels, false ball
   detections.
7. **Is the broadcast timeline continuous?** Decides whether 6.3 is a one-hour job or a research
   project.
8. **Does the video have audio?** Unknown to me. A whistle detector would be a direct dead-ball cue.
9. **Only 7 training matches from one country.** More labelled tracking with the same possession
   definition would help more than a bigger model. `[HYPOTHESIS]`
10. **No legal way to measure the real score before submitting.** Section 6.4 is the best proxy I
    can offer, and it is untested.
