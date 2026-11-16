# What spellF1 rewards (task P01)

Notes from `EVALUATION.md` and `evaluation/possession_metrics.py`, the scoring specification.

## How a file is scored

1. **Graded frames** are where the ground truth is not X: the two halves, 11,185–90,651 and
   110,808–190,197. Everything temporal is computed inside one half, never across halftime.
2. **X is filled first.** Inside each half, every X is replaced by the last non-X label (back-filled at
   the start of a half). So for spellF1, X means "repeat my previous answer", not "no answer".
3. **A possession is a maximal run of H or of A.** D is a stoppage and ends a possession. H directly
   followed by A is two possessions; so is H-D-H.
4. **Matching:** predicted and true possessions of the same team are paired one-to-one, greedily by
   temporal IoU (overlap / union). F1 is computed at IoU 0.3, 0.4, 0.5, 0.6 and 0.7 and averaged.
5. **macroF1** (tie-break) is per-frame F1 for H, A and D, averaged; X counts as a miss there.

## What follows

| What we do | Effect on spellF1 |
| --- | --- |
| A one-frame blip of the wrong team inside a long possession | Splits it into three; usually none of the pieces matches. Very expensive. |
| Missing a short stoppage between two possessions of the same team | Glues them into one; at most one of the two can match, and only at low IoU. |
| Inventing a stoppage inside a possession | Splits it in two, same cost as a blip. |
| Being late at a transition | 1 frame free, 3 frames −0.02, 8 frames −0.11, 1 s −0.35 (organisers' table). |
| X inside a possession | Free (filled with the holder). Safer than a guess that could flicker. |
| X across a hand-over | The old team is carried over the transition, so the new possession starts late. |
| Too many possessions | Each extra one is a false positive; precision falls. |

So: possession count and boundaries matter far more than per-frame accuracy. Fewer, cleaner spells
with transitions on time beat a flickering answer with higher frame accuracy (see the four
baselines: macroF1 moves 0.04 across them, spellF1 moves fivefold).

## Noise

- spellF1 differences under ~0.04 on the full match are noise; macroF1 under ~0.01.
- The public 5 minutes hold 17 possessions; one missed possession moves spellF1 by about 0.06 there.
  Tuning on it picks the wrong version (smooth125 looks best publicly, smooth50 is best on the full
  match). Rule: public window for correctness only.
