# Our pipeline

Two lanes, one merge. Each lane writes one file into `outputs/`; `merge.py` combines them into the
submission. Either lane can run the whole pipeline alone, because the merge falls back to the
baseline for a missing file.

| file | owner | writes | contents |
|---|---|---|---|
| `dead_ball.py` | Ilham | `outputs/dead.csv` | `frame,dead` — 199,319 rows, `1` = ball out of play, `0` = alive |
| `possession.py` | Syahmi | `outputs/team.csv` | `frame,team` — 199,319 rows, `H` / `A` / `X` (X = not sure) |
| `merge.py` | Ilham | `outputs/<team_name>.csv` | the submission: `frame,label`, `H` / `A` / `D` / `X` |

### The tree model (main approach)

Ilham's gradient-boosted model + minimum-duration Viterbi predicts one H/A/D label per frame, so it
fills both lanes at once. It ends with:

```python
from lanes import write_lanes
write_lanes(labels, "outputs/model")   # labels: 199,319 strings, H / A / D / X
```

That writes `outputs/model/dead.csv` and `outputs/model/team.csv` and checks them against the
contract. The rule model (`possession.py`) and the baseline stay as fallbacks in `outputs/`.

    python src/score.py --lanes outputs/model --log "model v1"
    python src/merge.py --team-name <team_name> --lanes outputs/model

The tree model is `src/model.py` (task M01). It trains on the 7 open DFL matches, which
`src/dfl.py` downloads from a pinned mirror and checks against SHA-256 hashes the first time it
runs (about 2.7 GB; CC-BY 4.0). Training happens inside every run, so the submission is rebuilt
from data and code alone, with no saved model file. The research behind it, and the bench it was
tested on, are in `research/` (start with `research/REPORT.md`).

    python src/check_tracking.py     # PASS / FAIL sanity checks on the match data and model output

`check_tracking.py` prints no numbers, so its output is safe to share.

`src/lanes.py` holds the contract code (frame count, graded halves, lane checks, merge, writers);
`score.py` and `merge.py` both use it, so they cannot disagree. `python src/lanes.py` runs a
self-test on random labels.

The main approach was chosen on leave-one-game-out results on the open Bundesliga games, never by
comparing versions on the public 5 minutes.

## Regenerate the submission (one command)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lawan.txt   # macOS/Linux: .venv/bin/python
.venv/Scripts/python src/run.py --team-name Lawan
```

`run.py` reads `data/` (or `MTAH_DATA_DIR`), runs the method in `DEFAULT_METHOD`, merges, validates
and prints the file's SHA-256. `--method rules|model|baseline` picks one explicitly. The tree model
plugs in as `src/model.py` with `main(out_dir)`; then set `DEFAULT_METHOD = "model"`.

## Merge rule

1. Outside the graded halves (frames 11,185–90,651 and 110,808–190,197) → `X`
2. Else if `dead == 1` → `D`
3. Else → the `team` value (`H`, `A` or `X`)

Then always run `python evaluation/validate_submission.py outputs/<team_name>.csv`.

## Rules we keep

- Labels come from code only — no hand labels or hand corrections on this match.
- The public 5-minute ground truth checks that the pipeline is correct; never use it to tune or to
  choose between versions.
- Never paste match data, tracking rows or video frames into a cloud LLM. AI may write code only.
- `data/` and `outputs/` are gitignored. Never commit the match data.
