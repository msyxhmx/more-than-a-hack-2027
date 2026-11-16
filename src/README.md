# Our pipeline

Two lanes, one merge. Each lane writes one file into `outputs/`; `merge.py` combines them into the
submission. Either lane can run the whole pipeline alone, because the merge falls back to the
baseline for a missing file.

| file | owner | writes | contents |
|---|---|---|---|
| `dead_ball.py` | Ilham | `outputs/dead.csv` | `frame,dead` — 199,319 rows, `1` = ball out of play, `0` = alive |
| `possession.py` | Syahmi | `outputs/team.csv` | `frame,team` — 199,319 rows, `H` / `A` / `X` (X = not sure) |
| `merge.py` | Ilham | `outputs/<team_name>.csv` | the submission: `frame,label`, `H` / `A` / `D` / `X` |

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
