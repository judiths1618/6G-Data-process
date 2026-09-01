# timeline regularization — design doc

Status: **implemented (v0.2)** · Date: 2026-08-31

**Scope:** why the pipeline puts a time series onto a uniform grid, when it
refuses to, and how the decision is made and recorded. Implemented in
`dataops._preprocess_impl.regularize` / `regularize_segments`, configured under
`timeline:` in `config/dataops.yaml`, surfaced in `meta.json` →
`regularization.decision` and in the dashboard's **Overview** tab.

The short version: **there is no single regularization that is correct for every
dataset**, so this module does not try to have one. It applies a small, ordered
set of rules, each with a stated reason, and *records why it decided what it
decided* for every bundle it writes. When no rule can be satisfied honestly it
declines to regularize rather than producing a grid that looks uniform and is
mostly invented.

---

## 1. Why regularize at all

Nothing in cleaning, validation, or remediation needs a uniform time axis. One
thing downstream does: **the imputation models.**

- WaveStitch+ slides Hann-weighted windows of fixed *length in samples* over the
  series. On an irregular axis a 100-sample window covers a different amount of
  real time at every position, so the window's meaning drifts.
- PyPOTS (`--window`) tiles fixed-length windows for the same reason.
- Diffusion/DDIM sampling assumes neighbouring array positions are neighbouring
  instants.

So regularization exists to serve the models, not the data. That framing decides
everything that follows: **a grid is only worth building if it makes the model's
assumption true.** A grid that is 95% synthetic does not make the assumption
true — it replaces the data with the imputer's own prior and then scores the
imputer on it.

## 2. Why one rule cannot fit every dataset

The four EUR subsets are the same experiment, collected by the same harness, and
they need four different answers. Measured, not assumed:

| subset | rows | span | campaigns | per-campaign cadence | one-grid emptiness |
|---|---|---|---|---|---|
| amf | 27,413 | 16.8 d | 2 | 13, 13s | 75.4% @ 13s |
| golang | 58,763 | 143.8 d | 6 | 69, 63, 37, 42, 9, 3s | 91.7% @ 20s |
| python | 15,308 | 132.3 d | 4 | 36, 41, 40, 8s | 97.5% @ 20s |
| rabbitmq | 22,556 | 45.4 d | 4 | 56, 70, 70, 69s | 61.7% @ 67s |

("one-grid emptiness" is measured at the cadence `regularize()` would infer for
the whole series — the median of *all* steps — which is the grid that would
actually have been built.)

Three independent things vary:

1. **Cadence varies *within* a dataset.** golang was sampled every 69s in its
   first campaign and every 3s in its last — a 23× range. Any single `base_dt`
   is simultaneously too coarse for one end and too fine for the other.
2. **Collection pauses dominate the span.** golang has a 107.9-day pause between
   two collection campaigns; it contributes almost all of the 91.7% emptiness
   while containing no data at all.
3. **Some series are bursty at every scale.** python's median step is 8s against
   a ~61s mean — dense bursts separated by minute-scale idle. This is not a
   pause artifact, so no amount of splitting fixes it (verified: at split
   thresholds from 86400s down to 900s, still only 9/17 segments fit).

A fixed `base_dt`, a fixed grid, or a fixed sparsity budget will be wrong for at
least one of these. What generalizes is not a parameter, it's a **procedure**.

## 3. The decision procedure

Applied in order. Each step states the reason it exists.

```
1. resolve row identity          (timeline.collision_policy)
   └─ a grid admits one row per instant, so tied timestamps must reduce first
2. enforce monotonicity          (timeline.disorder_policy)
   └─ a grid is an ordered axis; backward jumps have no position on it
3. estimate cadence              median of positive within-run steps
   └─ must match infer_base_dt, or the report and the grid disagree
4. split at collection pauses    (timeline.segment_gap_seconds)
   └─ a pause is not sampling; a grid across it is empty by construction
5. per campaign: grid or refuse  (validation.sparse_skip_pct)
   └─ a mostly-synthetic grid is the imputer's prior, not the data
6. all-or-nothing                (timeline.require_all_segments)
   └─ a mixed bundle splits two sampling regimes across train/test
```

Outcomes, recorded as `regularization.decision.code`:

| code | outcome | meaning |
|---|---|---|
| `fits_guard` | `single_grid` | one campaign, within the sparsity budget |
| `all_campaigns_fit` | `per_segment` | every campaign got its own grid + cadence |
| `too_sparse` | `not_regularized` | a grid would be mostly synthetic |
| `mixed_regimes_rejected` | `not_regularized` | only some campaigns fit; see §4.3 |

## 4. The three judgement calls, and their justification

Each of these is a threshold. None is derivable from first principles; each is
stated here with what it trades and how sensitive the result is to it.

### 4.1 `sparse_skip_pct = 80` — how synthetic is too synthetic

At 80%, four of every five rows handed to the imputer would be rows it invented.
Beyond that, holdout metrics increasingly measure the imputer's self-consistency
rather than its accuracy against real data.

The number is a convention, not a discovery. Its role is to be **explicit and
auditable**: the bundle records the measured emptiness next to the budget, so a
reader can disagree with the threshold and see immediately what changing it
would admit.

It also happens not to be load-bearing on these datasets. Across all 16
campaigns the measured emptiness is

```
accepted: -4.7 11.0 14.1 17.3 19.5 22.3 25.7 30.0 36.5 62.5 63.2 72.8 75.4 78.5
refused:                                                              86.9 92.6
```

— nothing falls in **[78.5, 86.9]**, so any threshold in that window produces
identical outcomes. The campaigns separate on their own; 80 merely sits in the
gap. A dataset whose campaigns land inside that window is one where the
threshold really is deciding, and that is exactly when it should be reviewed
rather than trusted.

### 4.2 `segment_gap_seconds = 86400` — what counts as a pause

A gap longer than a day is a decision by whoever ran the experiment to stop and
resume; a gap shorter than that is sampling behaviour. One day is the natural
unit for a benchmark harness driven by human working rhythm.

Sensitivity was measured rather than assumed:

| threshold | golang segments (fit) | python segments (fit) |
|---|---|---|
| 86400s | 6 (6) | 4 (2) |
| 7200s | 9 (9) | 4 (2) |
| 3600s | 13 (13) | 7 (4) |
| 900s | 21 (21) | 17 (9) |

golang is already fully resolved at 86400s; splitting further shrinks the grid
modestly (141k → 113k rows) without changing the outcome. python is *never*
resolved, which is the evidence for §2's third point: its sparsity is
within-campaign, so this knob is the wrong tool for it. The default is therefore
chosen to be the largest value that solves the cases it can solve — smaller
values fragment the series for no benefit.

### 4.3 `require_all_segments = true` — homogeneity over coverage

The one rule here that was learned from a failure rather than reasoned in
advance. Gridding only the campaigns that fit produced, on python:

- campaigns 1–2 gridded (68.5% and 48.3% empty), 9,435 rows — **all in train**
- campaigns 3–4 irregular (0% and 7% empty), 8,358 rows — **mostly in test**

The chronological 80/20 split put one sampling regime on each side: train was
36.2% NaN gridded data, test was dense irregular data. A model trained on that
is not being evaluated on the distribution it was trained on, and the resulting
holdout numbers mean nothing.

So the rule is: **regularize every campaign or none.** A dataset that keeps its
irregular timestamps is honest and internally consistent; a half-gridded one is
neither. Set `require_all_segments: false` only alongside a split that keeps
both regimes on both sides.

## 5. What each subset gets, and why

Produced by the procedure, not by per-dataset tuning:

| subset | outcome | rationale |
|---|---|---|
| **amf** | `per_segment` 2/2 | 1 pause; both campaigns 13s; 75.4% → 72.6% empty |
| **golang** | `per_segment` 6/6 | 5 pauses; cadences 69→3s; 91.7% → 63.6% empty; 527 step sizes → 11 |
| **rabbitmq** | `per_segment` 4/4 | 3 pauses; cadences ~56–70s; 61.5% → 11.4% empty |
| **python** | `not_regularized` | 4 campaigns, only 2 fit; refused to ship mixed regimes |

python is a **deliberate non-result**, not a gap in coverage. Its bundle keeps
the original timeline, train and test share one regime, and `meta.json` records
why plus the three things that would change the answer.

## 6. Applying this to a new dataset

1. Run the pipeline. Read `regularization.decision.rationale` — it is written
   for this dataset with its own numbers.
2. If the outcome is `not_regularized`, `decision.alternatives` lists what would
   change it. Choose on the data's meaning, not to make the warning go away:
   - sparsity from **pauses** → lower `segment_gap_seconds`
   - sparsity from **bursts** → a coarser `base_dt`, accepting that
     short-interval structure is averaged away
   - genuinely sparse data → leave it irregular and use imputers that do not
     assume a uniform axis
3. If it is `per_segment`, check `regularization.segments`: a campaign with
   negative emptiness (more rows than grid points, e.g. rabbitmq campaign 1 at
   −4.7%) means its `base_dt` is coarser than parts of that campaign sample.

## 7. Known limits

- **Cross-campaign windows.** The output is uniform *within* campaigns with real
  jumps between them. A model window spanning a boundary still crosses a
  discontinuity. `timeline.sweep_aware` emits a `run` column for splitting on;
  no runner consumes it yet.
- **`base_dt` is per campaign, not per regime.** A campaign that is itself
  bimodal (python's) gets one median cadence that suits neither mode.
- **No per-dataset `base_dt` override.** The only way to force a coarser grid
  today is to pass `base_dt` to `preprocess_csv` directly; it is not plumbed
  through `config/dataops.yaml`.
- **The sparsity budget is global.** `sparse_skip_pct` applies to every campaign
  equally, though a short campaign and a long one arguably deserve different
  budgets.

## 8. Contract stability

Regularization changed; **the prepared-bundle contract did not**. `meta.json`,
`train.csv`, `test_input.csv`, `test_gt.csv`, `eval_holdout_mask.npy`,
`col_masks/`, and `scaler/` keep their names, shapes, and meanings, so no
imputation runner (`Darts_app`, `ImputeGAP_app`, `PyPOTS_app`,
`WaveStitchPlus_app`) required any modification. `meta.json` only *gains* keys:
`regularized`, `regularization.segments`, and `regularization.decision`.

The evaluation contract is verified per run: no cell selected by
`eval_holdout_mask` may have unknown ground truth. On the current subsets that
holds exactly — 10,458 / 11,655 / 6,315 / 3,720 scored cells, zero with unknown
truth.

