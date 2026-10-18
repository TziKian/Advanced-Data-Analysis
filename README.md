# AIT403 · Advanced Data Analysis — Tutorial 1 Dashboard

Interactive Streamlit dashboard for the three Tutorial 1 experiments, built on
top of the original notebooks:

| Task | Question | Experiment |
|------|----------|------------|
| 1 | How does KNN cope with irrelevant features? | Inject up to 10,000 noise columns into the 64-feature digits data and measure accuracy, fit/predict time and memory. |
| 2 | Do distances stay meaningful in high dimensions? | Measure distance contrast `(d_max − d_min) / d_min` for L1/L2 and several sample sizes as dimensionality grows. |
| 3 | Does feature selection help? | Compare a Random Forest on all features, variance-filtered and correlation-filtered subsets, and size-matched random subsets. |

## Project structure

```
.
├── app.py                     # Streamlit dashboard (UI, Plotly charts, CSV exports)
├── analysis.py                # Reusable analysis functions + CLI
├── requirements.txt
├── Tutorial1.ipynb            # Original notebook (unchanged)
├── Tutorial1_fixed.ipynb      # Corrected notebook (unchanged)
└── README.md
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Already using a system Python? Just install the requirements:

```powershell
python -m pip install -r requirements.txt
```

## Run the dashboard

```powershell
streamlit run app.py
```

Then open <http://localhost:8501>. Use the sidebar to change the random seed
and test split, and each tab has its own controls:

- **Task 1** — choose noise-column counts, `k`, and timing repeats.
- **Task 2** — choose dimensions, sample sizes, metrics, and how many query
  points to average.
- **Task 3** — sweep variance thresholds, set the correlation cut-off, choose
  the number of random trials and trees.

Every task exports its raw results as CSV. Results are cached, so changing a
tab or re-running with identical settings is instant; the sidebar button clears
the cache.

## Run the analysis from the terminal

`analysis.py` doubles as a script that prints the same tables as the notebooks:

```powershell
python analysis.py                        # all three tasks
python analysis.py --tasks 1 3            # selected tasks
python analysis.py --max-dimension 1000   # faster Task 2
python analysis.py --random-trials 5 --seed 7
```

## Improvements over the notebooks

1. **Correct metadata** — `N` = samples (`X.shape[0]`), `D` = features
   (`X.shape[1]`). The original printed `len(X)` and `len(y)` as both.
2. **Reproducibility** — every random step (split, noise, query points, random
   subsets) uses an explicit seed.
3. **Fair timing** — KNN warm-up is discarded, fit and predict are timed
   separately, and results are averaged over repeats.
4. **Consistent units** — Task 1 reports MB (train + test); Task 3 reports KB,
   with labels that match the numbers.
5. **Stable distance contrast** — averaged over multiple query points with a
   standard deviation, instead of a single query.
6. **Fair feature selection** — several variance thresholds are swept, the
   correlation cut-off is configurable, and random baselines are matched to the
   reduced feature count.
7. **More metrics** — macro-F1 alongside accuracy, plus speed-up and memory
   saved against the baseline.
8. **Interactive and exportable** — all parameters adjustable at runtime, all
   result tables downloadable as CSV.

## Notes

- Task 2 with `D = 10,000` and `N = 5,000` allocates a large temporary matrix
  (about 400 MB for the points). Lower the sample size or maximum dimension on
  low-memory machines.
- Timings depend on machine load; accuracy and memory are the stable signals.
- The notebooks are kept for reference and are not modified by the dashboard.
