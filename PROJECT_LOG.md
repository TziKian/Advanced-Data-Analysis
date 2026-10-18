# Project Log — AIT403 Advanced Data Analysis

Record of everything done in this repository: environment setup, Git history,
the conversion of the Tutorial 1 notebooks into a Python module and Streamlit
dashboard, and the verification results.

**Date:** 2026-10-18
**Working directory:** `C:\TziKian\Vscode\Python\Year 4\AIT403 advanced data`
**Remote:** <https://github.com/TziKian/Advanced-Data-Analysis.git>
**Branch:** `main`

---

## 1. Environment setup

| Item | Before | Action | After |
|------|--------|--------|-------|
| Git | Not installed / not on PATH | `winget install --id Git.Git -e` | Git 2.55.0.5 (`C:\Program Files\Git\cmd\git.exe`) |
| GitHub CLI (`gh`) | Not installed | Not needed (existing remote URL used) | — |
| Python | 3.13.14 | — | unchanged |
| pip | 26.1.2 | — | unchanged |
| Streamlit | Not installed | `python -m pip install "streamlit>=1.40"` | 1.66.0 |

### Python packages already present

`numpy 2.3.5`, `pandas 2.3.3`, `matplotlib 3.10.7`, `scikit-learn 1.7.2`,
`scipy 1.16.3`, `plotly 6.7.0`, `seaborn 0.13.2`, `ipykernel 7.1.0`,
`ipywidgets 8.1.8` — nothing extra was required for the notebooks.

> After installing Git, a **new terminal** (or VS Code restart) is required so
> `git` appears on `PATH` in the user's own shell.

---

## 2. Repository setup and history

1. `git init -b main` in the project folder.
2. Added `.gitignore` covering Jupyter checkpoints, `__pycache__`, virtual
   environments, `.env`/secrets (including `.streamlit/secrets.toml`), and
   OS/editor files.
3. Configured identity globally: `TziKian <ytkyeoh@gmail.com>`.
4. Added the remote and pushed:
   `git remote add origin https://github.com/TziKian/Advanced-Data-Analysis.git`

### Commit history

| Commit | Message | Notes |
|--------|---------|-------|
| `6953404` | Initial commit: Tutorial 1 notebooks | `Tutorial1.ipynb`, `Tutorial1_fixed.ipynb`, `.gitignore` |
| `48262da` | Add TziKian's name to Tutorial1_fixed.ipynb | Added `**Name:** TziKian` to the title cell |
| `eb4b49f` | for testing | Pushed outside this session; added 10 lines to `Tutorial1.ipynb` |
| `278ef3f` | Revert repository content to 6953404 | Restored the exact `6953404` snapshot |
| `027d34a` | Add analysis module and interactive Streamlit dashboard for Tutorial 1 | `analysis.py`, `app.py`, `requirements.txt`, `README.md` |

### Revert to `6953404`

- A `git revert 6953404` was found **in progress with conflicts** (`.gitignore`
  staged as deleted, both notebooks `UD`). It was aborted with `git revert --abort`.
- The snapshot was restored with
  `git restore --source=6953404 --staged --worktree .`, verified to have **zero
  diff** against `6953404`, and committed as `278ef3f`.
- History was preserved (no force push), so the result is a normal revert commit
  rather than a rewritten branch.

---

## 3. Files added

```
.
├── app.py                  # Streamlit dashboard
├── analysis.py             # Analysis functions + CLI
├── requirements.txt        # numpy, pandas, scipy, scikit-learn, plotly, streamlit, matplotlib, jupyter
├── README.md               # Setup, usage, methodology
├── PROJECT_LOG.md          # This file
├── .gitignore
├── Tutorial1.ipynb         # Original notebook (unchanged)
└── Tutorial1_fixed.ipynb   # Corrected notebook (unchanged)
```

---

## 4. Notebook → Python conversion

`analysis.py` extracts all three notebook tasks into documented, reusable
functions:

| Function | Task | Purpose |
|----------|------|---------|
| `load_digits_data`, `split_data` | shared | Load digits data, seeded train/test split |
| `add_noise_columns` | 1 | Append uniform noise features in the 0–16 pixel range |
| `benchmark_knn` | 1 | Warm-up + repeated fit/predict timing, accuracy, macro-F1 |
| `run_task1` | 1 | One row per noise setting (accuracy, σ, F1, fit/predict ms, MB) |
| `compute_contrast`, `run_task2` | 2 | Contrast `(d_max − d_min)/d_min` averaged over queries, with σ |
| `evaluate_subset` | 3 | Random Forest accuracy, macro-F1, train time (ms), memory (KB) |
| `run_task3` | 3 | Baseline + variance sweep + correlation filter + matched random baselines + feature importances |
| `main` | — | CLI: `python analysis.py [--tasks 1 2 3] [--max-dimension N] [--seed N]` |

### Improvements applied

1. **Correct metadata** — `N` = samples (`X.shape[0]`), `D` = features
   (`X.shape[1]`); the original printed `len(X)`/`len(y)` as both.
2. **Reproducibility** — seeded splits, noise, query points and random subsets.
3. **Fair timing** — KNN warm-up excluded; `fit` and `predict` timed separately
   and averaged over repeats.
4. **Consistent units** — Task 1 in MB (train + test), Task 3 in KB.
5. **Stable distance contrast** — averaged over multiple query points with a
   standard deviation instead of a single query.
6. **Fair feature selection** — several variance thresholds swept, configurable
   correlation cut-off, random baselines matched to the reduced feature count.
7. **Extra metrics** — macro-F1, speed-up and memory saved vs. baseline.
8. **Exportable results** — every table downloadable as CSV from the dashboard.

---

## 5. Streamlit dashboard (`app.py`)

Run with:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

### Layout

- **Sidebar** — random seed, test split, dataset summary, "Clear results &
  cache" button, method notes expander.
- **Overview tab** — KPI row (1,797 samples / 64 features / 10 classes / pixel
  range 0–16), class distribution, one sample per class, task cards, and the
  list of improvements over the original notebook.
- **Task 1 tab** — noise-column multiselect, `k` slider, timing repeats; KPIs
  (clean accuracy, accuracy at max noise with Δ, feature count, memory growth);
  accuracy line chart, fit-vs-predict log timing chart, memory bar chart, result
  table, CSV export, automatic insight callout.
- **Task 2 tab** — dimensions, sample sizes, L1/L2 metrics, query count;
  KPI row (contrast at min/max D, % reduction); log–log contrast chart with σ
  error bars, pivot table, CSV export, interpretation panel.
- **Task 3 tab** — variance thresholds, correlation cut-off, random trials,
  tree count; KPIs (baseline, best subset with Δ, feature count, matched random
  baseline); accuracy-by-strategy bar chart with baseline line, accuracy vs.
  feature count scatter, top-15 feature importances, results table with Δ
  accuracy / speed-up / memory saved, matched random baselines table, CSV
  exports, methodology expander.

### Implementation details

- `st.cache_data` wrappers for all three tasks; results persist across reruns
  with identical parameters.
- Inputs use `st.form` so expensive computations run only on submit.
- Plotly charts (`plotly_express` + `graph_objects`), dark-theme friendly,
  `width="stretch"` layout.
- Results and parameters stored in `st.session_state` and shown in a caption
  under each task.

### Fixes made during verification

- Digit previews were stretched by a 10-panel heatmap subplot; replaced with
  `st.image` in 10 columns (square, labelled).
- Task 2 contrast chart produced cluttered minor tick labels on log axes; set
  `dtick=1` on both axes.
- Contrast-reduction KPI rounded to `100.0%`; now shows two decimals (`99.98%`).

---

## 6. Verification results (seed 42, test split 0.2)

### Task 1 — KNN with injected noise (k=5, 5 repeats)

| Noise cols | Features | Accuracy (%) | Fit (ms) | Predict (ms) | Memory (MB) |
|-----------:|---------:|-------------:|---------:|-------------:|------------:|
| 0 | 64 | 98.61 | 0.40 | 1.27 | 0.88 |
| 50 | 114 | 95.56 | 0.35 | 1.34 | 1.56 |
| 200 | 264 | 89.17 | 0.57 | 2.31 | 3.62 |
| 1,000 | 1,064 | 63.89 | 1.87 | 7.65 | 14.59 |
| 5,000 | 5,064 | 34.44 | 9.39 | 28.55 | 69.43 |

→ **−64.17 pp** accuracy and **79.1×** memory growth at 5,000 noise features.

### Task 2 — Distance concentration (20 queries)

| D | N=500 L1 | N=5,000 L1 | N=500 L2 | N=5,000 L2 |
|--:|---------:|-----------:|---------:|-----------:|
| 2 | 107.35 | 231.49 | 101.33 | 215.10 |
| 10 | 3.16 | 4.54 | 2.52 | 3.69 |
| 100 | 0.50 | 0.66 | 0.39 | 0.52 |
| 1,000 | 0.13 | 0.17 | 0.11 | 0.13 |
| 10,000 | 0.043 | 0.051 | 0.035 | 0.041 |

→ Contrast at `D=2` is 215.1 versus 0.041 at `D=10,000`; a **99.98%**
reduction. Distances concentrate as dimensionality grows.

### Task 3 — Feature selection (Random Forest, 100 trees)

| Method | Features | Accuracy (%) | Δ (pp) | Train (ms) | Memory (KB) |
|--------|---------:|-------------:|-------:|-----------:|------------:|
| Baseline (all features) | 64 | 96.11 | — | 138.6 | 898.5 |
| Variance threshold = 1 | 48 | 96.94 | +0.83 | 96.6 | 673.9 |
| Variance threshold = 5 | 45 | **97.22** | +1.11 | 104.3 | 631.8 |
| Variance threshold = 10 | 43 | 96.94 | +0.83 | 93.7 | 603.7 |
| Variance threshold = 22 | 29 | 96.11 | 0.00 | 93.4 | 407.1 |
| Correlation filter (\|r\| > 0.9) | 63 | 97.22 | +1.11 | 93.7 | 884.5 |
| Random subset (45 features, 10×) | 45 | 95.94 | −0.17 | 93.4 | 631.8 |

→ Variance threshold 5 keeps **45 of 64** features with the best accuracy,
beating the equally sized random subset by **+1.28 pp**. Top baseline features:
indices 21, 36, 26, 43, 42.

### Dashboard checks

All four tabs were exercised in a browser against the live dev server: no
tracebacks, charts and tables rendered, metrics and insight callouts correct for
every task. The CLI was run for all three tasks as well.

---

## 7. Notes and caveats

- Task 2 with `D=10,000` and `N=5,000` allocates roughly 400 MB of temporary
  memory for the random points; lower the sample size or maximum dimension on
  low-memory machines.
- Timings depend on machine load; accuracy and memory are the stable signals.
- During headless-browser verification, a Streamlit session reconnect once left
  the sidebar test-split slider at `0.10` instead of its `0.20` default. A fresh
  session showed the correct `0.20`, so this is a browser/session artifact, not
  an application bug. Each task's caption displays the parameters actually used.
- The two notebooks remain unchanged and are kept for reference.
