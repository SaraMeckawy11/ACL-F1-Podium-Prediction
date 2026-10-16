# Milestone 1 — task board

Deadline: **18 October 2026, 23:59**. Submit via the form: Kaggle notebook link, GitHub repo link, report PDF.

## Checklist → where it is done

| Checklist item | Notebook section | Status |
|---|---|---|
| EDA (before cleaning, with plots) | 1 | done in code |
| Sentinel handling (`\N`), semantic typing | 2.1 | done in code |
| PK uniqueness / FK reconciliation | 2.2 | done in code |
| Grain = one row per (raceId, driverId), duplicate rule documented | 2.3 | done in code |
| EDA after cleaning (impact of cleaning) | 3 | done in code |
| Q1 front-row conversion by circuit, pre/post 2014, grid vs qualifying | 4 / Q1 | done in code |
| Q2 home advantage controlling for grid (Mantel–Haenszel) | 4 / Q2 | done in code |
| Q3 mechanical-retirement rate 2014–21 vs 2022–24 (status mapping documented) | 4 / Q3 | done in code |
| Leakage audit + leakage-safe form features + leakage tests | 5.1–5.3 | done in code |
| Feature evidence (plots, correlations) and selection rule | 5.4–5.5 | done in code |
| Temporal split + justification | 6 | done in code |
| Pre-processing effect on distributions | 7 | done in code |
| ≥ 2 pre-processing-order experiments (3 included) | 7.1 | done in code |
| Baseline + ≥ 2 statistical models + shallow NN, ≥ 3 attempts | 8 | done in code |
| Metrics on train/val/test, primary/secondary metric justified | 8.3 | done in code |
| Feature-group ablation | 9 | done in code |
| Permutation importance + SHAP + LIME/local SHAP, not-causal statement | 10 | done in code |
| Inference function on complete and partially-missing record | 11 | done in code |
| Analytical report (PDF) | 12 → `outputs/report/report.html` | **run on Kaggle, review, add interpretation, print to PDF** |

## Still to do by the team
- [ ] Run the notebook on Kaggle with the real dataset (Run All) and check every output.
- [ ] Read every figure and write your own interpretation into the markdown cells and the report.
- [ ] Check the Q3 status → category mapping (`outputs/tables/q3_status_mapping.csv`) against the real status list.
- [ ] Check the nationality → country mapping printout in Q2 (any unmapped nationality is listed).
- [ ] Commit the executed notebook (with outputs) back to this repo.
- [ ] Make the GitHub repo and the Kaggle notebook accessible to the TAs; submit the form.

## Evaluation prep — anyone can be asked anything
Suggested owners for deep-dive (everyone must still know all of it):

| Area | Owner |
|---|---|
| Cleaning, keys, grain, sentinels, EDA | member 1 |
| Q1–Q3 (joins, definitions, Mantel–Haenszel, status mapping) | member 2 |
| Features, leakage, split, pre-processing experiments | member 3 |
| Models, metrics, thresholds, ablation | member 4 |
| XAI (permutation, SHAP, LIME), inference, limitations | member 5 |
