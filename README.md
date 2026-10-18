# ACL Milestone 1 — Formula 1 Podium Prediction

Predict, for every driver in a race, whether they finish in the top 3 (`positionOrder <= 3`) using only
information available **after qualifying and before the race starts**.

Dataset: [Formula 1 World Championship 1950–2024 (Kaggle / Ergast)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

## Repository layout

| path | what |
|---|---|
| `notebooks/f1_podium_prediction.ipynb` | **the deliverable**: one run-all notebook (audit, cleaning, EDA, Q1–Q3, features, models, ablation, XAI, inference, report) |
| `src/f1_podium_prediction.py` | the same notebook as a percent-format script (source of truth, easy to diff) |
| `src/build_notebook.py` | rebuilds the `.ipynb` from the script: `python src/build_notebook.py` |
| `tests/make_fake_f1_data.py` | generates a *fake* dataset with the same schema, only to smoke-test the notebook |
| `data/raw/` | put the 14 Kaggle CSVs here when running locally (they are git-ignored) |
| `docs/TASKS.md` | milestone checklist → notebook section, and the team |
| `outputs/` | created by the notebook: figures, tables, `report/report.html` (git-ignored) |

## Run on Kaggle (submission)
1. *Create → New Notebook → File → Import Notebook* and upload `notebooks/f1_podium_prediction.ipynb`.
2. *Add Input* → search **formula-1-world-championship-1950-2020** (rohanrao) → add.
3. *Run All*. The data folder is detected automatically under `/kaggle/input/`.
4. Download `outputs/report/report.html` from the output panel, open it in a browser, *Print → Save as PDF*.
   The report (cover page, answers, figures and interpretations) is built entirely from the numbers of that run.
   To show a logo on the cover, place `report_logo.png` next to the notebook or set `F1_REPORT_LOGO`.
5. Set the notebook to public (or share it) and submit the link.

Interpretation texts for the experiments, models, ablation and XAI are generated from the computed numbers
(`INTERP` in the notebook), so they stay correct if library versions change the scores slightly. The answers to the
three data-engineering questions are written in markdown cells.

## Run locally
```bash
pip install -r requirements.txt
# put the 14 CSVs in data/raw/   (or: export F1_DATA_DIR=/path/to/csvs)
cd notebooks && F1_DATA_DIR=../data/raw jupyter nbconvert --to notebook --execute --inplace f1_podium_prediction.ipynb
```

## Smoke test without the real data
```bash
python tests/make_fake_f1_data.py --out /tmp/fake_f1
F1_DATA_DIR=/tmp/fake_f1 F1_OUT_DIR=/tmp/fake_out jupyter nbconvert --to notebook --execute \
    notebooks/f1_podium_prediction.ipynb --output /tmp/fake_run.ipynb
```
The fake data is random: its numbers mean nothing and must never be reported.

## Key design decisions (details and evidence in the notebook)
- **Data errors**: a qualifying lap more than 2× the session median is a recording error and is set to missing
  (one case: 1995 Japanese GP, 16:42.640).
- **Grain**: one row per `(raceId, driverId)`; shared-drive duplicates keep the car the driver *started* in
  (pre-race information), with a sensitivity check against the best-finish rule.
- **Scope**: Indianapolis 500 (1950–1960) and non-starters (DNQ/DNPQ/withdrew) are excluded.
- **Leakage**: standings come from the previous round; all form features use `shift(1)`; a brute-force test and a
  deliberately leaky model demonstrate both.
- **Split**: temporal — train ≤ 2019, validation 2020–2021, test 2022–2024; training window chosen on validation.
- **Models**: grid-rule reference, logistic regression (baseline), random forest, histogram gradient boosting,
  shallow MLP; primary metric PR-AUC, secondary F1 at a validation-chosen threshold.
- **Model selection**: a race-level bootstrap compares every model with the logistic-regression baseline on
  validation; the baseline is kept unless another model is reliably better (95% interval above 0). The simpler
  "highest validation PR-AUC" rule is shown next to it for comparison.
- **LIME**: run with default settings and with the 0/1 flags treated as categorical; the categorical setting is
  used because it only creates realistic rows. Both are compared with local SHAP.
- **Statistics**: Q2 uses the Mantel–Haenszel pooled odds ratio (`statsmodels` `StratifiedTable`), Q3 Wilson
  intervals (`statsmodels` `proportion_confint`).
