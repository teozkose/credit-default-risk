# Credit Card Default Risk Scorecard

[![CI](https://github.com/teozkose/credit-default-risk/actions/workflows/ci.yml/badge.svg)](https://github.com/teozkose/credit-default-risk/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## TL;DR

- **Question:** which credit card accounts should keep their credit line open next year, and does a black-box model earn its place over a points-based scorecard?
- **Headline:** on {{MODEL_ROWS}} accounts, tuned XGBoost ranks next-month defaulters with ROC-AUC **{{AUC}}** (95% CI {{AUC_CI}}), {{AUC_GAP}} above a WoE scorecard. Declining accounts with a calibrated default probability of {{THRESHOLD}} or more cuts credit loss plus lost margin by **{{SAVING_SHARE}}** against keeping every line open ({{SAVING}} per 1,000 accounts, 95% CI {{SAVING_CI}}).
- **Decision:** deploy the scorecard. At their own best cut-offs the two models cost the same within noise ({{COST_DIFF}} per 1,000 accounts, 95% CI {{COST_DIFF_CI}}), and the scorecard is a {{POINTS_ROWS}}-row points table that a credit officer can read and explain to a customer.

## Problem and decision

A card issuer reviews its book before the next cycle: keep each account's line open, or freeze it and collect the balance. Keeping a line open for an account that defaults loses part of the balance; closing a good account gives up the interest and fees it would have paid. The risk team needs a probability of default (PD) it can trust, a cut-off that balances the two costs, and evidence that the rule treats customer groups consistently. The deciding metrics are ROC-AUC and KS for ranking, Brier score and calibration error for the probabilities, and expected cost per 1,000 accounts for the cut-off.

## Data

- **Source:** [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), UCI Machine Learning Repository (Yeh and Lien, 2009), CC BY 4.0. One 5.5 MB zip holding an `.xls` file, downloaded by `make data` and never committed.
- **Size and range:** {{ROWS}} accounts of a Taiwanese bank, six monthly statements from April to September 2005, target = missed payment in October 2005 (default rate {{DEFAULT_RATE}}). Credit limits NT${{LIMIT_MIN}} to NT${{LIMIT_MAX}}, ages {{AGE_MIN}} to {{AGE_MAX}}.
- **Fields:** credit limit, sex, education, marital status, age; for each of six months the repayment status (-2 no balance, -1 paid in full, 0 revolving, 1-9 months late), the bill amount and the amount paid. The source names the latest status `PAY_0` but the matching bill `BILL_AMT1`; the code renames it to `pay_status_1` so every month shares one index.
- **Validation** (`src/credit/validate.py`): required columns, no nulls, unique ids, allowed codes for sex, education, marriage and target, repayment status in [-2, 9], non-negative payments, plausible limits and ages. Any failure stops the run with the full list. Values the documentation does not mention are counted, not rejected: {{EDU_ODD}} education codes 0, 5 or 6, {{MAR_ODD}} marital status 0, and {{NEG_BILLS}} negative monthly bills (credit balances, kept as they are).
- **Cleaning** (counts from the last run):

{{CLEANING_TABLE}}

## Approach

```mermaid
flowchart LR
    A[download UCI zip] --> B[validate schema, codes, ranges, nulls, duplicates]
    B --> C[clean: fold undocumented codes, drop duplicates]
    C --> D[features: utilisation, payment ratios, delinquency streaks, bill trend]
    D --> E{stratified 5-fold CV}
    E --> F[baseline, WoE scorecard, logistic, XGBoost + inner random search, isotonic]
    F --> G[out-of-fold PDs]
    G --> H[ROC-AUC, PR-AUC, KS, Brier with paired bootstrap CIs]
    G --> I[calibration curves]
    G --> J[cost-based cut-off + LGD sensitivity]
    G --> K[segments + fairness by sex]
    H --> L[README + report page]
    I --> L
    J --> L
    K --> L
```

- **Features** (`src/credit/features.py`): balance-to-limit utilisation (latest, mean, max over six months), share of last month's bill paid (latest, mean, worst of three), months late, worst lateness, current late streak, months revolving, months with an unpaid bill and no payment, bill trend (least-squares slope over six months, scaled by the limit) and three-month balance change. Sex and marital status are never model inputs; sex is kept only for the fairness check.
- **Validation:** stratified {{FOLDS}}-fold cross-validation over accounts; every reported number is out-of-fold. XGBoost hyperparameters come from an 8-draw random search scored by 3-fold inner CV on each training fold, and the isotonic map is fitted on inner folds too, so no prediction ever comes from a model that saw that account.
- **Models:** a baseline that gives everyone the training default rate (the probabilistic version of "always predict no default"); a weight-of-evidence scorecard (quantile bins, small bins merged, features with information value below 0.02 dropped, logistic regression on the WoE values, scaled to 600 points at 50:1 odds with 20 points to double the odds); L2 logistic regression on standardised raw features; tuned XGBoost; and XGBoost with isotonic recalibration.
- **Uncertainty:** 1,000 bootstrap resamples of accounts, shared across models so differences are paired. The resamples are applied as integer weights to scores sorted once, which makes 5,000 metric evaluations take seconds.
- **Decision rule:** keep the line open when PD is below a cut-off. Cost = LGD × balance for every approved defaulter + margin × balance for every declined good account (LGD {{LGD}}, margin {{MARGIN}} of the balance over 12 months, balance floored at NT${{MIN_EXPOSURE}}). The cut-off is swept from 0.02 to 0.90.
- **Sensitivity check:** the cut-off is recomputed at LGD 40%, 60% and 80%.

## Results

Out-of-fold over {{MODEL_ROWS}} accounts ("Calib. error" is the count-weighted gap between predicted and observed default rate over PD deciles; "Fold SD" is the standard deviation of ROC-AUC across the five folds):

{{MODEL_TABLE}}

XGBoost beats the scorecard by {{AUC_GAP}} ROC-AUC (paired 95% CI {{AUC_GAP_CI}}) and plain logistic regression by {{LOGIT_GAP}}. Both scorecard and logistic regression are linear in their inputs; the scorecard's edge comes from its WoE bins, which let one coefficient per feature fit a non-linear step such as the one from "paid in full" to "one month late", where the raw status code gets a single straight-line slope.

![ROC and precision-recall curves](results/roc_pr.png)

**Calibration.** XGBoost trained on log-loss is already close to calibrated (calibration error {{XGB_ECE}}). Isotonic recalibration leaves it at {{ISO_ECE}}, and the Brier score changes from {{BRIER_XGB}} to {{BRIER_ISO}}, a difference whose 95% CI ({{BRIER_DIFF_CI}}) includes zero. The recalibrated model is still the one used for the cut-off, because the cut-off rule assumes calibrated PDs and isotonic is the safeguard if retraining drifts. The scorecard's error is {{CARD_ECE}}.

![Calibration curves](results/calibration.png)

**The cut-off.** Approving everyone costs {{COST_ALL}} per 1,000 accounts. Declining accounts with PD ≥ {{THRESHOLD}} keeps {{APPROVAL}} of lines open and brings the cost to {{COST_BEST}}, a {{SAVING_SHARE}} cut. Theory says that with calibrated PDs the best cut-off is margin / (LGD + margin) = {{ANALYTIC}}, whatever the balance; the empirical sweep lands next to it. The scorecard, with its own best cut-off of {{CARD_THRESHOLD}} ({{CARD_APPROVAL}} approved), saves {{CARD_SAVING_SHARE}}: XGBoost is cheaper in only {{COST_DIFF_SHARE}} of bootstrap resamples. Better ranking overall does not turn into a cheaper decision at this one cut-off.

![Cost across cut-offs](results/threshold_sweep.png)

**Sensitivity to LGD.** The cut-off moves from {{THR_40}} at 40% LGD ({{APPR_40}} approved) to {{THR_80}} at 80% ({{APPR_80}} approved). Keeping the 60% cut-off when the true LGD is 40% or 80% costs only {{REGRET_40}} or {{REGRET_80}} extra per 1,000 accounts, because the cost curve is flat near its minimum:

{{SENSITIVITY_TABLE}}

## What drives it / error analysis

- **Payment history dominates.** Highest information value: {{IV_TOP}}. XGBoost puts {{GAIN_TOP3_SHARE}} of its total gain on {{GAIN_TOP}}. Utilisation and payment ratios add a little on top; the scorecard keeps {{N_SELECTED}} of {{N_FEATURES}} candidate features and drops {{IV_DROPPED}} for IV below 0.02.
- **Segments.** Ranking quality is stable across age and education bands. It is weakest for the smallest limits (ROC-AUC {{LOW_LIMIT_AUC}} at NT$0-50k), which also have the highest default rate ({{LOW_LIMIT_DEF}}), so only {{LOW_LIMIT_APPR}} of those accounts are approved against {{HIGH_LIMIT_APPR}} above NT$240k. The "other" education group ({{OTHER_EDU_N}} accounts, mostly the undocumented codes) has ROC-AUC {{OTHER_EDU_AUC}} with a wide CI: too few defaulters to judge.

{{SEGMENT_TABLE}}

![Segments](results/segments.png)

- **Fairness check by sex.** Sex is not a model input, but other features can stand in for it, so outcomes are compared after the fact. Men in this sample default more often ({{M_DEF}} vs {{F_DEF}}), and the model's mean PD tracks that closely ({{M_PD}} vs {{F_PD}}), so it is calibrated within each group. As a result men are approved less often ({{M_APPR}} vs {{F_APPR}}). Among accounts that went on to pay, {{M_REJ}} of men and {{F_REJ}} of women would have been declined; among accounts that defaulted, {{M_MISS}} of men and {{F_MISS}} of women would have been approved. In plain terms: the rule does not penalise either group beyond its observed repayment record, but a good-paying man is slightly more likely to be turned down than a good-paying woman. A lender would review that gap against its fair-lending policy before deployment.

{{FAIRNESS_TABLE}}

- **The scorecard itself.** Scores run from about {{SCORE_LO}} to {{SCORE_HI}} (1st to 99th percentile); the median account that defaulted scores {{SCORE_DEF}} against {{SCORE_GOOD}} for one that paid. The first three features of the points table (full table in `results/scorecard_points.csv`):

{{POINTS_TABLE}}

![Score distribution](results/score_distribution.png)

![Drivers](results/drivers.png)

## Limitations and next steps

- **The label is a missed payment, not a loss.** LGD and margin are assumptions applied to the latest balance, so the NT$ figures show the shape of the trade-off, not a bank's P&L. With real recovery data, LGD would be modelled per account.
- **One bank, six months, one unusual year.** The data come from 2005, during Taiwan's card-debt crisis, and the {{DEFAULT_RATE}} default rate is far above normal. The cut-off would need refitting on current data.
- **No out-of-time test.** All accounts share the same observation window, so cross-validation is random over accounts. A later cohort is the right test for drift.
- **Fairness check is narrow.** One attribute and outcome rates only. Next: adverse-action reasons from the points table, and a check of which features stand in for sex.
- **Next steps:** monotonic WoE bins (now bins only need a minimum size), a reject-inference note, and a challenger test that runs the scorecard and XGBoost side by side on new accounts.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
make all     # data -> features -> run -> report -> notebook (about {{RUNTIME}} min on a laptop)
make test    # 8+ tests on synthetic accounts, no download
make lint
```

`run_all.sh` runs the same steps. Every number above is filled in from `results/metrics.json` by `src/credit/report.py`, which also writes the report page in `docs/index.html`. `xlrd` is the one library outside the usual stack: the source file is a legacy `.xls`.

## Project structure

```
credit-default-risk/
├── config.yaml              # paths, cleaning rules, model and decision parameters
├── Makefile / run_all.sh    # make data | run | report | all
├── data/download.py         # fetches the UCI zip (data files are not committed)
├── src/credit/
│   ├── data.py              # download, read the .xls, tidy column names
│   ├── validate.py          # schema, codes, ranges, nulls, duplicates
│   ├── features.py          # cleaning log + account-level features
│   ├── woe.py               # WoE binning, information value, points table
│   ├── models.py            # baseline, scorecard, logistic, XGBoost search, isotonic
│   ├── evaluate.py          # weighted-bootstrap metrics, calibration, segments
│   ├── decision.py          # cost-based cut-off, sensitivity, fairness table
│   ├── figures.py           # all charts
│   ├── report.py            # README + docs/index.html from metrics.json
│   └── pipeline.py          # stages and the cross-validation loop
├── tests/                   # synthetic-data tests, incl. an end-to-end run
├── notebooks/01_analysis.ipynb
├── results/                 # metrics.json, scorecard_points.csv, figures
└── docs/index.html          # report page
```

Licence: MIT for the code. Data: CC BY 4.0, Yeh, I. C., and Lien, C. H. (2009), *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*, Expert Systems with Applications 36(2).
