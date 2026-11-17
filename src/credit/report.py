"""Render README.md (from README.template.md) and docs/index.html from results/metrics.json,
so every number in the write-up comes from the last run."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from credit.config import ROOT, resolve

MODEL_LABELS = {"baseline": "Baseline: everyone gets the average default rate",
                "scorecard": "WoE logistic scorecard", "logistic": "Logistic regression (raw features)",
                "xgboost": "XGBoost (tuned)", "xgboost_isotonic": "XGBoost + isotonic recalibration"}
SEGMENT_LABELS = {"age_band": "Age", "education_label": "Education", "limit_band": "Credit limit"}
FIGURES = [
    ("roc_pr", "ROC and precision-recall curves for each model"),
    ("calibration", "Predicted against observed default rate by decile"),
    ("threshold_sweep", "Cost per 1,000 accounts across PD cut-offs for three LGD assumptions"),
    ("segments", "ROC-AUC and approval rate by age, education and credit limit"),
    ("drivers", "Scorecard information value and XGBoost gain importance"),
    ("score_distribution", "Scorecard score distribution for defaulters and non-defaulters"),
]


def pct(x: float, digits: int = 1) -> str:
    """0.1234 -> '12.3%'."""
    return f"{100 * x:.{digits}f}%"


def ci(pair: list[float], digits: int = 3) -> str:
    """[0.1, 0.2] -> '0.100–0.200'."""
    return f"{pair[0]:.{digits}f}–{pair[1]:.{digits}f}"


def nt(x: float) -> str:
    """NT$ amount, in thousands when large."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}NT${x / 1000:,.0f}k" if x >= 10_000 else f"{sign}NT${x:,.0f}"


def model_rows(m: dict[str, Any]) -> list[list[str]]:
    rows = []
    for key, label in MODEL_LABELS.items():
        r = m["models"][key]
        rows.append([label, f"{r['roc_auc']:.3f} ({ci(r['roc_auc_ci'])})", f"{r['pr_auc']:.3f} ({ci(r['pr_auc_ci'])})",
                     f"{r['ks']:.3f}", f"{r['brier']:.4f}", f"{r['ece']:.4f}", f"{r['fold_auc_sd']:.3f}"])
    return rows


def sensitivity_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[pct(r["lgd"], 0), f"{r['threshold']:.2f}", f"{r['analytic']:.2f}", pct(r["approval_rate"]),
             nt(r["saving_per_1000"]), pct(r["saving_share"]), nt(r["regret_of_base_threshold"])]
            for r in m["decision"]["sensitivity"]]


def segment_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[SEGMENT_LABELS[r["segment"]], r["level"], f"{r['n']:,}", pct(r["default_rate"]), pct(r["mean_pd"]),
             f"{r['roc_auc']:.3f} ({ci(r['roc_auc_ci'])})", pct(r["approval_rate"])] for r in m["segments"]]


def fairness_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[r["group"], f"{r['n']:,}", pct(r["default_rate"]), pct(r["mean_pd"]), pct(r["approval_rate"]),
             pct(r["rejected_good_rate"]), pct(r["approved_bad_rate"])] for r in m["fairness"]]


def cleaning_rows(m: dict[str, Any]) -> list[list[str]]:
    return [[s["step"], f"{s['rows_before']:,}", f"{s['changed']:,}", f"{s['dropped']:,}", f"{s['rows_after']:,}"]
            for s in m["data"]["cleaning"]]


def points_rows(m: dict[str, Any]) -> list[list[str]]:
    """The first few features of the points table (full table in results/scorecard_points.csv)."""
    path = resolve(m.get("_results", "results")) / "scorecard_points.csv"
    if not path.exists():
        return []
    pts = pd.read_csv(path)
    keep = [r["feature"] for r in m["scorecard"]["iv"] if r["selected"]][:3]
    pts = pts[pts["feature"].isin(keep)]
    return [[r.feature, r.bin, f"{r.accounts:,}", pct(r.default_rate), f"{r.woe:+.2f}", str(r.points)]
            for r in pts.itertuples()]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """Rows of strings -> a GitHub markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def html_table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


TABLES = {
    "MODEL_TABLE": (["Model", "ROC-AUC (95% CI)", "PR-AUC (95% CI)", "KS", "Brier", "Calib. error", "Fold SD"],
                    model_rows),
    "SENSITIVITY_TABLE": (["LGD", "Best cut-off", "Theory", "Approved", "Saving / 1,000 accts", "Saving %",
                           "Cost of keeping the 60% cut-off"], sensitivity_rows),
    "SEGMENT_TABLE": (["Segment", "Level", "Accounts", "Default rate", "Mean PD", "ROC-AUC (95% CI)", "Approved"],
                      segment_rows),
    "FAIRNESS_TABLE": (["Group", "Accounts", "Default rate", "Mean PD", "Approved", "Good accounts declined",
                        "Defaulters approved"], fairness_rows),
    "CLEANING_TABLE": (["Step", "Rows in", "Values changed", "Rows dropped", "Rows out"], cleaning_rows),
    "POINTS_TABLE": (["Feature", "Bin", "Accounts", "Default rate", "WoE", "Points"], points_rows),
}


def values(m: dict[str, Any]) -> dict[str, str]:
    """Scalar placeholders used by README.template.md and the HTML page."""
    h, d, mod, cmp = m["headline"], m["decision"], m["models"], m["comparisons"]
    sens = {r["lgd"]: r for r in d["sensitivity"]}
    fair = {r["group"]: r for r in m["fairness"]}
    seg = {(r["segment"], r["level"]): r for r in m["segments"]}
    limits = [r for r in m["segments"] if r["segment"] == "limit_band"]
    iv = m["scorecard"]["iv"]
    dropped = [r["feature"] for r in iv if not r["selected"]]
    diff = d["xgboost_minus_scorecard_cost"]
    return {
        "ROWS": f"{m['data']['rows']:,}", "MODEL_ROWS": f"{m['data']['modelling_rows']:,}",
        "DEFAULT_RATE": pct(m["data"]["default_rate"]),
        "EDU_ODD": f"{m['data']['undocumented_codes']['education']:,}",
        "MAR_ODD": f"{m['data']['undocumented_codes']['marriage']:,}",
        "NEG_BILLS": f"{m['data']['negative_bill_cells']:,}",
        "DUPES": f"{m['data']['duplicate_rows_ignoring_id']:,}",
        "LIMIT_MIN": f"{m['data']['limit_range'][0]:,}", "LIMIT_MAX": f"{m['data']['limit_range'][1]:,}",
        "AGE_MIN": str(m["data"]["age_range"][0]), "AGE_MAX": str(m["data"]["age_range"][1]),
        "FOLDS": str(m["cv"]["folds"]),
        "AUC": f"{h['roc_auc']:.3f}", "AUC_CI": ci(h["roc_auc_ci"]),
        "CARD_AUC": f"{mod['scorecard']['roc_auc']:.3f}", "CARD_AUC_CI": ci(mod["scorecard"]["roc_auc_ci"]),
        "LOGIT_AUC": f"{mod['logistic']['roc_auc']:.3f}",
        "XGB_AUC": f"{mod['xgboost']['roc_auc']:.3f}",
        "AUC_GAP": f"{cmp['xgboost_vs_scorecard_auc']['mean']:.3f}",
        "AUC_GAP_CI": ci(cmp["xgboost_vs_scorecard_auc"]["ci"]),
        "LOGIT_GAP": f"{cmp['xgboost_vs_logistic_auc']['mean']:.3f}",
        "KS": f"{mod['xgboost_isotonic']['ks']:.3f}", "CARD_KS": f"{mod['scorecard']['ks']:.3f}",
        "PR_AUC": f"{mod['xgboost_isotonic']['pr_auc']:.3f}",
        "XGB_ECE": f"{mod['xgboost']['ece']:.4f}", "ISO_ECE": f"{mod['xgboost_isotonic']['ece']:.4f}",
        "CARD_ECE": f"{mod['scorecard']['ece']:.4f}", "LOGIT_ECE": f"{mod['logistic']['ece']:.4f}",
        "BRIER_XGB": f"{mod['xgboost']['brier']:.4f}", "BRIER_ISO": f"{mod['xgboost_isotonic']['brier']:.4f}",
        "BRIER_DIFF_CI": ci(cmp["isotonic_vs_raw_brier"]["ci"], 4),
        "BRIER_BASE": f"{mod['baseline']['brier']:.4f}",
        "LGD": pct(d["lgd"], 0), "MARGIN": pct(d["margin"], 0), "MIN_EXPOSURE": f"{d['min_exposure']:,}",
        "THRESHOLD": f"{h['threshold']:.2f}", "ANALYTIC": f"{d['analytic_threshold']:.3f}",
        "APPROVAL": pct(h["approval_rate"]), "DECLINED": pct(1 - h["approval_rate"]),
        "SAVING": nt(h["saving_per_1000"]), "SAVING_CI": f"{nt(h['saving_ci'][0])}–{nt(h['saving_ci'][1])}",
        "SAVING_SHARE": pct(h["saving_share"], 0),
        "COST_ALL": nt(d["approve_all"]["per_1000"]), "COST_BEST": nt(d["best"]["per_1000"]),
        "CARD_THRESHOLD": f"{d['scorecard_best']['threshold']:.2f}",
        "CARD_APPROVAL": pct(d["scorecard_best"]["approval_rate"]),
        "CARD_SAVING_SHARE": pct(d["scorecard_saving_share"], 0),
        "COST_DIFF": nt(diff["mean"]), "COST_DIFF_CI": f"{nt(diff['ci'][0])} to {nt(diff['ci'][1])}",
        "COST_DIFF_SHARE": pct(diff["share_a_cheaper"], 0),
        "THR_40": f"{sens[0.4]['threshold']:.2f}", "THR_80": f"{sens[0.8]['threshold']:.2f}",
        "APPR_40": pct(sens[0.4]["approval_rate"]), "APPR_80": pct(sens[0.8]["approval_rate"]),
        "REGRET_40": nt(sens[0.4]["regret_of_base_threshold"]), "REGRET_80": nt(sens[0.8]["regret_of_base_threshold"]),
        "F_APPR": pct(fair["female"]["approval_rate"]), "M_APPR": pct(fair["male"]["approval_rate"]),
        "F_DEF": pct(fair["female"]["default_rate"]), "M_DEF": pct(fair["male"]["default_rate"]),
        "F_PD": pct(fair["female"]["mean_pd"]), "M_PD": pct(fair["male"]["mean_pd"]),
        "F_REJ": pct(fair["female"]["rejected_good_rate"]), "M_REJ": pct(fair["male"]["rejected_good_rate"]),
        "F_MISS": pct(fair["female"]["approved_bad_rate"]), "M_MISS": pct(fair["male"]["approved_bad_rate"]),
        "LOW_LIMIT_APPR": pct(limits[0]["approval_rate"]), "HIGH_LIMIT_APPR": pct(limits[-1]["approval_rate"]),
        "LOW_LIMIT_AUC": f"{limits[0]['roc_auc']:.3f}", "LOW_LIMIT_DEF": pct(limits[0]["default_rate"]),
        "OTHER_EDU_AUC": f"{seg[('education_label', 'other')]['roc_auc']:.3f}",
        "OTHER_EDU_N": f"{seg[('education_label', 'other')]['n']:,}",
        "IV_TOP": ", ".join(f"{r['feature']} ({r['iv']:.2f})" for r in iv[:4]),
        "IV_DROPPED": ", ".join(dropped) if dropped else "none",
        "N_SELECTED": str(len(m["scorecard"]["selected"])), "N_FEATURES": str(len(iv)),
        "POINTS_ROWS": str(m["scorecard"]["points_rows"]),
        "SCORE_LO": f"{m['scorecard']['score_range'][0]:.0f}", "SCORE_HI": f"{m['scorecard']['score_range'][1]:.0f}",
        "SCORE_DEF": f"{m['scorecard']['score_median_default']:.0f}",
        "SCORE_GOOD": f"{m['scorecard']['score_median_good']:.0f}",
        "GAIN_TOP": ", ".join(f"{g['feature']} ({pct(g['share'], 0)})" for g in m["importance"]["xgboost_gain"][:3]),
        "GAIN_TOP3_SHARE": pct(sum(g["share"] for g in m["importance"]["xgboost_gain"][:3]), 0),
        "RUNTIME": f"{m['runtime_s'] / 60:.0f}",
    }


def fill(template: str, m: dict[str, Any], table_fn: Any) -> str:
    """Replace {{KEY}} placeholders; tables are rendered with `table_fn(header, rows)`."""
    vals = values(m)
    for key, (header, rows_fn) in TABLES.items():
        vals[key] = table_fn(header, rows_fn(m))
    return re.sub(r"\{\{(\w+)\}\}", lambda mt: vals[mt.group(1)], template)


def _img(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"></figure>'


def render_html(m: dict[str, Any], results: Path) -> str:
    """One self-contained page: TL;DR, figures, tables, method, limitations."""
    v = values(m)
    t = {k: html_table(h, fn(m)) for k, (h, fn) in TABLES.items()}
    f = {name: _img(results / f"{name}.png", alt) for name, alt in FIGURES}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit Default Scorecard</title>
<style>
:root {{ --fg:#1a202c; --muted:#4a5568; --bg:#ffffff; --line:#e2e8f0; --accent:#2b6cb0; --card:#f7fafc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e2e8f0; --muted:#a0aec0; --bg:#171923;
  --line:#2d3748; --accent:#63b3ed; --card:#1a202c; }} }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.6 system-ui,-apple-system,sans-serif;
  max-width:960px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ margin-bottom:4px; }} h2 {{ margin-top:40px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
.sub {{ color:var(--muted); margin-top:0; }}
.tldr {{ background:var(--card); border-left:4px solid var(--accent); padding:12px 16px; border-radius:4px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px; }}
.kpi b {{ display:block; font-size:24px; color:var(--accent); }}
.kpi span {{ color:var(--muted); font-size:14px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:12px 0; display:block; overflow-x:auto; }}
th,td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; white-space:nowrap; }}
figure {{ margin:16px 0; }} img {{ max-width:100%; height:auto; background:#fff; border-radius:4px; }}
</style></head><body>
<h1>Credit Card Default Risk Scorecard</h1>
<p class="sub">Which credit card accounts should keep their line open next year, and how much does the
model behind that decision matter?</p>
<div class="tldr"><b>TL;DR.</b> On {v['MODEL_ROWS']} accounts, tuned XGBoost ranks next-month defaulters with
ROC-AUC <b>{v['AUC']}</b> (95% CI {v['AUC_CI']}), {v['AUC_GAP']} above a points-based WoE scorecard. Declining
accounts whose calibrated default probability is {v['THRESHOLD']} or more ({v['DECLINED']} of accounts) cuts
credit loss plus lost margin by <b>{v['SAVING_SHARE']}</b> against keeping every line open:
{v['SAVING']} per 1,000 accounts (95% CI {v['SAVING_CI']}). At their own best cut-offs the scorecard and
XGBoost cost the same within noise ({v['COST_DIFF']} per 1,000 accounts, CI {v['COST_DIFF_CI']}), so the
readable scorecard is the one to deploy.</div>
<div class="kpis">
<div class="kpi"><b>{v['AUC']}</b><span>ROC-AUC, XGBoost (scorecard {v['CARD_AUC']})</span></div>
<div class="kpi"><b>{v['SAVING_SHARE']}</b><span>lower expected cost than approving all</span></div>
<div class="kpi"><b>{v['THRESHOLD']}</b><span>PD cut-off at LGD {v['LGD']}</span></div>
<div class="kpi"><b>{v['ISO_ECE']}</b><span>calibration error after isotonic</span></div>
</div>
<h2>Model comparison</h2>
<p>Stratified {v['FOLDS']}-fold cross-validation; every number is out-of-fold. 95% CIs from 1,000 paired
bootstrap resamples of accounts.</p>
{t['MODEL_TABLE']}
{f['roc_pr']}
<h2>Calibration</h2>
<p>XGBoost trained on log-loss is already close to calibrated (expected calibration error {v['XGB_ECE']});
isotonic recalibration fitted on inner folds leaves it at {v['ISO_ECE']} and moves the Brier score by an amount
whose CI ({v['BRIER_DIFF_CI']}) includes zero. The scorecard is slightly worse calibrated ({v['CARD_ECE']}).</p>
{f['calibration']}
<h2>Choosing the cut-off</h2>
<p>Keeping a defaulter's line open loses LGD × balance; closing a good account's line loses
{v['MARGIN']} × balance in margin. The cost-minimising cut-off on calibrated PDs is {v['THRESHOLD']}
(theory: margin / (LGD + margin) = {v['ANALYTIC']}). Expected cost falls from {v['COST_ALL']} to
{v['COST_BEST']} per 1,000 accounts.</p>
{t['SENSITIVITY_TABLE']}
{f['threshold_sweep']}
<h2>Segments</h2>
{t['SEGMENT_TABLE']}
{f['segments']}
<h2>Fairness check by sex</h2>
<p>Sex and marital status are never model inputs. Men in this sample default more often ({v['M_DEF']} vs
{v['F_DEF']}) and the model's mean PD tracks that ({v['M_PD']} vs {v['F_PD']}), so men are approved less
often ({v['M_APPR']} vs {v['F_APPR']}). Among accounts that did not default, {v['M_REJ']} of men and
{v['F_REJ']} of women would have been declined.</p>
{t['FAIRNESS_TABLE']}
<h2>What drives it</h2>
<p>Highest information value: {v['IV_TOP']}. XGBoost puts {v['GAIN_TOP3_SHARE']} of its gain on three
features: {v['GAIN_TOP']}. Dropped from the scorecard for IV below 0.02: {v['IV_DROPPED']}.</p>
{f['drivers']}
{t['POINTS_TABLE']}
{f['score_distribution']}
<h2>Data</h2>
<p>{v['ROWS']} credit card accounts from a Taiwanese bank (April to September 2005), UCI Machine Learning
Repository, CC BY 4.0. Default rate {v['DEFAULT_RATE']}.</p>
{t['CLEANING_TABLE']}
<h2>Limitations</h2>
<ul>
<li>The label is a missed payment next month, not a charge-off; LGD and margin are assumptions, so the
NT$ figures show the shape of the trade-off rather than a bank's real P&amp;L.</li>
<li>One bank, six months in 2005, during a local card-debt crisis: the default rate is far above normal.</li>
<li>Cross-validation is random over accounts; there is no later time period to test drift.</li>
<li>The fairness check covers one attribute and outcome rates only.</li>
</ul>
</body></html>
"""


def render_all(cfg: dict[str, Any]) -> None:
    """Write README.md and docs/index.html."""
    results = resolve(cfg["paths"]["results"])
    m = json.loads((results / "metrics.json").read_text())
    m["_results"] = cfg["paths"]["results"]
    template = (ROOT / "README.template.md").read_text()
    (ROOT / "README.md").write_text(fill(template, m, md_table))
    docs = resolve(cfg["paths"]["docs"])
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(render_html(m, results))
    print("rendered README.md and docs/index.html")
