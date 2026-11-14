"""Q2+Q3: which fields move reached_10k_in_12m? Univariate lift + multivariate logistic."""
import sys; sys.path.insert(0, "build")
import pandas as pd, numpy as np
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
import clean
d = clean.load(sys.argv[1])
BASE = d.y.mean()
HEARD = {"Word of mouth": "word of mouth / customer / got a reward", "Current or former customer": "word of mouth / customer / got a reward",
         "Received a Tremendous reward": "word of mouth / customer / got a reward", "Search engine": "search engine",
         "Social media": "social / video / review / news", "YouTube or online video": "social / video / review / news",
         "Review website": "social / video / review / news", "News article": "social / video / review / news",
         "Conference or event": "event / partner", "Platform integration or partner vendor": "event / partner", "Other": "other", "unknown": "unknown"}
d["how_heard_pod"] = d.how_heard_c.map(HEARD)
d["emp_band_c"] = d.emp_band
d["usecase_x_vol"] = d.use_case + " | " + d.vol_picklist
# Does stated note volume agree with picklist?
d["vol_best"] = np.where(d.note_vol_band != "none stated", d.note_vol_band, d.vol_picklist)

def cramers_v(a, b):
    t = pd.crosstab(a, b); chi2, p = chi2_contingency(t)[:2]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))), p

feats = ["hubspot_grade", "grade_letter", "grade_number", "emp_band", "industry", "company_geo", "lead_geo", "free_email", "has_company",
         "seniority", "function", "target", "use_case", "vol_picklist", "vol_best", "note_intent", "note_vol_band", "source",
         "conversion_type", "form_page", "how_heard_pod", "conference_c", "mobile", "note_vendor_pain", "note_urgent", "note_finance", "note_intl", "half"]
print(f"Base rate reached_10k: {BASE:.1%} (n={len(d)}); ordered_within_6mo: {d.y_order.mean():.1%}")
print("\n===== Strength of association with reached_10k (Cramér's V; p from chi-square) =====")
rank = sorted(((*cramers_v(d[f], d.y), f) for f in feats), reverse=True)
for v, p, f in rank: print(f"  {v:.2f}  p={p:.1e}  {f}")

def lift(col, min_n=1):
    g = d.groupby(col).agg(n=("y", "size"), hit=("y", "mean"), ordered=("y_order", "mean"))
    g["lift"] = g.hit / BASE
    g["small_n"] = np.where(g.n < 30, "<30", "")
    return g.sort_values("hit", ascending=False).assign(hit=lambda x: (x.hit * 100).round(1), ordered=lambda x: (x.ordered * 100).round(1), lift=lambda x: x.lift.round(2))

for f in ["emp_band", "free_email", "seniority", "function", "target", "use_case", "vol_picklist", "note_intent", "note_vol_band", "vol_best",
          "source", "conversion_type", "how_heard_pod", "how_heard_c", "conference_c", "mobile", "company_geo", "industry", "form_page",
          "note_vendor_pain", "note_urgent", "note_finance", "note_intl"]:
    print(f"\n----- {f} -----"); print(lift(f).to_string())

print("\n----- picklist volume vs note-stated volume (rows with both) -----")
both = d[(d.note_vol_band != "none stated") & (d.vol_picklist != "unknown")]
print(pd.crosstab(both.vol_picklist, both.note_vol_band), "\nagree:", round((both.vol_picklist == both.note_vol_band).mean(), 2), "n=", len(both))
print("\n----- use case x volume (n>=30) -----")
t = lift("usecase_x_vol"); print(t[t.n >= 30].to_string())

print("\n===== Multivariate logistic (all leads) =====")
d["emp_log"] = np.log10(d.employees.fillna(1).clip(lower=1)); d["emp_known"] = d.employees.notna().astype(int)
f = ("y ~ C(vol_best, Treatment('unknown')) + emp_known + emp_log + C(free_email) + C(seniority, Treatment('unknown')) + C(use_case, Treatment('unknown'))"
     " + C(conversion_type_g, Treatment('Signup')) + C(note_intent, Treatment('no note')) + C(source, Treatment('Direct')) + C(company_geo, Treatment('unknown'))")
d["target_true"] = (d.target == "TRUE").astype(int)
d["conversion_type_g"] = d.conversion_type.where(d.conversion_type.map(d.conversion_type.value_counts()) >= 30, "other (small)")
for label, form in [("without target account", f)]:
    m = smf.logit(form, data=d).fit(disp=0)
    print(f"\n-- {label}: pseudo-R2 {m.prsquared:.3f}, n={int(m.nobs)}")
    s = pd.DataFrame({"odds_ratio": np.exp(m.params).round(2), "p": m.pvalues.map(lambda p: f"{p:.3f}")})
    print(s[m.pvalues < 0.1].sort_values("odds_ratio", ascending=False).to_string())

print("\n===== Lever ablation: cross-validated AUC (L2 logistic, 5-fold grouped by company) =====")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
groups_map = {
    "volume (picklist + note)": ["vol_best"], "title (seniority + function)": ["seniority", "function"],
    "company (size, industry, geo)": ["emp_band", "industry", "company_geo", "has_company"], "email type": ["free_email"],
    "use case": ["use_case"], "conversion type / form": ["conversion_type_g", "form_page"], "note intent": ["note_intent"],
    "source + how heard": ["source", "how_heard_pod"], "mobile": ["mobile"], "target account": ["target"],
}
grp = d.company_name.where(d.company_name != "", d.email)
def auc(cols):
    X = pd.get_dummies(d[cols].astype(str))
    p = cross_val_predict(LogisticRegression(C=0.5, max_iter=2000), X, d.y, cv=GroupKFold(5), groups=grp, method="predict_proba")[:, 1]
    return roc_auc_score(d.y, p)
all_no_t = [c for k, v in groups_map.items() if k != "target account" for c in v]
print(f"  HubSpot grade alone:          {auc(['hubspot_grade']):.3f}")
print(f"  all levers, no target acct:   {auc(all_no_t):.3f}")
print(f"  all levers + target acct:     {auc(all_no_t + ['target']):.3f}")
print(f"  target acct alone:            {auc(['target']):.3f}")
print("  drop one group from 'all levers, no target' (bigger drop = more unique signal):")
base_auc = auc(all_no_t)
res = []
for k, v in groups_map.items():
    if k == "target account": continue
    res.append((base_auc - auc([c for c in all_no_t if c not in v]), auc(v), k))
for drop, alone, k in sorted(res, reverse=True):
    print(f"    {k:32s} drop {drop:+.3f}   alone {alone:.3f}")

print("\n===== Preview: matched-volume precision (cross-validated, NOT the time-split eval) =====")
n_mql = (d.is_mql_today == "MQL").sum()
for label, cols in [("all levers, no target", all_no_t), ("all levers + target", all_no_t + ["target"])]:
    X = pd.get_dummies(d[cols].astype(str))
    p = cross_val_predict(LogisticRegression(C=0.5, max_iter=2000), X, d.y, cv=GroupKFold(5), groups=grp, method="predict_proba")[:, 1]
    top = np.argsort(-p)[:n_mql]
    print(f"  {label:24s} top {n_mql}: precision {d.y.iloc[top].mean():.1%}, recall {d.y.iloc[top].sum() / d.y.sum():.1%}")
m = d.is_mql_today == "MQL"
print(f"  {'HubSpot A1/A2/B1':24s} n={m.sum()}: precision {d.y[m].mean():.1%}, recall {d.y[m].sum() / d.y.sum():.1%}")
