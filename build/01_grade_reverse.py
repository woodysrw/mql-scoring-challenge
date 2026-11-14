"""Q1: what explains HubSpot's letter (fit) and number (engagement)?"""
import sys; sys.path.insert(0, "build")
import pandas as pd, numpy as np
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier, export_text
import clean

d = clean.load(sys.argv[1])

def cramers_v(a, b):
    t = pd.crosstab(a, b); chi2 = chi2_contingency(t)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))

cands = ["emp_band", "industry", "company_geo", "lead_geo", "free_email", "has_company", "seniority", "function",
         "target", "use_case", "vol_picklist", "note_intent", "source", "conversion_type", "form_page",
         "how_heard_c", "conference_c", "mobile", "half"]
for tgt in ["grade_letter", "grade_number"]:
    print(f"\n===== What tracks {tgt} (Cramér's V, 0-1) =====")
    r = sorted(((cramers_v(d[c], d[tgt]), c) for c in cands), reverse=True)
    for v, c in r: print(f"  {v:.2f}  {c}")

print("\n===== % A / B / C by employee band =====")
print((pd.crosstab(d.emp_band, d.grade_letter, normalize="index") * 100).round(0))
print("\n===== letter by free email =====")
print((pd.crosstab(d.free_email, d.grade_letter, normalize="index") * 100).round(0))
print("\n===== letter by target =====")
print((pd.crosstab(d.target, d.grade_letter, normalize="index") * 100).round(0))
print("\n===== letter by seniority =====")
print((pd.crosstab(d.seniority, d.grade_letter, normalize="index") * 100).round(0))
print("\n===== number by conversion_type =====")
print((pd.crosstab(d.conversion_type, d.grade_number, normalize="index") * 100).round(0).assign(n=d.conversion_type.value_counts()))

for tgt, feats in [("grade_letter", ["emp_band", "free_email", "target", "seniority", "industry", "company_geo", "vol_picklist", "use_case", "has_company"]),
                   ("grade_number", ["conversion_type", "form_page", "mobile", "source", "conference_c", "note_intent", "how_heard_c"])]:
    X = pd.get_dummies(d[feats])
    for depth in [3, 5, 8]:
        m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=15, random_state=0).fit(X, d[tgt])
        print(f"\n{tgt} tree depth {depth}: accuracy {m.score(X, d[tgt]):.0%} (majority class {d[tgt].value_counts(normalize=True).iloc[0]:.0%})")
    m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, random_state=0).fit(X, d[tgt])
    print(export_text(m, feature_names=list(X.columns)))
    imp = pd.Series(DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=0).fit(X, d[tgt]).feature_importances_, X.columns)
    print(imp.sort_values(ascending=False).head(8).round(2).to_string())

print("\n===== Hit rate by HubSpot grade =====")
g = d.groupby("hubspot_grade").agg(n=("y", "size"), hit_10k=("y", "mean"), ordered=("y_order", "mean"))
print(g.assign(hit_10k=lambda x: (x.hit_10k * 100).round(1), ordered=lambda x: (x.ordered * 100).round(1)).to_string())
print("\nby letter:"); print(d.groupby("grade_letter").y.agg(["size", "mean"]).round(3))
print("by number:"); print(d.groupby("grade_number").y.agg(["size", "mean"]).round(3))
m = d.is_mql_today == "MQL"
print(f"\nMQL today: n={m.sum()}, precision={d.y[m].mean():.1%}, recall={d.y[m].sum() / d.y.sum():.1%}; non-MQL hit rate={d.y[~m].mean():.1%}; base={d.y.mean():.1%}")
print("\nDrift by half-year:")
print(d.groupby("half").agg(n=("y", "size"), base=("y", "mean"), mql_share=("is_mql_today", lambda s: (s == "MQL").mean()),
                            mql_prec=("y", lambda s: s[d.loc[s.index, "is_mql_today"] == "MQL"].mean())).round(3).to_string())
