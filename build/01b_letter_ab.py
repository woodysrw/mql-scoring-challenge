"""Q1b: among leads with a known company, what separates fit A from B (and B from C)?"""
import sys; sys.path.insert(0, "build")
import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import clean
d = clean.load(sys.argv[1])
k = d[d.emp_band != "unknown"]
print("known-employee rows:", len(k), k.grade_letter.value_counts().to_dict())
for c in ["seniority", "target", "industry", "company_geo", "vol_picklist", "use_case", "emp_band", "free_email"]:
    t = pd.crosstab(k[c], k.grade_letter); t["n"] = t.sum(1); t["%A"] = (t.get("A", 0) / t.n * 100).round(0)
    print(f"\n-- {c}\n", t.sort_values("%A", ascending=False).to_string())
X = pd.get_dummies(k[["seniority", "target", "industry", "company_geo", "vol_picklist", "use_case", "emp_band"]])
m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0).fit(X, k.grade_letter)
print("\nA/B/C tree on known rows, depth 4 acc:", round(m.score(X, k.grade_letter), 3), "majority:", round(k.grade_letter.value_counts(normalize=True).iloc[0], 3))
print(export_text(m, feature_names=list(X.columns)))
# C with known employees: why?
print("\nC rows with known employees:\n", k[k.grade_letter == "C"][["emp_band", "seniority", "free_email", "industry", "company_geo", "target"]].value_counts().head(10))
