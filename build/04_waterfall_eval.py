"""Waterfall eval: filter -> score -> route, vs HubSpot, on a time split.
Train: leads before 2024-04-01. Test: 2024-04-01 to 2024-09-30 (last 6 months).
Companies that appear in test are dropped from train (group by company)."""
import sys; sys.path.insert(0, "build")
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import clean

d = clean.load(sys.argv[1])
d["company_key"] = d.company_name.where(d.company_name != "", d.email).str.lower()
d["conversion_type_g"] = d.conversion_type.replace({"Meeting Booked": "Request Demo"})
d["conversion_type_g"] = d.conversion_type_g.where(d.conversion_type_g.map(d.conversion_type_g.value_counts()) >= 30, "other (small)")
HEARD = {"Word of mouth": "wom", "Current or former customer": "wom", "Received a Tremendous reward": "wom", "Search engine": "search", "unknown": "unknown"}
for c in ["industry", "form_page"]:  # pool groups under 30 leads, like conversion type
    d[c] = d[c].where(d[c].map(d[c].value_counts()) >= 30, "other (small)")
d["how_heard_pod"] = d.how_heard_c.map(HEARD).fillna("other")
d["vol_best"] = np.where(d.note_vol_band != "none stated", d.note_vol_band, d.vol_picklist)

# ---- Step 1: filter (disqualifiers) ----
d["dq_reason"] = np.select(
    [d.note_intent == "recipient support (not a buyer)", d.free_email == "free email", d.mobile == "Yes"],
    ["recipient support note", "free email", "mobile signup"], "")
d["dq"] = d.dq_reason != ""

CUT = pd.Timestamp("2024-04-01")
test = d[d.date >= CUT].copy()
train = d[(d.date < CUT) & ~d.company_key.isin(set(test.company_key))].copy()
months = 6
print(f"train {len(train)} leads (dropped {((d.date < CUT) & d.company_key.isin(set(test.company_key))).sum()} sharing a company with test); "
      f"test {len(test)} leads over {months} months, {test.y.sum()} good ({test.y.mean():.1%})")

# ---- Step 2: score (points model fit on non-disqualified training leads) ----
FEATS = ["vol_best", "seniority", "emp_band", "industry", "company_geo", "use_case", "conversion_type_g", "form_page",
         "source", "how_heard_pod", "note_intent", "has_company"]
def X(df, cols): return pd.get_dummies(df[cols].astype(str))
def fit(cols):
    tr = train[~train.dq]
    Xtr = X(tr, cols)
    m = LogisticRegression(C=0.5, max_iter=3000).fit(Xtr, tr.y)
    p = pd.Series(0.0, index=test.index)
    Xte = X(test, cols).reindex(columns=Xtr.columns, fill_value=0)
    p[:] = m.predict_proba(Xte)[:, 1]
    p[test.dq] = 0.0  # disqualified never reach scoring
    ptr = pd.Series(m.predict_proba(Xtr)[:, 1], index=tr.index)
    return m, Xtr.columns, p, ptr
model, cols, test["p"], p_train = fit(FEATS)
_, _, test["p_target"], _ = fit(FEATS + ["target"])
print(f"AUC on test: HubSpot grade {roc_auc_score(test.y, test.hubspot_grade.map({g: i for i, g in enumerate(['C3','C2','C1','B3','B2','B1','A3','A2','A1'])})):.3f}, "
      f"waterfall {roc_auc_score(test.y, test.p):.3f}, waterfall + target {roc_auc_score(test.y, test.p_target):.3f}")

TOTAL_GOOD = test.y.sum()
def summarize(name, sent):
    s = test[sent]
    good = s.y.sum(); wasted = len(s) - good
    return dict(system=name, sent=len(s), per_month=round(len(s) / months, 1), good=int(good),
                precision=f"{good / max(len(s), 1):.1%}", recall=f"{good / TOTAL_GOOD:.1%}",
                wasted=int(wasted), wasted_per_month=round(wasted / months, 1))

hub = test.is_mql_today == "MQL"
N = hub.sum()
vol_rule = test.vol_best.isin(["$10K-$100K", "$100K+"]) & ~test.dq
top_n = test.p.rank(ascending=False, method="first") <= N
top_n_t = test.p_target.rank(ascending=False, method="first") <= N
# Same recall as HubSpot with the fewest leads
order = test.sort_values("p", ascending=False)
need = int(order.y.cumsum().searchsorted(test.y[hub].sum()) + 1)
same_recall = test.index.isin(order.index[:need])
# Recommended cutoff, chosen on TRAINING scores only: score >= the training score where precision ~ 40%? keep simple:
thr = np.quantile(p_train, 0.70)  # top 30% of qualified training leads
rec = test.p >= thr

rows = [summarize("HubSpot MQL (A1/A2/B1)", hub),
        summarize("Step 1 only: HubSpot MQL minus disqualified", hub & ~test.dq),
        summarize("Volume rule ($10K+ stated, not disqualified)", vol_rule),
        summarize("Waterfall, matched volume (top N)", top_n),
        summarize("Waterfall, same recall as HubSpot", same_recall),
        summarize("Waterfall, recommended cutoff (top 30% of train)", rec),
        summarize("Sensitivity: waterfall + target acct, top N", top_n_t)]
print("\n===== Test slice results =====")
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nrecommended cutoff score = {thr:.3f}")

print("\n===== Where the waterfall saves sales time (matched volume) =====")
dropped = test[hub & ~top_n]; added = test[~hub & top_n]; kept = test[hub & top_n]
for lab, s in [("HubSpot MQLs the waterfall drops", dropped), ("Non-MQLs the waterfall adds", added), ("MQLs both send", kept)]:
    print(f"  {lab:36s} n={len(s):4d}  hit {s.y.mean():.1%}  good={int(s.y.sum())}")
print("  Why dropped MQLs fall out (top reasons):")
print("   disqualified:", dropped.dq_reason.replace("", "not dq").value_counts().to_dict())
print("   their conversion types:", dropped.conversion_type.value_counts().head(5).to_dict())
print("   their stated volume:", dropped.vol_best.value_counts().to_dict())
print("  Added non-MQLs, by HubSpot grade:", added.hubspot_grade.value_counts().to_dict())
print("  Added non-MQLs, by stated volume:", added.vol_best.value_counts().to_dict())

print("\n===== Disqualifier step on test slice =====")
print(test.groupby(test.dq_reason.replace("", "(passes)")).agg(n=("y", "size"), good=("y", "sum"), hit=("y", "mean"), mql=("is_mql_today", lambda s: (s == "MQL").sum())).round(3).to_string())

print("\n===== Three-way routing on test slice (cutoffs from training scores) =====")
hi, lo = np.quantile(p_train, 0.70), np.quantile(p_train, 0.40)
test["lane"] = np.select([test.dq, test.p >= hi, test.p >= lo], ["0 filtered out", "1 sales now", "2 watch"], "3 self-serve")
print(test.groupby("lane").agg(n=("y", "size"), per_month=("y", lambda s: round(len(s) / months, 1)), hit_10k=("y", "mean"),
                               ordered=("y_order", "mean"), good=("y", "sum")).round(3).to_string())

print("\n===== Hit rate by new grade (letter from score band, test slice) =====")
q = np.quantile(p_train, [0.40, 0.70, 0.90])
test["new_grade"] = np.select([test.dq, test.p >= q[2], test.p >= q[1], test.p >= q[0]], ["D (filtered)", "A", "B", "C"], "C-")
print(test.groupby("new_grade").y.agg(["size", "mean"]).round(3).to_string())
print("\nHubSpot by grade on the same slice:")
print(test.groupby("hubspot_grade").y.agg(["size", "mean"]).round(3).to_string())

print("\n===== Bootstrap: is the matched-volume gain real? (2,000 resamples of the test slice) =====")
rng = np.random.default_rng(0)
idx = np.arange(len(test)); yv = test.y.values; h = hub.values; w = top_n.values
diffs = []
for _ in range(2000):
    b = rng.choice(idx, len(idx), replace=True)
    diffs.append(yv[b][w[b]].sum() - yv[b][h[b]].sum())
lo_, hi_ = np.percentile(diffs, [2.5, 97.5])
print(f"  extra good leads found by waterfall vs HubSpot at matched volume: {np.mean(diffs):.1f} (95% range {lo_:.0f} to {hi_:.0f}); "
      f"share of resamples where waterfall wins: {np.mean(np.array(diffs) > 0):.1%}")
