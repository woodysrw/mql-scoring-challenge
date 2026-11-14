"""Grade the newer leads with the lead waterfall: filter -> score -> route, plus a rep line.
Refit on all past leads (same features and settings as 04_waterfall_eval.py, which tested it on a time split).
Grade = lane. Cutoffs are the 70th and 40th percentiles of past scores, as in the eval.
Usage: python build/05_grade.py data/past_leads.csv data/newer_leads.csv  -> out/grades.csv"""
import sys; sys.path.insert(0, "build")
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
import clean

past, new = clean.load(sys.argv[1]), clean.load(sys.argv[2])
raw_new = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False).apply(lambda s: s.str.strip())

HEARD = {"Word of mouth": "wom", "Current or former customer": "wom", "Received a Tremendous reward": "wom", "Search engine": "search", "unknown": "unknown"}
conv_counts = past.conversion_type.replace({"Meeting Booked": "Request Demo"}).value_counts()
KEEP_CONV = set(conv_counts[conv_counts >= 30].index)  # small or never-seen types share one bucket

SMALL = {c: set(v[v >= 30].index) for c in ["industry", "form_page"] for v in [past[c].value_counts()]}

def prep(d):
    d["industry_raw"] = d.industry
    for c, keep in SMALL.items():  # pool groups under 30 past leads (and never-seen values), as in 04
        d[c] = d[c].where(d[c].isin(keep), "other (small)")
    d["conversion_type_g"] = d.conversion_type.replace({"Meeting Booked": "Request Demo"})
    d["conversion_type_g"] = d.conversion_type_g.where(d.conversion_type_g.isin(KEEP_CONV), "other (small)")
    d["how_heard_pod"] = d.how_heard_c.map(HEARD).fillna("other")
    d["vol_best"] = np.where(d.note_vol_band != "none stated", d.note_vol_band, d.vol_picklist)
    d["dq_reason"] = np.select(
        [d.note_intent == "recipient support (not a buyer)", d.free_email == "free email", d.mobile == "Yes"],
        ["recipient", "free email", "mobile"], "")
    return d
past, new = prep(past), prep(new)

# ---- Score: logistic regression on unfiltered past leads ----
FEATS = ["vol_best", "seniority", "emp_band", "industry", "company_geo", "use_case", "conversion_type_g", "form_page",
         "source", "how_heard_pod", "note_intent", "has_company"]
def X(df): return pd.get_dummies(df[FEATS].astype(str), prefix_sep="=")
tr = past[past.dq_reason == ""]
Xtr = X(tr)
m = LogisticRegression(C=0.5, max_iter=3000).fit(Xtr, tr.y)
Xnew = X(new).reindex(columns=Xtr.columns, fill_value=0)
p_tr = m.predict_proba(Xtr)[:, 1]
new["p"] = np.where(new.dq_reason == "", m.predict_proba(Xnew)[:, 1], 0.0)
HI, LO = np.quantile(p_tr, 0.70), np.quantile(p_tr, 0.40)

# ---- Route = grade ----
new["grade"] = np.select([new.dq_reason != "", new.p >= HI, new.p >= LO], ["D", "A", "B"], "C")
new["score"] = (new.p * 100).round().astype(int)

# ---- Rep line: action + the two fields that pushed the score most ----
coef = pd.Series(m.coef_[0], index=Xtr.columns)
base = Xtr.mean()
# Explain in plain groups: first action = conversion type + page; volume = stated volume + note
MERGE = {"form_page": "conversion_type_g", "note_intent": "vol_best"}
group = pd.Series([MERGE.get(c.split("=")[0], c.split("=")[0]) for c in Xtr.columns], index=Xtr.columns)
contrib = ((Xnew.astype(float) - base) * coef).T.groupby(group).sum().T  # rows = leads, cols = feature groups

def money(v): return f"${v/1000:,.0f}K" if v >= 1000 else f"${v:,.0f}"
CONV = {"Request Demo": "requested a demo", "Signup": "signed up", "Send Reward": "sent a test reward", "Chat Inquiry": "came in by chat",
        "Content Download": "downloaded content", "Recipient Demo": "toured the recipient demo", "LinkedIn Form": "filled a LinkedIn form",
        "Webinar Registration": "registered for a webinar", "Email Inquiry": "emailed in"}
def phrase(g, r, rw):
    if g == "vol_best":
        if not np.isnan(r.note_volume): return f"note says {money(r.note_volume)} a year"
        if r.note_intent == "buyer, pricing question": return "asked about pricing, no volume given"
        return f"says {r.vol_best} a year" if r.vol_best != "unknown" else "no volume given"
    if g == "seniority": return rw.job_title if rw.job_title else "no job title"
    if g == "emp_band": return f"{int(r.employees):,} employees" if not np.isnan(r.employees) else "company size unknown"
    if g == "use_case": return f"use case: {r.use_case.lower()}" if r.use_case != "unknown" else "no use case given"
    if g == "conversion_type_g": return CONV.get(r.conversion_type, r.conversion_type.lower())
    if g == "industry": return f"{r.industry_raw} industry" if r.industry_raw != "unknown" else "industry unknown"
    if g == "company_geo": return {"US": "US company", "non-US": "non-US company"}.get(r.company_geo, "country unknown")
    if g == "source": return f"via {r.source.lower()}"
    if g == "how_heard_pod": return f"heard via {r.how_heard_c.lower()}" if r.how_heard_c != "unknown" else "didn't say how they heard of us"
    if g == "note_intent": return {"buyer, pricing question": "asked about pricing", "buyer, exploring (no volume)": "note is exploring, no volume",
                                   "buyer, states $ volume": "stated volume in note"}.get(r.note_intent, "no note")
    if g == "form_page": return f"converted on {r.form_page.lower()}"
    if g == "has_company": return "company on record" if r.has_company == "yes" else "no company record"
    return g

ACTION = {"A": "Call within a day", "B": "Self-serve for now; reach out after a first order",
          "C": "Self-serve onboarding, no rep"}
DQ = {"recipient": "Reward recipient asking about a payment. Send to support",
      "free email": "Personal email. Nurture only",
      "mobile": "Mobile signup; past mobile signups never reached $10K. Nurture only"}
NEW_TYPES = {"Referred from Partner": "partner referral", "Client-to-Client Referral v1": "client referral"}

lines = []
for i, r in new.iterrows():
    rw = raw_new.loc[i]
    if r.dq_reason:
        line = DQ[r.dq_reason]
    else:
        c = contrib.loc[i].sort_values(ascending=(r.grade == "C"))
        top = [phrase(g, r, rw) for g in c.index[:2]]
        why = "Held back by" if r.grade == "C" else "Why"
        line = f"{ACTION[r.grade]}. {why}: {top[0]}; {top[1]}"
    flags = []
    if r.conversion_type in NEW_TYPES: flags.append(f"{NEW_TYPES[r.conversion_type]}, a lead type the model hasn't seen; check by hand")
    if r.source == "AI Referrals": flags.append("source 'AI Referrals' is new; the model gives it no weight")
    if flags: line += ". Note: " + "; ".join(flags)
    lines.append(line + ".")
new["rep_line"] = lines

out = new[["lead_id", "grade", "score", "rep_line"]]
out.to_csv("out/grades.csv", index=False)

# ---- Summary ----
months = (new.date.max().to_period("M") - new.date.min().to_period("M")).n + 1
print(f"graded {len(new)} newer leads over {months} months -> out/grades.csv")
print(f"cutoffs (past score percentiles): A >= {HI:.3f} (70th), B >= {LO:.3f} (40th)")
g = new.grade.value_counts().reindex(list("ABCD")).fillna(0).astype(int)
print(pd.DataFrame({"leads": g, "share": (g / len(new) * 100).round(1), "per_month": (g / months).round(1)}).to_string())
print(f"\nHubSpot MQLs in the 490: {(new.is_mql_today=='MQL').sum()} ({(new.is_mql_today=='MQL').sum()/months:.1f}/month)")
print("Filtered (D) by reason:", new[new.grade == "D"].dq_reason.value_counts().to_dict())
print("\nNew grade x HubSpot grade:"); print(pd.crosstab(new.grade, new.hubspot_grade).to_string())
print("\nNew grade x HubSpot MQL:"); print(pd.crosstab(new.grade, new.is_mql_today).to_string())
print("\nScore range by grade:"); print(new.groupby("grade").score.agg(["min", "median", "max"]).to_string())
print("\nTop 10 weights (log-odds):"); print(coef.sort_values(ascending=False).head(10).round(2).to_string())
print("\nExample lines:")
for gr in "ABCD":
    for _, r in out[out.grade == gr].sort_values("score", ascending=False).head(2).iterrows():
        print(f"  {r.lead_id} {gr} {r.score:3d}  {r.rep_line}")
