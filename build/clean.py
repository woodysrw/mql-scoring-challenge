"""Shared cleaning + feature engineering. Used by every analysis/build script."""
import re
import numpy as np
import pandas as pd

SOURCE_MAP = {
    "paid search": "Paid Search", "google / cpc": "Paid Search", "ppc": "Paid Search", "paid_search": "Paid Search",
    "direct traffic": "Direct", "direct": "Direct",
    "offline": "Offline", "offline sources": "Offline",
    "organic search": "Organic Search", "seo": "Organic Search", "organic": "Organic Search",
    "referrals": "Referral", "referral": "Referral",
    "paid social": "Paid Social", "paid_social": "Paid Social", "linkedin ads": "Paid Social",
    "other campaigns": "Other Campaigns", "email marketing": "Email Marketing", "social media": "Organic Social",
}
US = {"united states", "u.s.", "us", "usa", "united states of america"}
UK = {"united kingdom", "uk", "great britain"}
FREE = {"gmail.com", "yahoo.com", "outlook.com", "aol.com", "icloud.com", "hotmail.com"}


def country(x):
    x = x.strip()
    if not x: return ""
    l = x.lower()
    if l in US: return "United States"
    if l in UK: return "United Kingdom"
    return x.replace("The Netherlands", "Netherlands")


def emp_band(x):
    if not x: return "unknown"
    n = int(float(x))
    for hi, lab in [(10, "1-10"), (50, "11-50"), (200, "51-200"), (1000, "201-1K"), (5000, "1K-5K")]:
        if n <= hi: return lab
    return "5K+"


def seniority(t):
    t = t.lower()
    if not t: return "unknown"
    if re.search(r"intern|student", t): return "intern/student"
    if re.search(r"founder|owner|ceo|coo|cfo|president|managing partner|chief", t): return "founder/C-level"
    if re.search(r"\bvp\b|vice president", t): return "VP"
    if re.search(r"director|head of", t): return "director/head"
    if re.search(r"manager|lead\b", t): return "manager/lead"
    return "IC"


def function(t):
    t = t.lower()
    if not t: return "unknown"
    for pat, lab in [(r"founder|owner|ceo|coo|president|managing partner", "exec"),
                     (r"research|professor|clinical", "research"),
                     (r"marketing|growth|demand", "marketing"),
                     (r"\bhr\b|people|talent|employee experience", "HR/people"),
                     (r"operations", "operations"), (r"administration|office", "admin/office"),
                     (r"finance", "finance"), (r"customer success|community", "CS/community"),
                     (r"program", "program"), (r"\bit\b", "IT"), (r"business", "business")]:
        if re.search(pat, t): return lab
    return "other"


RECIPIENT = re.compile(r"never received my payment|is this legit|change the card|redeem|expired", re.I)
MONEY = re.compile(r"\$([\d,.]+)\s*(K)?\s*a\s*(year|month)", re.I)


def note_intent(n):
    if not n: return "no note"
    if RECIPIENT.search(n): return "recipient support (not a buyer)"
    if MONEY.search(n): return "buyer, states $ volume"
    if re.search(r"pricing", n, re.I): return "buyer, pricing question"
    return "buyer, exploring (no volume)"


def note_volume(n):
    m = MONEY.search(n or "")
    if not m: return np.nan
    v = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
    return v * (12 if m.group(3).lower() == "month" else 1)


def vol_band(v):
    if np.isnan(v): return "none stated"
    if v < 10_000: return "<$10K"
    if v < 100_000: return "$10K-$100K"
    return "$100K+"


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.apply(lambda s: s.str.strip())
    out = df.copy()
    out["date"] = pd.to_datetime(df.lead_created_date)
    out["half"] = out.date.dt.year.astype(str) + "-H" + np.where(out.date.dt.month <= 6, "1", "2")
    out["source"] = df.lead_source.str.lower().map(SOURCE_MAP).fillna(df.lead_source)
    out["company_country_c"] = df.company_country.map(country)
    out["lead_country_c"] = df.lead_country.map(country)
    out["company_geo"] = np.select([out.company_country_c == "United States", out.company_country_c == ""],
                                   ["US", "unknown"], "non-US")
    out["lead_geo"] = np.select([out.lead_country_c == "United States", out.lead_country_c == ""],
                                ["US", "unknown"], "non-US")
    out["emp_band"] = df.company_employees.map(emp_band)
    out["employees"] = pd.to_numeric(df.company_employees, errors="coerce")
    out["free_email"] = df.email.str.split("@").str[1].str.lower().isin(FREE).map({True: "free email", False: "work email"})
    out["has_company"] = (df.company_name != "").map({True: "yes", False: "no"})
    out["industry"] = df.company_industry.replace("", "unknown")
    out["seniority"] = df.job_title.map(seniority)
    out["function"] = df.job_title.map(function)
    out["form_page"] = df.conversion_form.map(lambda f: f.split(":")[0] if ":" in f else "(no page)")
    out["use_case"] = df.self_reported_use_case.replace("", "unknown")
    out["vol_picklist"] = df.self_reported_annual_volume.replace("", "unknown")
    out["how_heard_c"] = df.how_heard.replace("", "unknown")
    out["conference_c"] = df.conference.replace("", "no")
    out["mobile"] = df.mobile_signup.replace("", "unknown")
    out["target"] = df.is_target_account.replace("", "unknown")
    out["note_intent"] = df.lead_note.map(note_intent)
    out["note_volume"] = df.lead_note.map(note_volume)
    out["note_vol_band"] = out.note_volume.map(vol_band)
    for flag, pat in [("note_vendor_pain", r"vendor is slow|amazon cards by hand"),
                      ("note_urgent", r"live next month|can we get a demo"),
                      ("note_finance", r"one invoice"), ("note_intl", r"10 countries")]:
        out[flag] = df.lead_note.str.contains(pat, case=False, regex=True).map({True: "yes", False: "no"})
    out["grade_letter"] = df.hubspot_grade.str[0]
    out["grade_number"] = df.hubspot_grade.str[1]
    out["is_mql_today"] = df.hubspot_grade.isin(["A1", "A2", "B1"]).map({True: "MQL", False: "not MQL"})
    if "reached_10k_in_12m" in df:
        out["y"] = df.reached_10k_in_12m.astype(int)
        out["y_order"] = df.ordered_within_6mo.astype(int)
    return out
