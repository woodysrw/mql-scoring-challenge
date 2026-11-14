"""Before grading: do the 490 newer leads look like the past leads? Fill rates, mix, new values, note formats."""
import sys; sys.path.insert(0, "build")
import pandas as pd, numpy as np, clean
old = clean.load(sys.argv[1]); new = clean.load(sys.argv[2])
print(f"past {len(old)} leads, newer {len(new)} leads ({new.date.min().date()} to {new.date.max().date()})")
raw_new = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
raw_old = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
print("\n== Fill rate, past vs newer ==")
for c in raw_new.columns:
    a = (raw_old[c].str.strip() != "").mean() if c in raw_old else np.nan; b = (raw_new[c].str.strip() != "").mean()
    flag = "  <-- shift" if abs(a - b) > 0.10 else ""
    print(f"  {c:28s} {a:6.0%} {b:6.0%}{flag}")
print("\n== Mix shifts (share, past -> newer), values moving 5+ points ==")
for c in ["source", "conversion_type", "emp_band", "free_email", "seniority", "vol_picklist", "use_case", "note_intent",
          "mobile", "company_geo", "how_heard_c", "industry", "form_page", "hubspot_grade"]:
    a = old[c].value_counts(normalize=True); b = new[c].value_counts(normalize=True)
    t = pd.concat([a, b], axis=1, keys=["past", "newer"]).fillna(0)
    t = t[(t.newer - t.past).abs() >= 0.05]
    unseen = sorted(set(new[c]) - set(old[c]))
    if len(t) or unseen:
        print(f"-- {c}"); print((t * 100).round(1).to_string()) if len(t) else None
        if unseen: print("   values never seen in past leads:", unseen)
print("\n== ad_platform values in newer leads ==", raw_new.ad_platform.replace("", "(blank)").value_counts().to_dict())
print("\n== HubSpot MQL share: past", f"{(old.is_mql_today=='MQL').mean():.1%}", "newer", f"{(new.is_mql_today=='MQL').mean():.1%}")
print("\n== Notes: how many the regex reads ==")
n = new[raw_new.lead_note.str.strip() != ""]
print(n.note_intent.value_counts().to_string())
ex = n[n.note_intent == "buyer, exploring (no volume)"]
print(f"\nNotes with no recipient/volume/pricing match: {len(ex)}. Distinct texts (first 12):")
for t in raw_new.loc[ex.index, "lead_note"].drop_duplicates().head(12): print("  -", t[:150])
money_words = raw_new.lead_note.str.contains(r"\$|\d+k|thousand|per (year|month)|annual", case=False, regex=True) & (new.note_intent != "buyer, states $ volume") & (new.note_intent != "recipient support (not a buyer)")
print(f"\nNotes mentioning money that the regex missed: {money_words.sum()}")
for t in raw_new.loc[money_words, "lead_note"].head(8): print("  -", t[:150])
