import pandas as pd, sys
df = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
print(df.shape)
for c in df.columns:
    s = df[c].str.strip()
    filled = (s != "").mean()
    nun = s[s != ""].nunique()
    print(f"\n## {c}: filled {filled:.0%}, {nun} distinct")
    if c in ("lead_id","email","company_name","lead_created_date"): continue
    if nun <= 40:
        print(s[s!=""].value_counts().to_string())
    else:
        print(s[s!=""].value_counts().head(15).to_string())
