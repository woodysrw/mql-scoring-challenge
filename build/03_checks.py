"""Leakage + interaction checks."""
import sys; sys.path.insert(0, "build")
import pandas as pd, numpy as np, clean
d = clean.load(sys.argv[1])
print("Target-account hit rate by HubSpot letter:")
print(d[d.target != "unknown"].pivot_table(index="grade_letter", columns="target", values="y", aggfunc=["mean", "size"]).round(2))
print("\nTarget share by half (stable list would be flat):"); print(d[d.target != "unknown"].groupby("half").target.apply(lambda s: (s == "TRUE").mean()).round(2))
print("\nTarget TRUE: ordered but not 10K:", ((d.target == "TRUE") & (d.y_order == 1) & (d.y == 0)).sum(), "| never ordered:", ((d.target == "TRUE") & (d.y_order == 0)).sum())
print("\nMobile=Yes conversion types:", d[d.mobile == "Yes"].conversion_type.value_counts().to_dict())
print("\nSeniority hit rate within work-email, known-company leads:")
k = d[(d.free_email == "work email") & (d.emp_band != "unknown")]
print(k.groupby("seniority").y.agg(["size", "mean"]).round(3))
print("\nEmployee band hit rate (known company, work email):"); print(k.groupby("emp_band").y.agg(["size", "mean"]).round(3))
print("\nHubSpot grade vs outcome inside B2 by vol_picklist:")
b2 = d[d.hubspot_grade == "B2"]; print(b2.groupby("vol_picklist").y.agg(["size", "mean"]).round(3))
print("\nGood leads missed by MQL (y=1, not MQL):", ((d.y == 1) & (d.is_mql_today != "MQL")).sum(), "of", d.y.sum())
print(d[(d.y == 1) & (d.is_mql_today != "MQL")].hubspot_grade.value_counts().to_dict())
