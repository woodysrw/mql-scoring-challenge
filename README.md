# MQL scoring challenge

A three-step lead waterfall for Tremendous's GTM Engineer take-home. It filters out leads that almost never buy, scores the rest with a logistic regression, and sends each one to sales or tags it for a watch list or self-serve.

**Write-up:** https://woodysrw.github.io/mql-scoring-challenge/

| File | What it is |
|---|---|
| [Grades sheet](https://docs.google.com/spreadsheets/d/1NOZSrTFMm-xk71IV4JOopaQi1iQQnG1yD0W-p6gN1K4/edit?usp=sharing) | Grade and rep line for each of the 490 newer leads, sorted A to D (Google Sheet, view only) |
| [`out/grades.csv`](out/grades.csv) | The same grades as written by `05_grade.py` |
| [`build/`](build/) | Python scripts that produce the analysis and the grades |
| [`out/*.txt`](out/) | Script output behind every number in the write-up |
| [`process-log.txt`](process-log.txt) | Findings, decisions and caveats, logged as the work happened |

## Run it

The input data is Tremendous's MQL Scoring Challenge Data sheet, which isn't included here. Download its two tabs as CSV to `data/past_leads.csv` and `data/newer_leads.csv`, then:

```bash
pip install pandas scipy scikit-learn statsmodels
python build/04_waterfall_eval.py data/past_leads.csv
python build/05_drift.py data/past_leads.csv data/newer_leads.csv
python build/05_grade.py data/past_leads.csv data/newer_leads.csv   # writes out/grades.csv
```

All records in the challenge data are synthetic.
