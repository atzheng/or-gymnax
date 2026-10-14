Ad-hoc analysis scripts behind LSTD_DQ_REPORT.md Parts 4–5 (A = 0.2). They expect local copies of
the R2 outputs listed in PROJECT_SUMMARY.md's run log, laid out as in the session scratch dir:
`fscan/` (f-scan), `ate4/` (λ(p) truth), `fork4/`, `forktd*/`, `forkall/` (estimator CSVs).

- `fs_an.py`: f(s) and ATE vs a reference threshold A from the f-scan.
- `tr4.py`, `tr5.py`: true ATE and DQ estimand for B above / below A = 0.2.
- `summ.py <csv> <ate>`: mean / sd / bias % / RMSE / right-sign rate for every estimator column.
- `cmp.py`, `cmp5.py`: cross-setting ranking on the dev seed and held-out tables.
- `plot4.py`: f(s) curve with A and B marked (f_curve_a020.png).
