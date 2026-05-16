# mobility-beliefs-redistribution

Do mobility beliefs reduce demand for redistribution? Evidence from LiTS IV across 37 transition economies.

This repo tests the **Prospect of Upward Mobility (POUM)** hypothesis on the
EBRD/World Bank *Life in Transition Survey IV* (2022–23, 37 economies, 37,478
respondents), with a fairness-belief moderator and a five-region heterogeneity split.

## Layout

```
paper.tex                 Final research proposal (LaTeX)
paper.pdf                 Compiled PDF
analysis/
  build_dataset.py        Cleans LiTS IV .dta -> parquet
  regressions.py          OLS / WLS / ordered logit / horse race / heterogeneity
output/
  tab*_*.csv              Tables used in the paper
  fig_*.png               Figures used in the paper
```

## Reproduce

```bash
# 1. Set up environment (uses uv: https://docs.astral.sh/uv/)
uv sync

# 2. Download LiTS IV microdata (.dta) from
#    https://www.ebrd.com/what-we-do/economic-research-and-data/data/lits.html
#    Place the file at the repo root as `lits_iv.dta`.

# 3. Build cleaned dataset and run all regressions
uv run python analysis/build_dataset.py
uv run python analysis/regressions.py
```

## Data note

`lits_iv.dta` is licensed by the EBRD and is not redistributed here. Download it
directly from the EBRD LiTS data portal.
