# ML Feature Engineering

Cleans a monthly persona panel, explores its features, and tests whether time on other platforms (TikTok, Instagram, Chrome, ChatGPT, etc.) supports or cannibalises time on YouTube.

## Setup

Requires Python 3.12+. Dependencies are listed in `dependencies/pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -r dependencies/pyproject.toml
```

Put the input data in `data/input/` (this folder is git-ignored):

- `personas_monthly_sample.parquet`: the raw monthly panel
- `feature_dictionary.csv`: column definitions

## Usage

Run each step from the repo root:

```bash
# 1. Clean the raw data and write data/output/personas_cleaned.parquet
python -m src.feature_cleaning --verbose

# 2. Summarise features, focusing on YouTube (needs step 1)
python -m src.feature_selection

# 3. Estimate the cannibalisation effects and save the charts
python -m src.feature_analysis
```

## How the analysis works

- Uses people who watch YouTube and have time-spend data for at least 2 months.
- For each platform, uses within-person (fixed effects) regression with month dummies to estimate how platform time relates to YouTube time.
- Builds confidence intervals with a cluster bootstrap over people, and corrects for testing 8 platforms with Benjamini–Hochberg FDR.
- Repeats the estimate in robustness checks (Android only, iOS only, without heavy users, stable panel, full-month rows).
- Labels each platform as **Supports YouTube**, **Cannibalises YouTube**, **Fragile**, or **Not established**.

## Project structure

```
src/
  feature_cleaning.py    # load, clean and save the panel
  feature_selection.py   # feature support and YouTube summary stats
  feature_analysis.py    # cannibalisation model and robustness checks
  utils/
    cleaner.py           # date parsing and sentinel-value fixes
    aggregator.py        # feature column selection and summary stats
    bootstrapper.py      # cluster bootstrap
    plotter.py           # charts
data/
  input/                 # raw data (not committed)
  output/                # cleaned data and figures
dependencies/            # pyproject.toml and uv.lock
ai_notes/                # prompts used to generate the modules
```

## Outputs

- `data/output/personas_cleaned.parquet`: cleaned panel
- `data/output/figures/fig_1_data_audit.png`: data coverage and quality checks
- `data/output/figures/fig_2_cannibalisation_analysis.png`: effect per platform
