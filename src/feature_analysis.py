from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from .utils.bootstrapper import cluster_bootstrap
from .utils.cleaner import clean_df
from .utils.plotter import plot_cannibalisation, plot_data_audit

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/input/personas_monthly_sample.parquet"
FIG_DIR = ROOT / "data/output/figures"
SEED, N_BOOT, ALPHA = 42, 2000, 0.05
MIN_BASE = 30  # fewer identifying people than this and a robustness check is untestable
# gemini is left out: it has effectively zero metered time (max 2 seconds).
PLATFORMS = {"tiktok": "TikTok", "instagram": "Instagram", "facebook": "Facebook", "chrome": "Chrome",
             "safari": "Safari", "chatgpt": "ChatGPT", "amazon": "Amazon", "google_search": "Google Search"}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns the whole panel, the rows with time-spend data, and the analysis base."""
    df = clean_df(pd.read_parquet(INPUT_PATH))
    df["period"] = df["year"] * 100 + df["month"]
    days = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)).dt.days_in_month
    # On iOS covered_days is ~7x the calendar days, which breaks every per_day column.
    df["exposure_ratio"] = df["time_spent_covered_days"] / days

    # The time_spent module is blank for every channel at once: blank = not measured.
    ts = df[df["time_spent_youtube"].notna()].copy()
    # Sum channels ourselves: time_spent_total_seconds carries a 9,999,999 sentinel.
    ts["all_seconds"] = ts[[f"time_spent_{p}" for p in ["youtube", *PLATFORMS]]].sum(axis=1)

    # The base: YouTube users with 2+ measured months (one month has no within-person change).
    person = ts.groupby("user_id")
    base = ts[(person["time_spent_youtube"].transform("max") > 0) & (person["period"].transform("size") >= 2)]
    return df, ts, base


def estimate(d: pd.DataFrame, platform: str, rng: np.random.Generator) -> dict:
    """Within-person effect of platform time on YouTube time.

    Every variable is logged and has the person's own average subtracted, so
    only changes within a person remain (person fixed effects); month dummies
    remove panel-wide seasonality. 'Rest' (all other metered time) absorbs
    busy months and meter coverage, so the platform coefficient is the trade-off."""
    d = d[d.groupby("user_id")["period"].transform("size") >= 2]
    rest = (d["all_seconds"] - d["time_spent_youtube"] - d[f"time_spent_{platform}"]).clip(lower=0)
    X = pd.concat([np.log1p(d[f"time_spent_{platform}"]), np.log1p(rest),
                   pd.get_dummies(d["period"], drop_first=True, dtype=float)], axis=1)
    y = np.log1p(d["time_spent_youtube"])
    X = (X - X.groupby(d["user_id"]).transform("mean")).to_numpy()
    y = (y - y.groupby(d["user_id"]).transform("mean")).to_numpy()

    fit = lambda rows: np.linalg.lstsq(X[rows], y[rows], rcond=None)[0][0]
    lo, hi, p = cluster_bootstrap(fit, d["user_id"].to_numpy(), N_BOOT, rng)
    # Only people whose platform time actually changes month to month identify the effect.
    n_people = int((d.groupby("user_id")[f"time_spent_{platform}"].nunique() > 1).sum())
    return {"beta": fit(np.arange(len(y))), "lo": lo, "hi": hi, "p": p, "n_people": n_people}


def without_heavy_users(d: pd.DataFrame) -> pd.DataFrame:
    """Drops the top 5% of people by average monthly YouTube time."""
    heaviness = d.groupby("user_id")["time_spent_youtube"].mean().rank(pct=True)
    return d[d["user_id"].map(heaviness) <= 0.95]


CHECKS = {  # each re-asks the question on a base where one worry is removed
    "Android only": lambda d: d[d["identity_device_platform"] == "android"],
    "iOS only": lambda d: d[d["identity_device_platform"] == "ios"],
    "Without top 5% YouTube users": without_heavy_users,
    "Stable panel (6+ months)": lambda d: d[d.groupby("user_id")["period"].transform("size") >= 6],
    "Full-month exposure rows": lambda d: d[np.isclose(d["exposure_ratio"], 1) | np.isclose(d["exposure_ratio"], 7)],
}


def verdict(row: pd.Series, checks: pd.DataFrame) -> str:
    """Needs: significant after FDR, sign never flips in a testable check, most checks exclude zero."""
    if row["q"] >= ALPHA:
        return "Not established"
    c = checks[checks["n_people"] >= MIN_BASE]
    same_sign = (np.sign(c["beta"]) == np.sign(row["beta"])).all()
    mostly_significant = ((c["lo"] > 0) | (c["hi"] < 0)).mean() > 0.5
    if not (same_sign and mostly_significant):
        return "Fragile"
    return "Supports YouTube" if row["beta"] > 0 else "Cannibalises YouTube"


def main() -> None:
    rng = np.random.default_rng(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, ts, base = load()
    print(f"{df.user_id.nunique()} people in panel; {ts.user_id.nunique()} with time-spend data; "
          f"{base.user_id.nunique()} in the base.")

    # 1. Headline estimate per platform, corrected for testing 8 platforms at once
    res = pd.DataFrame.from_dict({p: estimate(base, p, rng) for p in PLATFORMS}, orient="index")
    res["q"] = multipletests(res["p"], method="fdr_bh")[1]

    # 2. Re-estimate under each robustness check, then decide a verdict
    checks = pd.DataFrame([{"check": name, "platform": p, **estimate(subset(base), p, rng)}
                           for name, subset in CHECKS.items() for p in PLATFORMS])
    res["verdict"] = [verdict(r, checks[checks["platform"] == p]) for p, r in res.iterrows()]
    print(res.round(3).to_string())
    print(checks.pivot(index="platform", columns="check", values="beta").round(2).to_string())

    # 3. Charts
    plot_data_audit(df, ts, PLATFORMS, FIG_DIR / "fig_1_data_audit.png")
    plot_cannibalisation(res, PLATFORMS, FIG_DIR / "fig_2_cannibalisation_analysis.png")


if __name__ == "__main__":
    main()
