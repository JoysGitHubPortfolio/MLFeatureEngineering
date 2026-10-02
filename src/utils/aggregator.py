import pandas as pd

META_COLS = ["year", "month"]


def feature_numeric_cols(
    df: pd.DataFrame,
    min_support_pct: float = 0.0,
    meta_cols: list[str] = META_COLS,
) -> pd.Index:
    """Numeric feature columns, excluding metadata and anything below the
    minimum non-zero support threshold."""
    cols = df.select_dtypes(include=["number"]).columns.difference(meta_cols)
    support = (df[cols] > 0).mean()
    return support[support >= min_support_pct].index


def active_stats(df: pd.DataFrame, cols: list[str] | pd.Index) -> pd.DataFrame:
    """Per-column summary over active (non-zero) rows, with the coefficient of
    variation normalising dispersion against the active mean."""
    active = df[cols].where(df[cols] > 0)
    stats = pd.DataFrame(
        {
            "non_zero_pct": active.notna().mean(),
            "mean_all": df[cols].mean(),
            "mean_active": active.mean(),
            "std_active": active.std(),
        }
    )
    stats["cv_active"] = stats["std_active"] / stats["mean_active"].abs()
    return stats
