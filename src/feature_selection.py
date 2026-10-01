import numpy as np
import pandas as pd

# 1. Load the cleaned dataset
file_path = "data/output/personas_cleaned.parquet"
df = pd.read_parquet(file_path)
print(f"Loaded Cleaned Data Shape: {df.shape}")

# 2. Isolate Feature Numeric Columns (excluding metadata and IDs)
all_numeric = df.select_dtypes(include=["number"]).columns
meta_cols = ["year", "month"]
feature_cols = [c for c in all_numeric if c not in meta_cols]

# 3. Apply Minimum Support Threshold (< 1% non-zero support filter)
min_support_pct = 0.01
non_zero_rates = (df[feature_cols] > 0).mean()
supported_cols = non_zero_rates[non_zero_rates >= min_support_pct].index

print(
    f"Filtered out {len(feature_cols) - len(supported_cols)} zero-inflated features (<1% support)."
)

# 4. Dedicated YouTube Feature Deep-Dive (Google Brief Focus)
yt_cols = [c for c in df.columns if "youtube" in c.lower()]
print(f"\n--- YouTube Feature Space Analysis ({len(yt_cols)} columns) ---")

# Compute rigorous normalised stats specifically for YouTube variables
yt_metrics = pd.DataFrame(
    {
        "non_zero_pct": (df[yt_cols] > 0).mean(),
        "mean_all": df[yt_cols].mean(),
        "mean_active": df[yt_cols].apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0
        ),
        "std_active": df[yt_cols].apply(
            lambda x: x[x > 0].std() if (x > 0).sum() > 1 else 0
        ),
    }
)
# Normalise standard deviation relative to the mean (Coefficient of Variation)
yt_metrics["cv_active"] = yt_metrics["std_active"] / yt_metrics["mean_active"].abs()
yt_metrics = yt_metrics.sort_values("mean_all", ascending=False)
print("\nTop YouTube Measures by Mean Volume & Normalised Dispersion (CV):")
print(yt_metrics.head(15))

# 5. Statistical Validity Audit for Core YouTube Engagement (`time_spent_youtube`)
target_col = "time_spent_youtube"
if target_col in df.columns:
    yt_active_series = df[df[target_col] > 0][target_col]
    
    # Adhering to Sath's rule: checking unique contributors vs person-months if user_id is present
    base_n = len(yt_active_series)
    unique_users = df.loc[df[target_col] > 0, "user_id"].nunique() if "user_id" in df.columns else base_n
    
    mean_val = yt_active_series.mean()
    std_err = yt_active_series.sem()
    ci_95 = 1.96 * std_err

    print(f"\n=== STATISTICAL VALIDITY AUDIT: `{target_col}` ===")
    print(f"  - Active Contributor-Months (Base N): {base_n} / {len(df)}")
    print(f"  - Unique Active Contributors: {unique_users}")
    print(f"  - Mean Time Spent (Active): {mean_val:.2f} seconds")
    print(f"  - 95% Confidence Interval: ±{ci_95:.2f} seconds")