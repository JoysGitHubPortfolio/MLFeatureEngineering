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

# 3. Apply Minimum Support Threshold (Filter out features with < 1% non-zero support)
min_support_pct = 0.01
non_zero_rates = (df[feature_cols] > 0).mean()
supported_cols = non_zero_rates[non_zero_rates >= min_support_pct].index

print(
    f"Filtered out {len(feature_cols) - len(supported_cols)} zero-inflated features (<{min_support_pct*100}% non-zero support)."
)

# 4. Compute Statistical Dispersion Metrics (CV & IQR) for Supported Features
numeric_df = df[supported_cols]
metrics = pd.DataFrame(
    {
        "non_zero_pct": non_zero_rates[supported_cols],
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "cv": numeric_df.std() / numeric_df.mean().abs(),
        "iqr": numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
    }
).sort_values("cv", ascending=False)

print("\nTop 20 Supported Features by Relative Dispersion (CV):")
print(metrics.head(20))

# 5. Extract YouTube & Cross-Platform Focus Areas (Google's Core Brief)
yt_features = [c for c in df.columns if "youtube" in c.lower()]
transition_features = [c for c in df.columns if c.startswith("transitions_")]
time_features = [c for c in df.columns if c.startswith("time_spent_")]

print(f"\n--- Google Brief Feature Scope ---")
print(f"YouTube-specific features identified: {len(yt_features)}")
print(f"Transition feature pathways: {len(transition_features)}")
print(f"Time-spend behavioral metrics: {len(time_features)}")

# Example: Quick statistical validity check for YouTube time spend
if "time_spent_youtube" in df.columns:
    yt_active = df[df["time_spent_youtube"] > 0]["time_spent_youtube"]
    mean_val = yt_active.mean()
    std_err = yt_active.sem()
    ci_95 = 1.96 * std_err
    base_n = len(yt_active)

    print(f"\nStatistical Validity Audit — `time_spent_youtube` (Active Users):")
    print(f"  - Base (N): {base_n} / {len(df)} total contributor-months")
    print(f"  - Mean Time Spent: {mean_val:.2f} seconds")
    print(f"  - 95% Confidence Interval: ±{ci_95:.2f} seconds")