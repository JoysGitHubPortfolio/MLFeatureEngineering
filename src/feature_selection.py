import pandas as pd

from .utils.aggregator import active_stats, feature_numeric_cols

INPUT_PATH = "data/output/personas_cleaned.parquet"
MIN_SUPPORT_PCT = 0.01
TARGET_COL = "time_spent_youtube"

# 1. Load the cleaned dataset
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded Cleaned Data Shape: {df.shape}")

# 2. Isolate feature numerics, then drop zero-inflated features (<1% support)
feature_cols = feature_numeric_cols(df)
supported_cols = feature_numeric_cols(df, MIN_SUPPORT_PCT)
print(
    f"Filtered out {len(feature_cols) - len(supported_cols)} zero-inflated features (<1% support)."
)

# 3. Dedicated YouTube feature deep-dive (Google brief focus)
yt_cols = [c for c in feature_cols if "youtube" in c.lower()]
print(f"\n--- YouTube Feature Space Analysis ({len(yt_cols)} columns) ---")
yt_metrics = active_stats(df, yt_cols).sort_values("mean_all", ascending=False)
print("\nTop YouTube Measures by Mean Volume & Normalised Dispersion (CV):")
print(yt_metrics.head(15))

# 4. Statistical validity audit for core YouTube engagement
if TARGET_COL in df.columns:
    active = df[df[TARGET_COL] > 0]
    # Adhering to Sath's rule: unique contributors vs person-months
    unique_users = active["user_id"].nunique() if "user_id" in df.columns else len(active)

    print(f"\n=== STATISTICAL VALIDITY AUDIT: `{TARGET_COL}` ===")
    print(f"  - Active Contributor-Months (Base N): {len(active)} / {len(df)}")
    print(f"  - Unique Active Contributors: {unique_users}")
    print(f"  - Mean Time Spent (Active): {active[TARGET_COL].mean():.2f} seconds")
    print(f"  - 95% Confidence Interval: ±{1.96 * active[TARGET_COL].sem():.2f} seconds")
