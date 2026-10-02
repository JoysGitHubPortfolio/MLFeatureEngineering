# --- 1. GLOBALS & IMPORTS ---
import sys
import pandas as pd
from pathlib import Path
from .utils.aggregator import META_COLS, feature_numeric_cols
from .utils.cleaner import clean_df
from .utils.plotter import plot_top_cv_distributions

INPUT_PATH = "data/input/personas_monthly_sample.parquet"
OUTPUT_PATH = "data/output/personas_cleaned.parquet"


# --- 2. CORE LOGIC WRAPPED IN FUNCTIONS ---
def run_cleaning_pipeline(
    input_path: str = INPUT_PATH, 
    output_path: str = OUTPUT_PATH,
    verbose: bool = False
) -> pd.DataFrame:
    """Loads raw data, applies data quality fixes, and saves output."""
    print(f"Loading raw data: {input_path}...")
    df_raw = pd.read_parquet(input_path)

    print("Applying data cleaning rules...")
    df_clean = clean_df(df_raw)

    print(f"Saving cleaned data: {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(output_path, index=False)

    # --- Detailed Column & Type Summary ---
    string_cols = df_clean.select_dtypes(include=['string', 'object']).columns
    date_cols = df_clean.select_dtypes(include=['datetime', 'datetime64']).columns
    all_numeric = df_clean.select_dtypes(include=['number']).columns
    meta_cols = [c for c in META_COLS if c in all_numeric]
    feature_cols = feature_numeric_cols(df_clean)

    if verbose:
        print("\nColumn Type Summary:")
        print("-" * 35)
        print(f"String Variables:          {len(string_cols)}")
        print(f"Date Variables:            {len(date_cols)}")
        print(f"Metadata Variables:        {len(meta_cols)}")
        print(f"Feature Numeric Variables: {len(feature_cols)}")
        print("-" * 35)
        print(
            "Total Accounted:          "
            f" {len(string_cols) + len(date_cols) + len(meta_cols) + len(feature_cols)} / {df_clean.shape[1]}"
        )
    return df_clean


# --- 3. ENTRY POINT GUARD ---
if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    plot = "--plot" in sys.argv
    df = run_cleaning_pipeline(verbose=verbose)

    # --- 4. DATA SUMMARY & PLOTTING ---
    # Filter for features with meaningful non-zero proportion
    numeric_df = df[feature_numeric_cols(df, min_support_pct=0.01)]
    cv = (numeric_df.std() / numeric_df.mean().abs()).sort_values(ascending=False)

    # Call the modular plotter function
    if plot:
        plot_top_cv_distributions(df, cv, top_n=20)