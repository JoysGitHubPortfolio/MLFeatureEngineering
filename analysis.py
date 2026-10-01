import cleaner as cl
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
file_path = "data/input/personas_monthly_sample.parquet"
df = pd.read_parquet(file_path)
df = cl.clean_df(df)

# 2. Convert Date Columns explicitly
date_column_names = [c for c in df.columns if 'date' in c]
for col in date_column_names:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# 3. Categorize Dtypes accurately
string_cols = df.select_dtypes(include=['string', 'object']).columns
date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
all_numeric = df.select_dtypes(include=['number']).columns
meta_cols = [c for c in ['year', 'month'] if c in all_numeric]
feature_numeric_cols = all_numeric.difference(meta_cols)

# Print concise row-wise summary
print(f"Data Shape: {df.shape}")
print("-" * 35)
print(f"String Variables:          {len(string_cols)}")
print(f"Date Variables:            {len(date_cols)}")
print(f"Metadata Variables:        {len(meta_cols)}")
print(f"Feature Numeric Variables: {len(feature_numeric_cols)}")
print("-" * 35)
print(f"Total Accounted:           {len(string_cols) + len(date_cols) + len(meta_cols) + len(feature_numeric_cols)} / {df.shape[1]}")

# 4. Filter for features with meaningful non-zero proportion of total data
min_support_pct = 0.01  # 5% minimum support
non_zero_rates = (df[all_numeric] > 0).mean()
valid_numeric_cols = non_zero_rates[non_zero_rates >= min_support_pct].index
print(f"\nFiltered out {len(all_numeric) - len(valid_numeric_cols)} zero-inflated features (<{min_support_pct*100}% non-zero support).")

# 5. Compute CV only on non-zero supported numeric features
top_n = 20
numeric_df = df[valid_numeric_cols]
cv = (numeric_df.std() / numeric_df.mean().abs()).sort_values(ascending=False)
print(f"\nTop {top_n} Features by Relative Dispersion (CV):")
print(cv.head(top_n))

# 6. Plot conditional distributions (filtering out zeros for plotting)
for col in cv.head(top_n).index:
    try:        
        plt.figure(figsize=(8, 4))
        plt.hist(df[col], edgecolor='black', alpha=0.75, color='skyblue')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency (Log Scale)")
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error occurred while processing column {col}: {e}")
        continue