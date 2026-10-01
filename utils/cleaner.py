import numpy as np
import pandas as pd

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # 1. Parse dates
    date_cols = [c for c in df_clean.columns if 'date' in c]
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    # 2. Nullify item price sentinel
    if 'purchase_avg_item_price' in df_clean.columns:
        df_clean['purchase_avg_item_price'] = df_clean['purchase_avg_item_price'].replace(999999.00, np.nan)

    # 3. Nullify total spent overflow sentinels and negative values
    if 'purchase_total_spent' in df_clean.columns:
        df_clean.loc[df_clean['purchase_total_spent'] >= 1e9, 'purchase_total_spent'] = np.nan
        df_clean.loc[df_clean['purchase_total_spent'] < 0, 'purchase_total_spent'] = np.nan

    return df_clean