Write a Python function named clean_df that takes a pandas DataFrame and applies targeted data quality fixes without mutating the original input object.

During my initial exploratory data analysis on the 359-column personas dataset, I identified several critical ETL artifacts and pipeline sentinels that are artificially inflating statistical variance (CV) and distorting column means. Specifically, implement the following sanitization logic:

Data Type Casting: Isolate columns containing the substring 'date' and cast them from string/object types to datetime64.

Targeted Sentinel Removal: purchase_avg_item_price contains 9 rows with a hardcoded database sentinel of 999999.00. Target this column and replace that specific value with np.nan.

Overflow / Cap Handling: purchase_total_spent contains 11 rows capped at an integer overflow boundary of 1.000000e+10. Target this column and replace any values >= 1e9 with np.nan.

Negative Spend Extraction: purchase_total_spent contains negative transaction values (e.g., -250.75), likely representing refunds or chargebacks. Extract this signal into a new boolean indicator column (has_refund), and then replace the negative values in the original column with np.nan to protect downstream magnitude calculations.

Use .copy() to preserve the raw state and ensure pandas .loc is used for all conditional assignments to avoid SettingWithCopy warnings.