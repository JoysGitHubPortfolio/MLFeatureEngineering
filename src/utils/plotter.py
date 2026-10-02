import matplotlib.pyplot as plt
import pandas as pd


def plot_top_cv_distributions(df: pd.DataFrame, 
                              cv_series: pd.Series, 
                              top_n: int = 20,
                              log_scale: bool = True) -> None:
  """Plots log-scale histograms for the top N features by Coefficient of Variation."""
  for col in cv_series.head(top_n).index:
    try:
      plt.figure(figsize=(8, 4))
      plt.hist(df[col], edgecolor="black", alpha=0.75, color="skyblue")
      plt.title(f"Distribution of {col}")
      plt.xlabel(col)
      plt.ylabel("Frequency (Log Scale)")
      if log_scale:
        plt.yscale("log")
      plt.grid(True, linestyle="--", alpha=0.5)
      plt.tight_layout()
      plt.show()
    except Exception as e:
      print(f"Error occurred while processing column {col}: {e}")
      continue