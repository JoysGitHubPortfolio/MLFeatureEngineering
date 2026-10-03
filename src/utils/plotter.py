import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, RED, GREY, INK = "#2a78d6", "#eb6834", "#e34948", "#8a8985", "#52514e"
VERDICT_COLOURS = {"Supports YouTube": BLUE, "Cannibalises YouTube": RED}  # anything else is grey
STYLE = {
  "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
  "axes.titleweight": "bold", "font.size": 11, "axes.grid": True, "axes.axisbelow": True,
  "grid.color": "#e6e5e0", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
}
MONTHS = ["Jun 25", "Jul 25", "Aug 25", "Sep 25", "Oct 25", "Nov 25",
          "Dec 25", "Jan 26", "Feb 26", "Mar 26", "Apr 26", "May 26"]


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


def _bar_labels(ax, values) -> None:
  for i, v in enumerate(values):
    ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=16, fontweight="bold")


def plot_data_audit(df: pd.DataFrame, ts: pd.DataFrame, platforms: dict[str, str], path) -> None:
  """Four panels on what the time-spend data can support: who is in it, the
  iOS covered_days bug, what the meter sees per OS, and heavy-user concentration.
  `df` is the whole panel, `ts` the rows with time-spend data."""
  with plt.rc_context(STYLE):
    fig, ((coverage, exposure), (capture, heavy)) = plt.subplots(2, 2, figsize=(14, 10))
    android = ts[ts["identity_device_platform"] == "android"]
    ios = ts[ts["identity_device_platform"] == "ios"]

    # 1. Who has time-spend data, month by month
    coverage.plot(MONTHS, df.groupby("period")["user_id"].nunique().values, color=BLUE, marker="o", lw=2.5)
    coverage.plot(MONTHS, ts.groupby("period")["user_id"].nunique().values, color=ORANGE, marker="o", lw=2.5)
    coverage.text(0, 640, "Everyone in the panel", color=BLUE, fontweight="bold")
    coverage.text(8, 120, "Has time-spend data", color=ORANGE, fontweight="bold", ha="center")
    coverage.set(title="1. Only half the panel has time-spend data,\nand who is in it changes month to month",
                 ylabel="People", ylim=(0, 700))
    coverage.tick_params(axis="x", rotation=45)

    # 2. Months claiming more metered days than they have
    impossible = [(android["exposure_ratio"] > 1.05).mean() * 100, (ios["exposure_ratio"] > 1.05).mean() * 100]
    exposure.bar(["Android", "iPhone"], impossible, color=[BLUE, ORANGE], width=0.5)
    _bar_labels(exposure, impossible)
    exposure.set(title="2. Most iPhone months claim more metered days than the\n"
                       "month has (usually 7×), so 'per day' columns are wrong",
                 ylabel="% of measured months", ylim=(0, 100))

    # 3. What the meter can see on each OS
    y = np.arange(len(platforms))
    capture.barh(y + 0.2, [(android[f"time_spent_{p}"] > 0).mean() * 100 for p in platforms],
                 height=0.4, color=BLUE, label="Android")
    capture.barh(y - 0.2, [(ios[f"time_spent_{p}"] > 0).mean() * 100 for p in platforms],
                 height=0.4, color=ORANGE, label="iPhone")
    capture.set_yticks(y, platforms.values())
    capture.set(title="3. The meter barely sees Amazon, ChatGPT and\nGoogle Search on iPhone", xlim=(0, 100),
                xlabel="% of measured months with any time on the platform")
    capture.legend(loc="upper left", bbox_to_anchor=(0.55, 0.8), frameon=False)

    # 4. How concentrated YouTube time is
    yt = np.sort(ts.groupby("user_id")["time_spent_youtube"].sum().to_numpy())[::-1]
    share = [yt[: len(yt) * k // 100].sum() / yt.sum() * 100 for k in (1, 10, 50)]
    heavy.bar(["Heaviest 1%", "Heaviest 10%", "Heaviest 50%"], share, color=BLUE, width=0.5)
    _bar_labels(heavy, share)
    heavy.set(title=f"4. A few heavy viewers hold most YouTube time\n"
                    f"(share of all metered YouTube time, {len(yt)} people)",
              ylabel="% of all metered YouTube time", ylim=(0, 112))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cannibalisation(res: pd.DataFrame, platforms: dict[str, str], path) -> None:
  """One row per platform: the within-person effect on YouTube time as a %
  change for a 10% rise in platform time, with its 95% interval and verdict.
  `res` needs columns beta, lo, hi, n_people and verdict."""
  res = res.sort_values("beta")
  pct = lambda b: (1.1 ** b - 1) * 100  # elasticity -> % change in YouTube time
  with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axvspan(-4, 0, color=RED, alpha=0.06)
    ax.axvspan(0, 4, color=BLUE, alpha=0.06)
    ax.text(-3.9, 7.6, "← Takes time from YouTube", color=RED, fontweight="bold")
    ax.text(3.9, 7.6, "Goes with YouTube →", color=BLUE, fontweight="bold", ha="right")

    for i, (_, r) in enumerate(res.iterrows()):
      col = VERDICT_COLOURS.get(r["verdict"], GREY)
      ax.plot([pct(r["lo"]), pct(r["hi"])], [i, i], color=col, lw=3, solid_capstyle="round")
      ax.plot(pct(r["beta"]), i, "o", ms=11, color=col, mec="white", mew=2)
      ax.text(4.2, i, f"{r['verdict']}  ({r['n_people']} people)", va="center",
              color=INK if col == GREY else col, fontweight="bold")

    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(res)), [platforms[p] for p in res.index])
    ax.set(xlim=(-4, 4), ylim=(-0.6, 7.9),
           xlabel="Change in a person's YouTube time when their time on the platform rises 10%  (%)")
    ax.set_title("No platform measurably takes time away from YouTube\n", fontsize=14)
    ax.text(0, 1.02, "Same people followed month to month, net of their other screen time. "
            "Dot = estimate, bar = 95% bootstrap interval.", transform=ax.transAxes, color=INK)
    fig.text(0.01, 0.01, "Verdict: significant after correcting for 8 tests, and holds on Android, iPhone, "
             "without heavy users, on the stable panel and on clean-exposure rows.", color=GREY, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 0.8, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
