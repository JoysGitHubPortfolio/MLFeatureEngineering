import numpy as np


def cluster_bootstrap(fit, groups: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Re-run `fit` on `n_boot` resamples of whole clusters (people), drawn
    with replacement. `fit` takes an array of row indices and returns one
    number. Returns the 95% percentile interval and a two-sided p-value vs 0.
    Captures sampling noise only: it cannot correct measurement bias."""
    clusters = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(clusters), len(clusters))
        draws[b] = fit(np.concatenate([clusters[i] for i in picked]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return lo, hi, p
