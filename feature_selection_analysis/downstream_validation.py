"""
Downstream Validation: k-NN LOO + Randomized Baseline (empirical p-value)
===========================================================================
Reusable module to evaluate whether a feature-selection method's Top-K
ranking is genuinely predictive of a real target, without requiring
knowledge of a ground truth ranking.

Method (precedent: Wild et al. 2024/2025, Nature Communications, Fig. 2b-c):
  1. For a given method's Top-K features, train a k-NN classifier with
     Leave-One-Out (LOO) cross-validation, predicting the real target y.
  2. Compare against a baseline: N_RANDOM random draws of K features
     (same K), each evaluated the same way (k-NN LOO).
  3. Report:
       - method accuracy
       - baseline mean / std accuracy
       - empirical p-value = fraction of random draws with accuracy >=
         the method's accuracy (one-sided; lower is more significant)

Explicitly NOT claiming ground-truth validation: this is a predictive-
usefulness proxy, not a causal/ground-truth check. On the synthetic
dataset (where ground truth IS known), this module's ranking-by-accuracy
should be cross-checked against ranking-by-tau as a sanity check before
trusting the same proxy on real data without ground truth.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


def knn_loo_accuracy(X, y, feature_idx, k_neighbors=5):
    """
    Leave-one-out accuracy of a k-NN classifier using only the features
    in feature_idx (list/array of column indices into X).

    Implementation note: LOO accuracy for k-NN is computed via a single
    pairwise-distance matrix (self excluded) rather than by explicitly
    refitting a classifier once per held-out point. "Who are point i's k
    nearest OTHER points, and what's their majority label?" is exactly
    what LOO asks, so this is mathematically equivalent to the naive
    fit-per-fold loop, but O(n^2) once instead of ~n classifier
    fits/predicts — the difference between this finishing in seconds and
    potentially not finishing in minutes/hours, given this function is
    called thousands of times (per K x per random baseline draw x per
    method) in evaluate_method_vs_baseline below.

    Trade-off made explicit: X is standardised ONCE on the full sample
    (not per LOO fold), which leaks each held-out point's own value into
    the mean/std used to scale it. With n in the thousands (as in this
    project), the effect on the mean/std of removing a single point is
    O(1/n) and negligible in practice; refitting per fold would be more
    "pure" but reintroduces the performance problem above. This trade-off
    should be named explicitly if asked, not left implicit.

    X : array, shape (n_samples, n_features) — should already be numeric,
        NaNs handled by caller.
    y : array, shape (n_samples,) — classification target (integer/bool
        labels).
    feature_idx : indices of columns to use.
    k_neighbors : number of neighbours for k-NN (default 5, consistent
        with K_CMI used elsewhere in this project). Use an odd number
        for binary targets to avoid tie votes.
    """
    X_sub = X[:, feature_idx]
    n = len(y)
    if k_neighbors >= n:
        raise ValueError(f"k_neighbors ({k_neighbors}) must be < n_samples ({n})")

    scaler = StandardScaler()
    X_sub = scaler.fit_transform(X_sub)

    dist = squareform(pdist(X_sub))
    np.fill_diagonal(dist, np.inf)  # exclude self as a neighbour

    # indices of the k nearest OTHER points for every row (unsorted within
    # the k-set, which is fine — only the set matters for majority vote)
    nn_idx = np.argpartition(dist, kth=k_neighbors - 1, axis=1)[:, :k_neighbors]

    preds = np.empty(n, dtype=y.dtype)
    for i in range(n):
        neighbour_labels = y[nn_idx[i]]
        vals, counts = np.unique(neighbour_labels, return_counts=True)
        preds[i] = vals[np.argmax(counts)]  # majority vote

    # Balanced accuracy (average of per-class recall) instead of plain
    # accuracy: with imbalanced targets (e.g. a rare complication at ~3%
    # prevalence), plain accuracy is dominated by the majority class and
    # both the method and the random baseline cluster near (1 - prevalence),
    # making genuine differences hard to see. Balanced accuracy fixes this
    # without changing anything for already-balanced targets (aggregate MI
    # complications, FRED-MD, synthetic median split all ~50/50), so this
    # is a strict improvement, not a trade-off.
    return float(balanced_accuracy_score(y, preds))


def evaluate_method_vs_baseline(X, y, method_ranks, k_values,
                                 n_random=200, k_neighbors=5, seed=42,
                                 method_name="method"):
    """
    For each K in k_values: evaluate the method's Top-K features via
    knn_loo_accuracy, then compare against n_random random K-feature
    draws (baseline), returning accuracy, baseline mean/std, and an
    empirical one-sided p-value.

    method_ranks : array, shape (n_features,) — rank per feature
        (1 = most important), aligned with X's columns.

    Returns: list of dicts, one per K.
    """
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    results = []

    for K in k_values:
        if K > n_features:
            print(f"  [{method_name:<12} K={K:2d}]  SKIPPED "
                  f"(K > n_features={n_features})")
            results.append(dict(method=method_name, K=K, method_acc=np.nan,
                                 baseline_mean=np.nan, baseline_std=np.nan,
                                 p_value=np.nan))
            continue
        top_k_idx = np.argsort(method_ranks)[:K]  # lowest rank = best
        method_acc = knn_loo_accuracy(X, y, top_k_idx, k_neighbors)

        baseline_accs = np.empty(n_random)
        for i in range(n_random):
            rand_idx = rng.choice(n_features, size=K, replace=False)
            baseline_accs[i] = knn_loo_accuracy(X, y, rand_idx, k_neighbors)

        # one-sided empirical p-value: fraction of random draws with
        # accuracy >= method's accuracy (method is "significant" if this
        # is small, e.g. < 0.05)
        p_value = float(np.mean(baseline_accs >= method_acc))

        results.append(dict(
            method=method_name, K=K,
            method_acc=method_acc,
            baseline_mean=float(baseline_accs.mean()),
            baseline_std=float(baseline_accs.std()),
            p_value=p_value,
        ))
        print(f"  [{method_name:<12} K={K:2d}]  acc={method_acc:.3f}  "
              f"baseline={baseline_accs.mean():.3f}±{baseline_accs.std():.3f}  "
              f"p={p_value:.3f}")

    return results


def rf_cv_accuracy(X, y, feature_idx, n_estimators=50, cv=5, seed=42):
    """
    Stratified k-fold CV accuracy of a Random Forest using only the
    features in feature_idx. Used as a second classifier alongside k-NN
    LOO (knn_loo_accuracy) to check that a downstream-validation result
    isn't an artifact of k-NN's distance metric specifically — relevant
    when features are a mix of binary/categorical/continuous, where
    Euclidean-distance-based k-NN is known to be a weaker fit than a
    tree-based model.

    Uses k-fold CV (not LOO) for computational reasons: refitting a
    Random Forest N times (LOO) is far more expensive than refitting
    k-NN N times, so this function trades LOO's slightly lower variance
    for CV's large speedup — acceptable here since this is a secondary,
    confirmatory check, not the primary evidence.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    X_sub = X[:, feature_idx]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X_sub, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_sub[train_idx])
        X_test = scaler.transform(X_sub[test_idx])
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                      random_state=seed, n_jobs=-1)
        clf.fit(X_train, y[train_idx])
        preds_fold = clf.predict(X_test)
        accs.append(balanced_accuracy_score(y[test_idx], preds_fold))
    return float(np.mean(accs))


def evaluate_method_vs_baseline_rf(X, y, method_ranks, k_values,
                                    n_random=50, n_estimators=50, cv=5,
                                    seed=42, method_name="method"):
    """
    Same logic as evaluate_method_vs_baseline, but using Random Forest +
    stratified k-fold CV (rf_cv_accuracy) instead of k-NN + LOO — a
    secondary check to rule out that a null/positive downstream result is
    specific to k-NN's distance-based mechanism. n_random defaults lower
    (50, not 200) because RF is much more expensive per call than k-NN;
    this trades some precision on the baseline distribution for staying
    within a tight time budget, and should be reported as such (fewer
    random draws -> coarser p-value resolution, minimum achievable
    p-value is 1/n_random = 0.02 here, not 0.005 as with n_random=200).
    """
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    results = []

    for K in k_values:
        if K > n_features:
            print(f"  [{method_name:<12} K={K:2d}]  SKIPPED "
                  f"(K > n_features={n_features})")
            results.append(dict(method=method_name, K=K, method_acc=np.nan,
                                 baseline_mean=np.nan, baseline_std=np.nan,
                                 p_value=np.nan))
            continue
        top_k_idx = np.argsort(method_ranks)[:K]
        method_acc = rf_cv_accuracy(X, y, top_k_idx, n_estimators, cv, seed)

        baseline_accs = np.empty(n_random)
        for i in range(n_random):
            rand_idx = rng.choice(n_features, size=K, replace=False)
            baseline_accs[i] = rf_cv_accuracy(X, y, rand_idx, n_estimators, cv, seed)

        p_value = float(np.mean(baseline_accs >= method_acc))

        results.append(dict(
            method=method_name, K=K,
            method_acc=method_acc,
            baseline_mean=float(baseline_accs.mean()),
            baseline_std=float(baseline_accs.std()),
            p_value=p_value,
        ))
        print(f"  [RF {method_name:<12} K={K:2d}]  acc={method_acc:.3f}  "
              f"baseline={baseline_accs.mean():.3f}±{baseline_accs.std():.3f}  "
              f"p={p_value:.3f}")

    return results