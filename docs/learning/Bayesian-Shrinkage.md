# Controlling Small-Sample Noise with Bayesian Shrinkage

## Problem

Our dataset contains limited historical coverage, primarily from the 2023–2025 seasons. This results in small sample sizes for many driver–track combinations.

With few observations, extreme performances (outliers) can disproportionately influence feature values. For example, a single unusually strong or weak race at a specific circuit may cause large swings in predicted overtakes, qualifying performance, and race results.

Without correction, this leads to unstable features and overfitting.

We therefore require a method that allows us to evaluate track-specific performance while preventing small-sample noise from skewing the data.

---

## Method: Bayesian Shrinkage

We apply Bayesian shrinkage to stabilize track-level performance estimates by pulling extreme values toward a global baseline.

The shrunk estimate is computed as: shrunk_estimate = (sample_mean × n + prior_mean × k) / (n + k)


This formula combines observed data with prior information using a weighted average.


This approach is closely related to empirical Bayes methods and Stein’s shrinkage estimator.

These methods demonstrate that shrinking noisy estimates toward a shared mean often improves overall prediction accuracy, particularly in high-variance, low-sample settings.


---

## Parameter Definitions

- `sample_mean`  
  Observed average performance at a specific track  
  (e.g., Leclerc’s average position gain at Monaco: +5)

- `n`  
  Number of observations at that track  
  (e.g., 1 race)

- `prior_mean`  
  Baseline performance expectation  
  (e.g., driver’s global average across all tracks, often near 0)

- `k`  
  Prior strength (regularization constant)  
  Controls how strongly estimates are pulled toward the baseline

---

## Implementation

Shrinkage is computed directly in SQL and stored as a feature in the modeling dataframe.

This allows stabilized track-affinity estimates to be used consistently during both training and inference.

Implementation reference:

- Track-level feature engineering pipeline (SQL preprocessing stage)

---

## Conclusion

Bayesian shrinkage provides a principled solution to small-sample instability by balancing track-specific observations with global performance trends.

This approach:

- Reduces overfitting  
- Improves feature reliability  
- Preserves meaningful performance signals  
- Enhances model robustness  

As a result, the model produces more stable and interpretable predictions across circuits with limited historical data.
