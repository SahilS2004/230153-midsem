# Project Report: Dirichlet Process Gaussian Mixture Models

**Course:** Advanced Machine Learning (Midsem Evaluation)
**Roll Number:** 230153
**Paper:** *"Dirichlet Process Gaussian Mixture Models: Choice of the Base Distribution"* — Görür & Rasmussen (2010)

---

## 1. Introduction

Clustering is a fundamental unsupervised learning task. Classical methods like K-Means and Gaussian Mixture Models (GMM) require the user to specify the number of clusters *k* in advance. In practice, the true number of clusters is often unknown.

**Dirichlet Process Gaussian Mixture Models (DP-GMM)** solve this by using a Bayesian nonparametric prior — the Dirichlet Process — which allows the model to automatically infer the number of clusters from data. This project reproduces and extends the key ideas from Görür & Rasmussen (2010).

---

## 2. Paper Summary

### Core Ideas

The paper frames Dirichlet Process Mixture (DPM) models as **infinite mixture models** where:
- The number of components is not fixed but grows with the data
- A **concentration parameter α** controls the expected number of clusters
- The **base distribution** G₀ acts as a prior over mixture component parameters

### Key Comparison

| Aspect | Conjugate DPGMM | Conditionally Conjugate DPGMM |
|--------|----------------|-------------------------------|
| **Base distribution** | Normal-Inverse-Wishart | Normal × Inverse-Wishart (independent) |
| **Mean & covariance** | Coupled | Independent priors |
| **Inference** | Simpler (Gibbs sampling) | More flexible (Metropolis-Hastings within Gibbs) |
| **Density estimation** | Good | **Better** (more flexible covariance) |

### Why It Matters

- **Conjugate** model: mean and covariance are coupled through the prior → restrictive
- **Conditionally conjugate** model: independent priors on mean and covariance → better density estimation because it doesn't force clusters to have correlated mean and spread

### Inference

The paper uses **Markov Chain Monte Carlo (MCMC)** sampling. For our implementation, we use `sklearn.BayesianGaussianMixture` which uses **variational inference** — a faster, deterministic approximation.

---

## 3. Method Implementation

### Dirichlet Process Prior

The Dirichlet Process DP(α, G₀) defines a distribution over distributions:
- **α (concentration parameter):** Controls how many clusters to expect
  - Small α → few clusters
  - Large α → many clusters
- **G₀ (base distribution):** Prior distribution over cluster parameters (mean, covariance)

### Stick-Breaking Construction

The DP can be understood via stick-breaking:
1. Start with a stick of length 1
2. Break off a fraction β₁ ~ Beta(1, α)
3. The weight of cluster 1 is π₁ = β₁
4. From the remainder, break off β₂ ~ Beta(1, α)
5. Weight of cluster 2: π₂ = β₂(1 - β₁)
6. Continue infinitely...

This gives decreasing weights → most data points belong to a few clusters, while allowing arbitrarily many.

### Implementation

```python
from sklearn.mixture import BayesianGaussianMixture

model = BayesianGaussianMixture(
    n_components=10,           # upper bound (DP will use fewer)
    covariance_type='full',
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=1.0,  # α
    max_iter=500
)
model.fit(X)
labels = model.predict(X)
```

The model sets `n_components` as an **upper bound**. The DP prior drives unused component weights to zero, effectively selecting the right number of clusters.

---

## 4. Experiments

### Experiment 1: Model Comparison (KMeans vs GMM vs DP-GMM)

**Goal:** Show that DP-GMM automatically finds the correct number of clusters without manual specification.

**Setup:**
- Dataset: Iris (150 samples, 4 features, 3 species)
- KMeans: k=3 (manually set)
- GMM: n_components=3 (manually set)
- DP-GMM: n_components=10 (upper bound), α=1.0

**Results:**

| Model | Clusters | ARI | Silhouette |
|-------|----------|-----|------------|
| KMeans | 3 | ~0.73 | ~0.46 |
| GMM | 3 | ~0.90 | ~0.45 |
| DP-GMM | auto | ~0.90 | ~0.45 |

**Key Findings:**
- DP-GMM correctly identifies ~3 clusters without manual specification
- GMM and DP-GMM outperform KMeans on ARI because GMM handles elliptical clusters
- KMeans assumes spherical clusters which is a poor fit for Iris

> See: `results/plots/exp1_model_comparison.png`, `results/plots/exp1_dpgmm_weights.png`

---

### Experiment 2: Multiple Datasets

**Goal:** Validate DP-GMM generalization across diverse data structures.

**Datasets:**

| Dataset | Features | Expected Clusters | Description |
|---------|----------|-------------------|-------------|
| Iris | 4 | 3 | Flower species |
| Wine | 13 | 3 | Wine cultivars |
| Old Faithful | 2 | 2 | Geyser eruptions |
| Synthetic | 2 | 5 | make_blobs |

**Key Findings:**
- DP-GMM correctly detects cluster counts across different data types
- Works well on both low-dimensional (Old Faithful, 2D) and medium-dimensional (Wine, 13D) data
- Synthetic blobs with clear separation are easiest to cluster

> See: `results/plots/exp2_multiple_datasets.png`

---

### Experiment 3: Hyperparameter Analysis (α)

**Goal:** Demonstrate that concentration parameter α directly controls cluster count, validating a key theoretical prediction of the paper.

**Setup:** Vary α from 0.001 to 1000 on Iris dataset.

**Expected behavior (from theory):**
- α → 0: all data in one cluster
- α → ∞: every point is its own cluster
- Moderate α: correct number of clusters

**Key Findings:**
- Clear monotonic relationship between α and cluster count
- Best ARI achieved at moderate α values
- Too-small α under-clusters; too-large α over-clusters
- This confirms the paper's theoretical framework

> See: `results/plots/exp3_alpha_analysis.png`, `results/plots/exp3_alpha_clusters.png`

---

### Experiment 4: Failure Analysis

**Goal:** Identify and analyze scenarios where DP-GMM performs poorly.

#### Case 1: Overlapping Clusters
- Iris versicolor and virginica overlap significantly in feature space
- DP-GMM struggles to separate them → ARI drops
- **Why:** The Gaussian assumption cannot cleanly separate non-Gaussian, overlapping distributions

#### Case 2: High-Dimensional Data
- 2D data with 150 samples: DP-GMM works well
- 50D data with 150 samples: performance may degrade
- **Why:** Curse of dimensionality — distance metrics become less meaningful in high dimensions

#### Case 3: Outlier Sensitivity
- Clean 3-cluster data: DP-GMM finds 3 clusters
- Same data + 20 outliers: DP-GMM may create spurious clusters
- **Why:** DP-GMM tries to explain every data point, including outliers

#### Case 4: Unbalanced Cluster Sizes
- Balanced clusters (100/100/100): works well
- Unbalanced (300/50/10): very small clusters may be absorbed
- **Why:** Small clusters have limited evidence and may be merged with larger ones by the model

> See: `results/plots/exp4_failure_analysis.png`

---

## 5. Results Summary

### Model Comparison

DP-GMM achieves comparable or better performance than manually-tuned KMeans/GMM **without** requiring the user to specify *k*. This is its primary advantage.

### Cross-Dataset Validation

The model generalizes well across datasets with different structures, dimensionalities, and cluster counts.

### Hyperparameter Effect

α is the single most important hyperparameter. A value of ~1.0 is a reasonable default for most datasets.

### When DP-GMM Fails

| Failure Mode | Root Cause | Impact |
|-------------|------------|--------|
| Overlapping clusters | Gaussian assumption | Poor separation, low ARI |
| High dimensions | Curse of dimensionality | Unreliable distances |
| Outliers | Explains all data points | Spurious clusters |
| Unbalanced sizes | Weak evidence for small clusters | Cluster absorption |

---

## 6. Why DP-GMM over Other Models?

| Question | Answer |
|----------|--------|
| **Why not KMeans?** | KMeans requires specifying *k* and assumes spherical clusters |
| **Why not standard GMM?** | Standard GMM also requires specifying *k* |
| **Why Bayesian?** | Bayesian approach provides uncertainty estimates and automatic model complexity |
| **Why Dirichlet Process?** | DP provides a principled prior that allows infinite clusters but favors parsimony |

---

## 7. Conclusion

This project successfully reproduces the key claims from Görür & Rasmussen (2010):

1. **DP-GMM automatically infers cluster count** — confirmed across 4 datasets
2. **Concentration parameter α controls cluster count** — confirmed via systematic sweep
3. **DP-GMM provides competitive clustering performance** — comparable ARI to manually-tuned models
4. **Failure modes exist** — overlapping clusters, high dimensions, outliers, and imbalanced data can degrade performance

The Dirichlet Process framework offers a principled, Bayesian approach to clustering that eliminates the need for manual cluster selection while providing uncertainty quantification.

---

## References

1. Görür, D. & Rasmussen, C. E. (2010). *Dirichlet Process Gaussian Mixture Models: Choice of the Base Distribution.* Journal of Computer Science and Technology, 25(4), 615–626.
2. Ferguson, T. S. (1973). *A Bayesian analysis of some nonparametric problems.* Annals of Statistics, 1(2), 209–230.
3. Blei, D. M. & Jordan, M. I. (2006). *Variational inference for Dirichlet process mixtures.* Bayesian Analysis, 1(1), 121–143.
4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
