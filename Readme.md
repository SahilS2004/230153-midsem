# 230153-midsem — Dirichlet Process Gaussian Mixture Models

## Paper Reference

**"Dirichlet Process Gaussian Mixture Models: Choice of the Base Distribution"**
*Dilan Görür and Carl Edward Rasmussen, 2010*
Journal of Computer Science and Technology, 25(4), 615–626.

## Overview

This project reproduces and extends the key ideas from the above paper for the **Advanced Machine Learning** midsem evaluation. The paper shows how Gaussian Mixture Models can be extended to an infinite number of clusters using the Dirichlet Process, and compares conjugate vs conditionally conjugate base distributions.

We implement simplified experiments using `sklearn.mixture.BayesianGaussianMixture` (a variational inference–based DP-GMM) to demonstrate:

1. **DP-GMM vs classical models** — automatic cluster inference vs fixed-K methods
2. **Generalization across datasets** — Iris, Wine, Old Faithful Geyser, Synthetic
3. **Hyperparameter sensitivity** — effect of concentration parameter α on cluster count
4. **Failure analysis** — overlapping clusters, high dimensions, outliers, imbalanced data

## Project Structure

```
230153-midsem/
├── README.md
├── requirements.txt
├── GoeRas10.pdf                          # Original paper
├── dataset/
│   └── old_faithful.csv                  # Old Faithful geyser data
├── experiments/
│   ├── experiment1_model_comparison.py   # KMeans vs GMM vs DP-GMM
│   ├── experiment2_multiple_datasets.py  # Cross-dataset evaluation
│   ├── experiment3_hyperparameter_alpha.py  # α sensitivity
│   └── experiment4_failure_analysis.py   # Edge cases & failures
├── results/
│   └── plots/                            # Auto-generated visualizations
└── report/
    └── report.md                         # Full project report
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python experiments/experiment1_model_comparison.py
python experiments/experiment2_multiple_datasets.py
python experiments/experiment3_hyperparameter_alpha.py
python experiments/experiment4_failure_analysis.py
```

All plots are saved to `results/plots/`. Each script prints comparison tables to stdout.

## Key Findings

- DP-GMM automatically infers the correct number of clusters without manual specification
- Concentration parameter α directly controls the expected number of clusters
- DP-GMM outperforms KMeans on non-spherical, overlapping cluster structures
- Failure modes include heavily overlapping clusters and high-dimensional sparse data

## Author

**Roll Number:** 230153
