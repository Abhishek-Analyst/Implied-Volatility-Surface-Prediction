# NIFTY Implied Volatility Surface — Missing-Value Reconstruction

Solution for the **Finance Club, IIT Roorkee — Open Projects 2026** challenge: predicting missing implied-volatility (IV) values across strikes and timestamps in the NIFTY options chain.

## Problem

The data is a matrix of **timestamps × strikes** of implied volatilities with ~20% of entries missing. The task is to estimate the missing values as accurately as possible, judged on **mean squared error (MSE)** against the hidden ground truth.

## Approach

The IV surface is treated as a **low-rank matrix-completion** problem, guided by options theory.

- A volatility surface is driven by a few latent factors (level, skew, smile curvature), so it is **approximately rank-3** — confirmed in the EDA (the first 3 singular values capture ~99% of the call-surface energy).
- **Almost all the difficulty is concentrated on the expiry day** (27-Jan-2026), where IV explodes as time-to-expiry `T → 0`.

**Pipeline**

1. Separate **calls (CE)** and **puts (PE)** — they sit on different parts of the smile.
2. **Non-expiry days:** iterative rank-3 SVD completion in **log-IV** space.
3. **Expiry day:** iterative rank-3 SVD completion in **IV × √T** space (variance scales with time), then invert the transform. `T` = minutes to the 15:30 expiry.
4. Assemble the submission CSV.

The notebook also includes EDA on the **volatility smile** and **moneyness** structure, and a **masked hold-out validation** that uses only observed cells (no leakage of hidden targets, no lookahead on targets).

## Files

| File | Description |
|------|-------------|
| `best_solution.ipynb` | Full, documented, reproducible notebook (EDA → model → validation → submission) |
| `submission.csv` | Predicted IV values in the required `id,value` format |
| `dataset.csv` | Input data (provided by the competition) |

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook best_solution.ipynb   # run all cells top to bottom
