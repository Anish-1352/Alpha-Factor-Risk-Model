# Multi-Factor Alpha Ranking & Risk-Constrained Optimization

## Project Overview
This project implements a systematic equity trading strategy that ranks stocks using ML-generated alpha signals and constructs a portfolio using a risk model. It explicitly compares linear (Lasso) vs. non-linear (Gradient Boosting) alpha models and handles portfolio optimization using the **Woodbury Matrix Identity** for computational efficiency.

## Key Features
* **Alpha Generation:** Benchmarked Lasso, Random Forest, and Gradient Boosting. [cite_start]**Gradient Boosting** achieved the lowest MSE (0.000270)[cite: 212].
* [cite_start]**Risk Model:** Decomposed risk into Factor Risk and Idiosyncratic Risk using a custom implementation of the **Woodbury Matrix Identity** to invert the covariance matrix $(XFX^T + D)$ efficiently[cite: 215].
* [cite_start]**Performance:** Achieved a **Sharpe Ratio of 1.70** in out-of-sample backtesting (2004-2006 period)[cite: 296].

## Technical Implementation
* **Language:** Python (Pandas, NumPy, Scikit-Learn, SciPy)
* **Math:** Implemented `woodbury_inverse` to solve $(XFX^T + D)^{-1}$ for high-dimensional covariance inversion.
* **Optimization:** Mean-Variance optimization with strict position limits and gross exposure constraints.

## Results
* **Sharpe Ratio:** 1.70
* [cite_start]**Avg Daily PnL:** ~2.36% (Gross) [cite: 296]
* **Risk Decomposition:** Maintained idiosyncratic risk dominance >50% to ensure alpha was not purely beta-driven.

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run the analysis: `jupyter notebook notebooks/analysis.ipynb`