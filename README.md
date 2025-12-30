# Alpha Factor Ranking & Risk-Constrained Optimization

## Project Overview
This project implements a systematic equity trading strategy that ranks stocks using ML-generated alpha signals and constructs a portfolio using a risk model. It explicitly compares linear (Lasso) vs. non-linear (Gradient Boosting) alpha models and handles portfolio optimization using the **Woodbury Matrix Identity** for computational efficiency.

## Key Features
* **Alpha Generation:** Benchmarked Lasso, Random Forest, and Gradient Boosting. [cite_start]**Gradient Boosting** achieved the lowest MSE (0.000270)[cite: 212].
* [cite_start]**Risk Model:** Decomposed risk into Factor Risk and Idiosyncratic Risk using a custom implementation of the **Woodbury Matrix Identity** to invert the covariance matrix $(XFX^T + D)$ efficiently[cite: 215].
* [cite_start]**Performance:** Achieved a **Sharpe Ratio of 1.70** in out-of-sample backtesting (Test Period: 227 days)[cite: 296].

## Technical Implementation
* **Language:** Python (Pandas, NumPy, Scikit-Learn, SciPy)
* **Math:** Implemented `woodbury_inverse` to solve $(XFX^T + D)^{-1}$ for high-dimensional covariance inversion.
* **Optimization:** Mean-Variance optimization with strict position limits and gross exposure constraints.

## Results
* **Sharpe Ratio:** 1.70
* [cite_start]**Cumulative Return:** ~2.36% (Gross, Unlevered) [cite: 296]
* **Risk Decomposition:** Maintained idiosyncratic risk dominance >50% to ensure alpha was not purely beta-driven.

## Project Structure
```text
alpha-factor-risk-model/
├── data/                   # Contains processed pickle/bz2 data files (GitIgnored)
├── notebooks/
│   └── analysis.ipynb      # Main research notebook containing the backtest logic
├── src/
│   ├── risk_model.py       # Helper functions for matrix inversion and risk calc
│   └── factors.py          # Factor exposure and universe selection logic
├── .gitignore              # Ensures data files are not committed
├── README.md               # Project documentation
└── requirements.txt        # Dependencies (pandas, numpy, scikit-learn, patsy)