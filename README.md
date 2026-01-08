
---

##  Production Script (`main.py`)

The `main.py` file implements the full **data → signal → portfolio** pipeline:

1. Fetches historical prices from **Yahoo Finance**
2. Builds **industry-level indices** (equal-weight)
3. Computes:
   - Momentum (with skip)
   - Rolling Hurst exponent
4. Constructs a **long-only portfolio**
5. Outputs **daily industry weights** to an Excel file

### Output
- `industry_weights.xlsx`
  - rows: trading dates  
  - columns: industries  
  - values: portfolio weights  

Weights are held constant between rebalancing dates and extended until the most recent available date.

---

## Research Notebook (`Projet_Python.ipynb`)

The notebook contains:
- exploratory data analysis,
- backtests of momentum and Hurst×Momentum strategies,
- comparison with benchmarks,
- parameter analysis and robustness checks.

It serves as the **research and validation layer**, while `main.py` is designed for **production usage**.

---

## Data

- Price data is fetched dynamically using `yfinance`
- Industry and sector classifications are provided via `Bdd_Projet_Python.xlsx`
- No proprietary data is used


