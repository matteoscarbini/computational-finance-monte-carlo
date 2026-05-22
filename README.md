# Computational Finance – Monte Carlo Pricing
 
**Computational Finance Exam | University of Bologna – April 2023**
 
## Overview
This project implements Monte Carlo simulation methods to price two families of financial instruments: Asset Backed Securities (ABS) and path-dependent options. A custom Python library (`library.py`) is developed to support trajectory generation and pricing across multiple stochastic models.
 
## Instruments Priced
 
### 1. Asset Backed Securities (Securitization)
- Simulation of 20 correlated underlying assets via multidimensional Black-Scholes
- Computation of default times and cumulative default rates
- Pricing of three tranches (Junior, Mezzanine, Senior) with varying attachment/detachment levels
- Convergence analysis across 16K to 1M simulations
### 2. Path-Dependent Options
Pricing of **Asian Options** and **Barrier Options** (Down-and-Out Call, Up-and-Out Put, Double Barrier) under four stochastic models:
- **Black-Scholes**
- **Heston** (stochastic volatility, with tower property optimisation)
- **Variance Gamma**
- **Jump Merton**
## Validation
All trajectory generators are validated via martingale tests. Monte Carlo errors are verified to halve as the number of simulations is quadrupled, confirming correct implementation. Analytical solutions are used as benchmarks where available (Black-Scholes barrier options).
 
## Structure
```
├── Ex1.ipynb        # ABS pricing notebook
├── Ex2.ipynb        # Path-dependent options notebook
├── library.py       # Custom pricing library (trajectory generators, MC engine)
├── Report_Computational_Scarbini.pdf   # Full project report with results and analysis
└── README.md
```
 
## Requirements
```
numpy
scipy
matplotlib
```
 
Install with:
```bash
pip install numpy scipy matplotlib
```
 
## How to Run
1. Clone the repository
2. Ensure `library.py` is in the same directory as the notebooks
3. Open `Ex1.ipynb` or `Ex2.ipynb` in Jupyter and run all cells
 
