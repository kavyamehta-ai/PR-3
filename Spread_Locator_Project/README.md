# Spread Locator: A Statistical Distribution Analysis Model

This project analyzes the supplied `spread_locator_dataset.xlsx` data, exported into CSV, to explore transaction statuses, counts, transaction amount distributions, normality, transformations, and probabilities.

## Files
- `data/spread_locator_dataset.csv` — the provided dataset converted to CSV.
- `src/analyze.py` — complete Python analysis and chart generation.
- `reports/` — analysis summary, JSON results, descriptive statistics, and charts (created by running the script).
- `THEORY_NOTES.md` — definitions for all 11 theory topics in the assignment.
- `requirements.txt` — Python dependencies.

## Run
Recommended: Python 3.9+.

```bash
python -m pip install -r requirements.txt
python src/analyze.py
```

Run these commands from the extracted `Spread_Locator_Project` folder. Results are written to `reports/`.

## Assignment tasks covered
1. Bernoulli model for Success/Fail transaction status.
2. Approximate Binomial fit for the `transaction_count` field.
3. Poisson fit to daily transaction counts, including dates with zero recorded transactions within the observed date range.
4. Log-normal and Pareto (power-law-style) fits for positive transaction amounts.
5. Q-Q plots for raw and log-transformed transaction amounts, with Shapiro-Wilk normality diagnostics.
6. Box-Cox transformation.
7. Z-score at ₹5000, Normal-approximation probability, and empirical proportion above ₹5000.
8. PDF and CDF plots, plus descriptive summaries.

## Important assumptions and limitations
- The Binomial fit is illustrative because the dataset does not specify a fixed number of possible weekly transaction opportunities.
- The Pareto fit is a simple power-law-style comparison, not a full specialist power-law inference procedure.
- A Normal approximation for transaction amounts may be inaccurate if the data are skewed. Compare it with the empirical proportion over ₹5000.
- A fitted distribution or p-value does not prove that the model is true. Use plots and business context.
- Findings describe the supplied dataset and should not automatically be generalized to all customers.

## Suggested GitHub upload
Upload the complete folder to a repository with this README, source code, dataset, theory notes, and generated reports.
