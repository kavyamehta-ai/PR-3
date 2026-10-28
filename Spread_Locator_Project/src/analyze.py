"""Spread Locator: Statistical Distribution Analysis Model.
Run from project root: python src/analyze.py
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import ndtr
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "spread_locator_dataset.csv"
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA, parse_dates=["transaction_date"])
df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
df["transaction_count"] = pd.to_numeric(df["transaction_count"], errors="coerce")
df = df.dropna(subset=["transaction_amount", "transaction_date"])
amounts = df["transaction_amount"].astype(float)
positive = amounts[amounts > 0]
alpha = 0.05

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()

def clean_num(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    return x

results = {
    "dataset": {
        "rows_used": int(len(df)),
        "columns": list(df.columns),
        "amount_min": float(amounts.min()),
        "amount_max": float(amounts.max()),
        "amount_mean": float(amounts.mean()),
        "amount_median": float(amounts.median()),
        "amount_std": float(amounts.std()),
        "amount_skewness": float(stats.skew(amounts, nan_policy="omit")),
        "note": "Analysis uses the supplied dataset. Interpretations are descriptive; the sample may not represent all customers."
    },
    "distribution_fits": {},
    "tests_and_probabilities": {}
}

# Bernoulli model: success/fail transaction status
status = df["transaction_status"].astype(str).str.strip().str.lower()
success = status.isin(["success", "successful", "1", "true"])
p_success = float(success.mean())
results["distribution_fits"]["bernoulli_status"] = {
    "success_count": int(success.sum()), "failure_count": int((~success).sum()),
    "estimated_p_success": p_success, "estimated_p_failure": 1-p_success,
    "interpretation": f"Estimated success probability is {p_success:.3f} based on observed transaction statuses."
}
plt.bar(["Success", "Fail"], [success.sum(), (~success).sum()])
plt.title("Transaction status counts")
plt.xlabel("Status"); plt.ylabel("Number of transactions")
savefig("bernoulli_status.png")

# Binomial fit: transaction_count distribution as count out of a weekly opportunity count.
# The assignment gives transaction_count but no fixed number of weekly opportunities.
# Estimate n as the observed maximum count and p from the sample mean/n; clearly label approximation.
counts = df["transaction_count"].dropna()
count_int = counts[counts >= 0].round().astype(int)
n_trials = max(1, int(count_int.max()))
p_binom = float(np.clip(count_int.mean()/n_trials, 0, 1))
binom_x = np.arange(0, n_trials+1)
observed_counts = count_int.value_counts().reindex(binom_x, fill_value=0)
binom_probs = stats.binom.pmf(binom_x, n_trials, p_binom)
results["distribution_fits"]["binomial_transaction_count"] = {
    "assumed_n_trials": n_trials, "estimated_p": p_binom,
    "sample_mean_count": float(count_int.mean()),
    "note": "Approximate illustrative fit: the dataset does not specify a fixed number of possible weekly transaction opportunities."
}
plt.bar(binom_x, observed_counts.values / max(1, len(count_int)), label="Observed proportion", alpha=0.7)
plt.plot(binom_x, binom_probs, "o-", label="Fitted Binomial PMF")
plt.title("Transaction count: observed vs approximate Binomial fit")
plt.xlabel("Transaction count"); plt.ylabel("Probability / proportion"); plt.legend()
savefig("binomial_transaction_count.png")

# Poisson: number of records per calendar day
daily = df.groupby(df["transaction_date"].dt.date).size()
# Include days between min and max with zero transactions
daily_idx = pd.date_range(df["transaction_date"].min().normalize(), df["transaction_date"].max().normalize(), freq="D")
daily = daily.reindex(daily_idx.date, fill_value=0)
lam = float(daily.mean())
pois_x = np.arange(0, max(int(daily.max()), int(stats.poisson.ppf(.999, lam)))+1)
obs_pois = daily.value_counts(normalize=True).reindex(pois_x, fill_value=0)
pois_probs = stats.poisson.pmf(pois_x, lam)
results["distribution_fits"]["poisson_daily_transactions"] = {
    "lambda_mean_daily_count": lam,
    "daily_mean": float(daily.mean()), "daily_variance": float(daily.var()),
    "note": "Daily counts are aggregated from transaction dates, including zero-transaction dates within the observed date range."
}
plt.bar(pois_x, obs_pois.values, alpha=0.7, label="Observed proportion")
plt.plot(pois_x, pois_probs, "o-", label="Poisson PMF")
plt.title("Daily transaction counts: observed vs Poisson")
plt.xlabel("Transactions per day"); plt.ylabel("Probability / proportion"); plt.legend()
savefig("poisson_daily_transactions.png")

# Lognormal and Pareto-like power-law distribution fits on positive transaction amounts
ln_shape, ln_loc, ln_scale = stats.lognorm.fit(positive, floc=0)
ln_ks, ln_p = stats.kstest(positive, "lognorm", args=(ln_shape, ln_loc, ln_scale))
# Pareto type-I fit, constrained to nonnegative location; descriptive comparison only.
par_b, par_loc, par_scale = stats.pareto.fit(positive, floc=0)
par_ks, par_p = stats.kstest(positive, "pareto", args=(par_b, par_loc, par_scale))
results["distribution_fits"]["lognormal_amounts"] = {
    "shape": float(ln_shape), "loc": float(ln_loc), "scale": float(ln_scale),
    "ks_statistic": float(ln_ks), "ks_p_value": float(ln_p)
}
results["distribution_fits"]["pareto_power_law_amounts"] = {
    "shape": float(par_b), "loc": float(par_loc), "scale": float(par_scale),
    "ks_statistic": float(par_ks), "ks_p_value": float(par_p),
    "note": "A Pareto distribution is used as a simple power-law-style comparison; rigorous power-law inference often requires tail selection and specialized methods."
}
x_grid = np.linspace(positive.min(), positive.quantile(.99), 300)
plt.hist(positive, bins=25, density=True, alpha=.45, label="Observed amounts")
plt.plot(x_grid, stats.lognorm.pdf(x_grid, ln_shape, loc=ln_loc, scale=ln_scale), label="Fitted Log-normal PDF")
plt.plot(x_grid, stats.pareto.pdf(x_grid, par_b, loc=par_loc, scale=par_scale), label="Fitted Pareto PDF")
plt.title("Transaction amounts and fitted distributions")
plt.xlabel("Transaction amount (₹)"); plt.ylabel("Density"); plt.legend()
savefig("amount_distribution_fits.png")

# Q-Q plots raw amounts and log amounts
plt.figure()
stats.probplot(amounts, dist="norm", plot=plt)
plt.title("Q-Q plot: raw transaction amounts vs Normal")
savefig("qq_plot_raw_amounts.png")
plt.figure()
stats.probplot(np.log(positive), dist="norm", plot=plt)
plt.title("Q-Q plot: log transaction amounts vs Normal")
savefig("qq_plot_log_amounts.png")
raw_sw = stats.shapiro(amounts.sample(min(5000, len(amounts)), random_state=42))
log_sw = stats.shapiro(np.log(positive).sample(min(5000, len(positive)), random_state=42))
results["tests_and_probabilities"]["normality_shapiro"] = {
    "raw_amounts_W": float(raw_sw.statistic), "raw_amounts_p": float(raw_sw.pvalue),
    "log_amounts_W": float(log_sw.statistic), "log_amounts_p": float(log_sw.pvalue),
    "interpretation": "Small p-values provide evidence against normality; use Q-Q plots alongside tests."
}

# Box-Cox requires strictly positive data
bc_values, bc_lambda = stats.boxcox(positive)
results["tests_and_probabilities"]["box_cox"] = {
    "lambda": float(bc_lambda),
    "transformed_mean": float(np.mean(bc_values)),
    "transformed_std": float(np.std(bc_values, ddof=1)),
    "note": "Box-Cox was applied to positive transaction amounts only."
}
plt.hist(bc_values, bins=25)
plt.title(f"Transaction amounts after Box-Cox transformation (lambda={bc_lambda:.3f})")
plt.xlabel("Transformed amount"); plt.ylabel("Frequency")
savefig("box_cox_transformed_amounts.png")

# Z-score and probabilities over ₹5000
mean, sd = float(amounts.mean()), float(amounts.std(ddof=1))
threshold = 5000.0
z = (threshold - mean)/sd if sd else float("nan")
normal_tail = float(stats.norm.sf(z)) if np.isfinite(z) else float("nan")
empirical_tail = float((amounts > threshold).mean())
results["tests_and_probabilities"]["amount_over_5000"] = {
    "threshold_rupees": threshold, "z_score_at_threshold": float(z),
    "normal_approx_probability_over_threshold": normal_tail,
    "empirical_proportion_over_threshold": empirical_tail,
    "empirical_count_over_threshold": int((amounts > threshold).sum()),
    "sample_size": int(len(amounts)),
    "note": "The normal-approximation probability assumes a Normal model and may be unreliable for skewed transaction amounts; compare with the empirical proportion."
}

# PDF and CDF plot using fitted lognormal model and empirical CDF
plt.figure()
plt.hist(positive, bins=25, density=True, alpha=.35, label="Empirical density")
plt.plot(x_grid, stats.lognorm.pdf(x_grid, ln_shape, loc=ln_loc, scale=ln_scale), label="Fitted Log-normal PDF")
plt.title("Probability Density Function (PDF) of transaction amounts")
plt.xlabel("Transaction amount (₹)"); plt.ylabel("Density"); plt.legend()
savefig("transaction_pdf.png")
sorted_amounts = np.sort(positive.to_numpy())
emp_cdf = np.arange(1, len(sorted_amounts)+1)/len(sorted_amounts)
plt.figure()
plt.step(sorted_amounts, emp_cdf, where="post", label="Empirical CDF")
plt.plot(x_grid, stats.lognorm.cdf(x_grid, ln_shape, loc=ln_loc, scale=ln_scale), label="Fitted Log-normal CDF")
plt.title("Cumulative Distribution Function (CDF) of transaction amounts")
plt.xlabel("Transaction amount (₹)"); plt.ylabel("Cumulative probability"); plt.legend()
savefig("transaction_cdf.png")

# summary csv
df.describe(include="all").transpose().to_csv(OUT/"descriptive_statistics.csv")
summary = [
    "SPREAD LOCATOR — ANALYSIS SUMMARY",
    "="*40,
    f"Transactions analyzed: {len(df)}",
    f"Amount mean: ₹{mean:.2f}; median: ₹{amounts.median():.2f}; standard deviation: ₹{sd:.2f}",
    f"Amount skewness: {stats.skew(amounts):.3f}",
    "",
    "Distribution fits:",
    f"- Bernoulli estimated success probability: {p_success:.3f}",
    f"- Binomial approximate n={n_trials}, p={p_binom:.3f} (fixed weekly opportunities not specified)",
    f"- Poisson mean daily transactions λ={lam:.3f}",
    f"- Log-normal KS statistic={ln_ks:.4f}, p={ln_p:.4g}",
    f"- Pareto/power-law-style KS statistic={par_ks:.4f}, p={par_p:.4g}",
    "",
    "Diagnostics:",
    f"- Shapiro raw amounts p={raw_sw.pvalue:.4g}; log amounts p={log_sw.pvalue:.4g}",
    f"- Box-Cox lambda={bc_lambda:.3f}",
    f"- Z-score at ₹5000={z:.3f}",
    f"- Normal-approx P(amount > ₹5000)={normal_tail:.4f}",
    f"- Empirical proportion amount > ₹5000={empirical_tail:.4f}",
    "",
    "Interpretation guidance:",
    "- Compare fitted distributions using plots and goodness-of-fit diagnostics; a larger KS p-value alone does not prove a model is correct.",
    "- The Bernoulli model treats each transaction status as success/failure.",
    "- Daily Poisson counts assume events are independent with a stable average rate; inspect mean and variance before relying on this model.",
    "- Box-Cox requires positive values and transforms data to reduce skewness/variance instability.",
    "- Z-score probability uses a Normal approximation, which can be inaccurate for skewed transaction amounts.",
    "- All findings describe this supplied dataset and do not guarantee future customer behavior."
]
(OUT/"analysis_summary.txt").write_text("\n".join(summary), encoding="utf-8")
(OUT/"analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\n".join(summary))
