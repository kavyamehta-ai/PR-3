# Theory Notes — Spread Locator

## 1. Statistical distributions
A statistical distribution describes the possible values of a variable and how frequently or likely those values occur. It helps summarize patterns and estimate probabilities.

## 2. Q-Q plot
A quantile-quantile (Q-Q) plot compares sample quantiles with theoretical quantiles from a chosen distribution. Points close to a straight line suggest the data may fit that distribution; systematic curves or tail departures suggest a mismatch.

## 3. Discrete vs continuous distributions
A discrete distribution assigns probabilities to countable outcomes, such as 0, 1, 2 transactions. A continuous distribution describes values over a continuum, such as transaction amounts; probabilities are measured over intervals using a density.

## 4. Bernoulli distribution
A Bernoulli distribution models one trial with two outcomes, commonly success (1) or failure (0). It is described by the success probability p, with failure probability 1 − p.

## 5. Binomial distribution
A Binomial distribution models the number of successes in a fixed number n of independent Bernoulli trials, each with the same success probability p. Its parameters are n and p.

## 6. Log-normal distribution
A positive variable is log-normally distributed when its natural logarithm follows a Normal distribution. It is often considered for positive monetary values that are right-skewed.

## 7. Power-law distribution
A power-law distribution has a heavy tail: large values are less common, but can occur more often than under many light-tailed models. A Pareto distribution is a common power-law-style model; careful tail selection is important when fitting it.

## 8. Box-Cox transformation
The Box-Cox transformation changes positive numerical data using a power parameter λ. It can reduce skewness and make variance more stable; the standard transformation requires strictly positive values.

## 9. Poisson distribution
A Poisson distribution models the number of events in a fixed interval when events occur independently at a constant average rate λ. For example, it can model the number of transactions recorded per day if its assumptions are reasonably met.

## 10. Z-score probability
A z-score expresses how many standard deviations an observation is above or below the mean: z = (x − μ) / σ. Under a Normal model, the standard Normal distribution can be used to estimate probabilities beyond a threshold.

## 11. PDF vs CDF
A Probability Density Function (PDF) describes the relative density of a continuous variable; the area under it over an interval gives the probability of that interval. A Cumulative Distribution Function (CDF) gives P(X ≤ x), the probability that a variable is at most x.

## Notes on interpreting this project
- A goodness-of-fit test does not prove a distribution is correct. Use visual diagnostics and domain knowledge too.
- The dataset's `transaction_count` column does not specify a fixed number of weekly opportunities, so the Binomial fit in the script is explicitly approximate.
- The normal approximation for amounts above ₹5000 may be weak if amounts are skewed; compare it with the empirical proportion.
