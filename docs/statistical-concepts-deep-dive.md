# Statistical Concepts Deep Dive

A learning guide designed to build intuition before formulas.

---

## 1. Empirical Residual Calibration

### The Problem (Start Here)

Imagine you're a weather forecaster. You predict "72°F" for tomorrow. But how confident should you be?

You could say "somewhere between 60°F and 84°F" but that's useless. You could say "between 71°F and 73°F" but you'd be wrong half the time.

**The insight**: Look at how wrong you've been historically, and use that to set your uncertainty bands.

### Building Intuition

Let's say over the past 100 days, you predicted temperatures and tracked your errors:

```
Day 1: Predicted 70, Actual 72 → Error = +2
Day 2: Predicted 75, Actual 71 → Error = -4
Day 3: Predicted 68, Actual 69 → Error = +1
... (100 days)
```

If you sort all 100 errors and find:
- The 5th smallest error was -3°F
- The 95th smallest error was +4°F

Then you can say: "90% of the time, my predictions are between 3°F too high and 4°F too low."

**That's it.** That's the whole technique.

### The Math

```
residuals = actual_values - predicted_values

ci_lower = percentile(residuals, 5)   # e.g., -4.2
ci_upper = percentile(residuals, 95)  # e.g., +3.8

# For a new prediction:
prediction_low  = new_prediction + ci_lower
prediction_high = new_prediction + ci_upper
```

### In Your Code (pred_race.py:464-496)

```python
# Train model on 80% of historical data
cal_model.fit(X_cal_train, Y_cal_train)

# Get predictions on held-out 20%
cal_preds = cal_model.predict(X_cal_val)

# Measure errors: how many positions off?
residuals = Y_cal_val.values - cal_preds

# Find the 5th and 95th percentile of errors
ci_lower_bound = np.percentile(residuals, 5)   # e.g., -4.2 positions
ci_upper_bound = np.percentile(residuals, 95)  # e.g., +3.8 positions
```

Later, for new predictions (line 549-550):
```python
race_preds['pred_race_pos_p10'] = (predicted_score + ci_lower_bound).clip(1, 20)
race_preds['pred_race_pos_p90'] = (predicted_score + ci_upper_bound).clip(1, 20)
```

### Why Not Quantile Regression?

Quantile regression trains separate models for different percentiles (p10, p50, p90). The problem:

1. Each model optimizes for score percentiles, not rank percentiles
2. When you convert scores to ranks, the order can flip
3. You get nonsense like "90% CI: positions 4-2" (impossible)

Empirical calibration avoids this by measuring actual position errors.

### Exercise to Verify Understanding

Add this to your code temporarily to visualize:

```python
import matplotlib.pyplot as plt

# After computing residuals (around line 491)
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black')
plt.axvline(ci_lower_bound, color='red', linestyle='--', label=f'5th %ile: {ci_lower_bound:.1f}')
plt.axvline(ci_upper_bound, color='red', linestyle='--', label=f'95th %ile: {ci_upper_bound:.1f}')
plt.xlabel('Prediction Error (positions)')
plt.ylabel('Frequency')
plt.title('Distribution of Model Errors')
plt.legend()
plt.savefig('residual_distribution.png')
```

**What you should see**: A roughly bell-shaped distribution centered near 0, with the red lines capturing 90% of the area.

### Further Reading

- "A Gentle Introduction to Conformal Prediction" (Angelopoulos & Bates, 2021)
- This technique is a simplified version of "split conformal prediction"

---

## 2. Bayesian Shrinkage

### The Problem (Start Here)

Charles Leclerc finished P1 at Monaco in 2024. Should you predict he'll win Monaco every year?

Obviously not. That single data point could be:
- Skill (signal)
- Lucky safety car timing (noise)
- Competitors had failures (noise)

With only 1-2 races per driver at each track, **small samples lie**.

### Building Intuition: The Batting Average Analogy

Baseball example: A player goes 4-for-4 in their first game. Their batting average is 1.000 (perfect).

Should you bet they'll hit 1.000 all season? No, because:
1. Sample size is tiny (4 at-bats)
2. League average is ~.250
3. The "true" ability is probably somewhere between

**Shrinkage** pulls extreme estimates toward a more reasonable baseline.

### The Formula

```
shrunk_estimate = (sample_mean × n + prior_mean × k) / (n + k)
```

Where:
- `sample_mean` = what you observed (e.g., Leclerc's Monaco average: +5 positions)
- `n` = how many observations (e.g., 1 race)
- `prior_mean` = baseline expectation (e.g., Leclerc's global average: 0)
- `k` = "prior strength" — essentially a set constant to ensure that if n = 1 we don't skew

### Concrete Example

**Leclerc at Monaco**: 1 race, gained 5 positions
**Leclerc globally**: averages 0 position change
**Prior strength**: k = 3

```
shrunk = (5 × 1 + 0 × 3) / (1 + 3) = 5/4 = 1.25
```

Instead of predicting +5 positions (overfit to one race), we predict +1.25.

**Verstappen at Spa**: 5 races, averages +2 positions
**Verstappen globally**: averages +1 position

```
shrunk = (2 × 5 + 1 × 3) / (5 + 3) = 13/8 = 1.625
```

With more data, the raw track average dominates.

### In Your Code (pred_race.py:17-58)

```sql
-- The SQL implements exactly this formula
((t.raw_track_avg * t.n_races) + (g.global_avg * 3)) / (t.n_races + 3)
    as shrunk_track_affinity
```

The `3` is the prior strength — chosen because most drivers have ~2-3 races per track in your 2023-2025 dataset.

### Why "Bayesian"?

This is actually **empirical Bayes** — a shortcut to full Bayesian inference:

- Full Bayesian: Define a prior distribution, compute posterior using Bayes' theorem
- Empirical Bayes: Use the data itself to estimate the prior (global average), then apply shrinkage

The formula is the **posterior mean** under a normal prior with known variance.

### Visual Intuition

```
                    More data at track
                    ─────────────────────────►

Prior mean ●────────────────────────────────● Raw track mean
(global avg)                                  (overfit danger)

           With n=1: ●─────────────────────── (mostly prior)
           With n=3: ────────●──────────────── (balanced)
           With n=10: ──────────────────●───── (mostly data)
```

### Exercise to Verify Understanding

Query your database:

```sql
SELECT
    driver_id,
    circuit_name,
    raw_track_avg,
    n_races,
    global_avg,
    shrunk_track_affinity,
    -- See how much shrinkage occurred
    ABS(raw_track_avg - shrunk_track_affinity) as shrinkage_amount
FROM (
    -- Your existing query from _compute_track_affinity_with_shrinkage
)
ORDER BY shrinkage_amount DESC
LIMIT 20;
```

**What you should see**: Drivers with n_races=1 have the most shrinkage.

### Further Reading

- "Stein's Paradox in Statistics" (Efron & Morris, Scientific American 1977) — the original accessible explanation
---

## 3. Tire Degradation Proxy

### The Problem

Tire data is proprietary. Pirelli doesn't publish degradation curves. But tire management is crucial — a driver who destroys their tires will fade in the race.

### The Insight

We can't measure tire wear directly, but we can measure its **effect**: lap time degradation.

```
Early stint (laps 1-10): Driver averages 92.5 seconds
Late stint (laps 51-60): Driver averages 94.0 seconds

Degradation proxy = 94.0 - 92.5 = 1.5 seconds
```

A driver with 1.5s degradation has worse tire management than one with 0.8s.

### Why This Works

The proxy captures multiple real effects:
1. **Tire compound choice** — softer compounds degrade faster
2. **Driving style** — aggressive drivers wear tires more
3. **Car setup** — some setups are harder on tires
4. **Track characteristics** — high-deg tracks hurt everyone

We don't need to separate these — we just need a predictive signal.

### In Your Code (pred_race.py:61-98)

```sql
AVG(CASE WHEN lap_number <= 10 THEN lap_time END) as early_stint_avg
AVG(CASE WHEN lap_number > (total_laps - 10) THEN lap_time END) as late_stint_avg

-- Proxy = how much slower at the end
late_stint_avg - early_stint_avg as tire_deg_proxy
```

Then you lag it (use previous races to predict future):
```python
df[f'tire_deg_lag_{i}'] = df.groupby('driver_id')['tire_deg_proxy'].shift(i)
```

### The Lagging is Crucial

You can't use this race's degradation to predict this race's result (that's leakage). You use **past** degradation patterns to predict **future** performance.

### Exercise

Plot tire degradation by constructor:

```python
tire_df = pd.read_sql("""
    SELECT r.constructor_id, AVG(tire_deg_proxy) as avg_deg
    FROM your_tire_query
    GROUP BY constructor_id
    ORDER BY avg_deg
""", engine)

# Lower = better tire management
tire_df.plot(kind='barh', x='constructor_id', y='avg_deg')
```

---

## 4. DNF-Adjusted Expected Value

### The Problem

Driver A: Predicted P3, 30% DNF risk
Driver B: Predicted P5, 5% DNF risk

Who's the better fantasy pick?

Raw prediction says Driver A. But accounting for risk...

### Expected Value 101

From probability theory:

```
E[X] = Σ (outcome × probability of that outcome)
```

For our problem:
- Outcome 1: Driver finishes (probability = 1 - DNF_prob)
- Outcome 2: Driver DNFs (probability = DNF_prob)

We treat DNF as "effectively P20" (last place) for fantasy purposes.

### The Calculation

```
EV = (Predicted_Position × Finish_Prob) + (20 × DNF_Prob)
```

**Driver A**: (3 × 0.70) + (20 × 0.30) = 2.1 + 6.0 = **8.1**
**Driver B**: (5 × 0.95) + (20 × 0.05) = 4.75 + 1.0 = **5.75**

Driver B has better expected value despite worse raw prediction.

### In Your Code (pred_race.py:576-585)

```python
race_preds['ev_race_pos'] = (
    race_preds['predicted_pos'] * (1 - race_preds['dnf_prob']) +
    20 * race_preds['dnf_prob']
).round(1)
```

### Why This Matters for Fantasy

Fantasy scoring often has big penalties for DNF (0 points or negative). Risk-adjusted predictions help you avoid:
- The fast but fragile drivers (high ceiling, high risk)
- Favoring the consistent midfielders in tight matchups

### Generalization

This concept extends everywhere:
- **Stock picking**: Expected return = (gain × P(win)) + (loss × P(lose))
- **Insurance**: Premium = Expected claims + margin
- **Poker**: EV of a bet = (pot × P(win)) - (bet × P(lose))

---

## 5. SHAP Values (SHapley Additive exPlanations)

### The Problem

Your XGBoost model predicts Verstappen P2 at Silverstone. But **why**?

Is it because:
- His grid position?
- Red Bull's recent form?
- His practice lap times?
- Track-specific affinity?

Black-box models don't tell you. SHAP does.

### The Intuition: A Team Bonus Analogy

Imagine a project team of 4 people delivers a project worth $100,000. How do you fairly split the bonus?

Option 1: Equal split ($25k each) — Ignores contribution differences
Option 2: Manager decides — Arbitrary

**Shapley's solution** (from game theory, 1953):
1. Consider all possible orderings of when people joined
2. For each ordering, measure what each person added
3. Average across all orderings

This gives a "fair" contribution that accounts for:
- How much you add when joining an empty team
- How much you add when joining a partial team
- How much you add when joining a nearly-complete team

### Applying to ML Features

Replace "team members" with "features":

1. Start with no features (baseline prediction)
2. Add features one by one, measure prediction change
3. Try all possible orderings
4. Average each feature's contribution

### A Simple Example

Model predicts Verstappen P2.0 for the race.
Baseline prediction (average driver) = P10.0

The model moved the prediction from P10 → P2, a total shift of -8 positions.

SHAP decomposes this:
```json
{
  "grid": -3.5,           // Starting P1 contributes -3.5 positions
  "constructor_prev_1": -2.1,  // Red Bull's form contributes -2.1
  "best_lap_1": -1.8,     // Practice pace contributes -1.8
  "tire_deg_lag_1": -0.4, // Good tire management contributes -0.4
  "teammate_gap": -0.2    // Beating Perez contributes -0.2
}
// Total: -8.0 (matches the shift from baseline)
```

**Key property**: SHAP values always sum to (prediction - baseline).

### In Your Code (pred_race.py:537-563)

```python
# Create explainer from trained model
explainer = shap.TreeExplainer(model)

# Get SHAP values for each prediction
shap_values = explainer.shap_values(X_test)

# Extract top 5 factors for each driver
def _get_shap_top_factors(shap_values, feature_names, top_n=5):
    shap_dict = dict(zip(feature_names, shap_values))
    # Sort by absolute magnitude
    top_factors = dict(sorted(shap_dict.items(),
                              key=lambda x: abs(x[1]),
                              reverse=True)[:top_n])
    return json.dumps(top_factors)
```

### Reading SHAP Output

```json
{"grid": -2.1, "constructor_prev_1": -0.8, "tire_deg_lag_1": 0.5}
```

- **Negative values** → Push prediction toward P1 (better)
- **Positive values** → Push prediction toward P20 (worse)
- **Magnitude** → How many positions this feature contributed

### Why TreeExplainer is Fast

Computing exact Shapley values requires exponential time (2^n orderings for n features).

TreeExplainer exploits the tree structure to compute exact values in polynomial time. It's one of the reasons SHAP became practical for real applications.

### Exercise: Create a SHAP Summary Plot

```python
import shap
import matplotlib.pyplot as plt

# After training your model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot shows feature importance + direction
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
plt.savefig('shap_summary.png')
```

**What you should see**:
- Features ordered by importance (top = most influential)
- Red/blue dots showing how high/low values affect predictions
- Spread showing how consistent the effect is

### Further Reading

- Original paper: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, NeurIPS 2017)
- Interactive tutorial: https://shap.readthedocs.io
- The math: Based on Shapley values from cooperative game theory (1953 Nobel Prize)

---

## Putting It All Together

Here's how these concepts flow through your prediction:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. FEATURE ENGINEERING                                         │
│     - Bayesian shrinkage on track affinity (handles small n)    │
│     - Tire degradation proxy (indirect measurement)             │
│     - Lagged features (no data leakage)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. MODEL TRAINING                                              │
│     - XGBoost learns complex interactions                       │
│     - Calibration model measures historical errors              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. PREDICTION + UNCERTAINTY                                    │
│     - Point prediction from model                               │
│     - Empirical CI from calibration residuals                   │
│     - DNF-adjusted EV incorporates risk                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. EXPLAINABILITY                                              │
│     - SHAP values decompose each prediction                     │
│     - Risk flags provide human-readable warnings                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommended Learning Path

1. **Week 1**: Empirical calibration
   - Implement the histogram exercise
   - Read the Angelopoulos & Bates paper (first 10 pages)

2. **Week 2**: Bayesian shrinkage
   - Query your database to see shrinkage in action
   - Read the Efron & Morris Scientific American article

3. **Week 3**: Expected value
   - Practice with simple betting scenarios
   - Calculate EV for a few drivers by hand

4. **Week 4**: SHAP
   - Generate summary plots for your models
   - Read the Lundberg paper introduction + section 2

---

## Quick Reference

| Concept | One-Sentence Summary |
|---------|---------------------|
| Empirical CI | Measure past errors, apply to future predictions |
| Bayesian Shrinkage | Don't trust small samples; blend with baseline |
| Tire Deg Proxy | Can't measure directly? Measure the effect |
| DNF-Adjusted EV | Risk-weight your predictions |
| SHAP | Fairly attribute prediction to each feature |
