# Position Uncertainty via Empirical Residual Calibration

## Problem

Predicting the exact finishing position of a Formula 1 driver is highly uncertain due to unpredictable factors such as crashes, weather, strategy changes, and mechanical failures. With limited historical data, producing precise position estimates requires strong assumptions and excessive rounding, making single-point predictions unreliable.

Instead of predicting an exact position, we provide a **range of expected positions** for each driver.

To achieve this, we use **Empirical Residual Calibration**.

---

## Method

1. Run the trained model to generate position predictions.
2. Compute residuals using validation data:
3. Analyze the empirical distribution of residuals.
4. Use the 5th and 95th percentiles of residuals to construct 90% confidence intervals.
5. Create lower and upper bounds around each prediction:


These bounds represent the worst and best expected outcomes for each driver.

---

## Implementation

Residual calibration is applied separately for each prediction task:

- **Race** (`ml/pred_race.py`, lines 464–496): `[-4.8, 5.0]`
- **Sprint** (`ml/pred_sprint.py`, lines 217–246): `[-7.0, 5.0]`
- **Qualifying** (`ml/pred_quali.py`, lines 156–185): `[-6.2, 6.3]`

Each range corresponds to the empirical residual bounds used for interval construction.

---

## Results

Model calibration is evaluated using coverage, defined as the percentage of true outcomes that fall within the predicted interval.

| Prediction Type | 90% CI Coverage | Target |
|-----------------|----------------|--------|
| Race            | 88.1%          | ~90%   |
| Qualifying      | 87.9%          | ~90%   |
| Sprint          | 85.8%          | ~90%   |

---

## Conclusion

Empirical Residual Calibration allows the model to account for real-world race uncertainty by converting point predictions into well-calibrated confidence intervals. The resulting coverage values are close to the target 90%, indicating that the method provides reliable and interpretable uncertainty estimates for F1 position predictions.

