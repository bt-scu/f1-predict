⏺ F1 Prediction System: Knowledge Transfer Study Guide

  1. Empirical Residual Calibration (The CI Fix)
                                          
  Problem
                                                                                                                     
  Quantile regression predicts scores, but we need positions (ranks). When you rank the 5th percentile scores
  independently from the 95th percentile scores, you get nonsensical results like [4-2] where p10 > p90.             
                                                                                   
  Solution

  Train a single model, measure how wrong it is on held-out data, then apply those error bounds to new predictions.

  Why It's More Stable

  - Quantile regression optimizes for score percentiles, not rank percentiles
  - Empirical residuals directly measure "how many positions off are we typically?"
  - Calibrated to actual historical performance, not theoretical quantiles

  # 1. Split historical data for calibration
  split_idx = int(len(X_hist) * 0.8)
  X_cal_train, X_cal_val = X_hist.iloc[:split_idx], X_hist.iloc[split_idx:]
  Y_cal_train, Y_cal_val = Y_hist.iloc[:split_idx], Y_hist.iloc[split_idx:]

  # 2. Train calibration model and get residuals
  cal_model.fit(X_cal_train, Y_cal_train)
  cal_preds = cal_model.predict(X_cal_val)
  residuals = Y_cal_val.values - cal_preds  # actual - predicted

  # 3. Get 5th/95th percentile of errors (90% CI)
  ci_lower = np.percentile(residuals, 5)   # e.g., -4.2
  ci_upper = np.percentile(residuals, 95)  # e.g., +3.8

  # 4. Apply to new predictions
  pred_p10 = (predicted_score + ci_lower).clip(1, 20).round()
  pred_p90 = (predicted_score + ci_upper).clip(1, 20).round()

  ---
  2. Bayesian Shrinkage (Track Affinity)

  Problem

  With only 2023-2025 data, most drivers have only 1-2 races per track. A single outlier result (e.g., Leclerc P1 at
  Monaco once) would make the model overconfident about future Monaco performance.

  Solution

  "Shrink" the track-specific average toward the driver's global average. The fewer races at a track, the more we
  trust the global average.

  The Math

  shrunk_affinity = (track_avg × n_races + global_avg × prior_weight) / (n_races + prior_weight)

  Where prior_weight = 3 means "trust track-specific data once we have 3+ races there."

  WITH TrackRaw AS (
      SELECT driver_id, circuit_name,
             AVG(grid_position - finish_position) as raw_track_avg,
             COUNT(*) as n_races
      FROM results r JOIN races ra ON r.race_id = ra.race_id
      WHERE ra.year >= 2023
      GROUP BY driver_id, circuit_name
  ),
  GlobalAvg AS (
      SELECT driver_id,
             AVG(grid_position - finish_position) as global_avg
      FROM results GROUP BY driver_id
  )
  SELECT
      t.driver_id, t.circuit_name,
      -- SHRINKAGE: weighted average toward global mean
      ((t.raw_track_avg * t.n_races) + (g.global_avg * 3)) / (t.n_races + 3)
          as shrunk_track_affinity
  FROM TrackRaw t JOIN GlobalAvg g ON t.driver_id = g.driver_id

  Example:
  - Driver has 1 race at Monaco: (+5 × 1 + 0 × 3) / 4 = 1.25 (heavy shrinkage)
  - Driver has 5 races at Monaco: (+5 × 5 + 0 × 3) / 8 = 3.125 (raw data dominates)

  ---
  3. Tire Degradation Proxy

  Problem

  We don't have direct tire sensor data, but tire management skill is crucial for race performance.

  Solution

  Compare lap times at the START vs END of a race. Drivers who degrade more show larger time deltas.

  WITH LapStats AS (
      SELECT race_id, driver_id,
             MAX(lap_number) as total_laps,
             AVG(CASE WHEN lap_number <= 10 THEN lap_time END) as early_stint_avg,
             AVG(CASE WHEN lap_number > (MAX(lap_number) - 10)
                 THEN lap_time END) as late_stint_avg
      FROM laps
      GROUP BY race_id, driver_id
  )
  SELECT race_id, driver_id,
      CASE
          WHEN total_laps >= 40  -- Filter out DNFs/short races
          THEN late_stint_avg - early_stint_avg
          ELSE NULL
      END as tire_deg_proxy
  FROM LapStats

  Interpretation:
  - tire_deg_proxy = 1.5 → Driver's last 10 laps were 1.5s slower than first 10
  - Higher values = worse tire management
  - Fill missing with field median (~1.2s)

  # Lag the feature (use PREVIOUS race's degradation to predict)
  for i in range(1, 4):
      df[f'tire_deg_lag_{i}'] = df.groupby('driver_id')['tire_deg_proxy'].shift(i)
      df[f'tire_deg_lag_{i}'] = df[f'tire_deg_lag_{i}'].fillna(1.2)  # field median

  ---
  4. DNF-Adjusted Expected Value (EV)

  Problem

  Predicted position alone ignores risk. A driver predicted P3 with 30% DNF risk is actually WORSE than a driver
  predicted P5 with 5% DNF risk.

  Solution

  Weight the prediction by survival probability, treating DNF as "effective P20."

  The Formula

  EV_Position = (Predicted_Pos × Survival_Prob) + (20 × DNF_Prob)

  Example:
  - Driver A: Predicted P3, 30% DNF risk → (3 × 0.7) + (20 × 0.3) = 2.1 + 6.0 = 8.1
  - Driver B: Predicted P5, 5% DNF risk → (5 × 0.95) + (20 × 0.05) = 4.75 + 1.0 = 5.75

  Driver B has better EV despite worse raw prediction!

  # Get DNF probability from risk flags
  race_preds['dnf_prob'] = race_preds['driver_id'].apply(
      lambda d: risk_flags_lookup.get(d, {}).get('dnf_prone', 0.15)
  )
  race_preds['dnf_prob'] = race_preds['dnf_prob'].fillna(0.15)  # 15% default

  # Calculate Expected Value
  race_preds['ev_race_pos'] = (
      race_preds['predicted_pos'] * (1 - race_preds['dnf_prob']) +
      20 * race_preds['dnf_prob']
  ).round(1)

  ---
  5. SHAP TreeExplainer

  What It Does

  SHAP (SHapley Additive exPlanations) decomposes a prediction into contributions from each feature. For tree-based
  models, TreeExplainer is optimized to compute these efficiently.

  How It Works

  For each prediction, SHAP answers: "How much did each feature push the prediction up or down from the baseline?"

  import shap

  # Create explainer from trained XGBoost model
  explainer = shap.TreeExplainer(model)

  # Get SHAP values for test set (shape: n_samples × n_features)
  shap_values = explainer.shap_values(X_test)

  # For each prediction, extract top 5 influential factors
  def _get_shap_top_factors(shap_values: np.ndarray, feature_names: list, top_n: int = 5) -> str:
      shap_dict = dict(zip(feature_names, shap_values))
      # Sort by absolute value (magnitude matters, not direction)
      top_factors = dict(
          sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
      )
      top_factors = {k: round(float(v), 3) for k, v in top_factors.items()}
      return json.dumps(top_factors)

  # Apply to each row
  race_preds['race_factors_json'] = [
      _get_shap_top_factors(shap_values[i], feature_names)
      for i in range(len(shap_values))
  ]

  Reading the Output

  {"grid": -2.1, "constructor_prev_1": -0.8, "tire_deg_lag_1": 0.5, ...}
  - Negative values → Push prediction toward P1 (better)
  - Positive values → Push prediction toward P20 (worse)
  - grid: -2.1 means "starting position contributed 2.1 positions better than average"

  ---
  Quick Reference Card
  ┌───────────────────┬────────────────────────────────────────┬────────────────────────────────────────────────────┐
  │      Concept      │             Problem Solved             │                    Key Insight                     │
  ├───────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Empirical CI      │ Quantile regression breaks for         │ Measure actual errors, not theoretical quantiles   │
  │                   │ rankings                               │                                                    │
  ├───────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Bayesian          │ Small samples overfit                  │ Trust global average until we have enough track    │
  │ Shrinkage         │                                        │ data                                               │
  ├───────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ Tire Deg Proxy    │ No direct sensor data                  │ Late laps - early laps = degradation signal        │
  ├───────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ DNF-Adjusted EV   │ Risk ignored in raw prediction         │ Weight by survival probability                     │
  ├───────────────────┼────────────────────────────────────────┼────────────────────────────────────────────────────┤
  │ SHAP              │ Black-box predictions                  │ Decompose into per-feature contributions           │
  └───────────────────┴────────────────────────────────────────┴────────────────────────────────────────────────────┘


    Key Clarification: You're NOT Using Quantile Regression                                                                  
                  
  Looking at your pred_race.py, the docs actually explain that quantile regression was abandoned because it breaks when    
  converting scores to ranks. Your code uses Empirical Residual Calibration instead — a simpler, more robust approach.

  ---
  Where Each Technique Lives in Your Code
  ┌───────────────────────────────┬─────────────────┬────────────────────────────────────────────────┐
  │           Technique           │  File Location  │                     Lines                      │
  ├───────────────────────────────┼─────────────────┼────────────────────────────────────────────────┤
  │ Empirical CI (residual-based) │ ml/pred_race.py │ 464-496 (calibration), 547-556 (application)   │
  ├───────────────────────────────┼─────────────────┼────────────────────────────────────────────────┤
  │ Bayesian Shrinkage            │ ml/pred_race.py │ 17-58 (_compute_track_affinity_with_shrinkage) │
  ├───────────────────────────────┼─────────────────┼────────────────────────────────────────────────┤
  │ Tire Degradation Proxy        │ ml/pred_race.py │ 61-98 (_compute_tire_deg_proxy)                │
  ├───────────────────────────────┼─────────────────┼────────────────────────────────────────────────┤
  │ DNF-Adjusted EV               │ ml/pred_race.py │ 576-585                                        │
  ├───────────────────────────────┼─────────────────┼────────────────────────────────────────────────┤
  │ SHAP Explainability           │ ml/pred_race.py │ 537-563                                        │
  └───────────────────────────────┴─────────────────┴────────────────────────────────────────────────┘
  ---
  Learning Resources by Technique

  1. Empirical Residual Calibration (what you actually have)
  - This is a form of conformal prediction — a hot topic in ML uncertainty quantification
  - Paper: "A Gentle Introduction to Conformal Prediction" by Angelopoulos & Bates (2021) — very readable
  - Key idea: measure how wrong your model is on held-out data, use those errors as your uncertainty bands

  2. Bayesian Shrinkage (James-Stein / Empirical Bayes)
  - Classic paper: Efron & Morris "Stein's Paradox in Statistics" (Scientific American, 1977) — accessible intro
  - Book: "Computer Age Statistical Inference" by Efron & Hastie, Chapter 6
  - The formula (track_avg × n + global_avg × prior) / (n + prior) is a posterior mean under a normal prior

  3. SHAP Values
  - Original paper: Lundberg & Lee "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)
  - Interactive tutorial: https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explai
  nable%20AI%20with%20Shapley%20values.html
  - The key insight: Shapley values come from cooperative game theory — each feature is a "player" and SHAP computes their
  fair contribution

  4. DNF-Adjusted Expected Value
  - This is basic expected value from probability theory
  - Any intro probability textbook covers this (e.g., Blitzstein's "Introduction to Probability")
  - Formula: E[X] = Σ outcome × P(outcome)
