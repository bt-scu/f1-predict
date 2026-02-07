import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import xgboost as xgb
import shap
import json

# 1. SETUP DATABASE
DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

# Weather source for future-proof predictions (Friday night runs)
from weather_source import get_race_weather


def _compute_quali_track_affinity_with_shrinkage(engine) -> pd.DataFrame:
    """
    Compute track-specific qualifying affinity with Bayesian shrinkage.

    For qualifying, we measure: how does this driver typically qualify at this track
    relative to their global average?

    Uses shrinkage toward global mean to handle small sample sizes
    (only ~2-3 quali sessions per driver-track combination in 2023-2025 data).

    Prior weight of 3 means we trust track-specific data once we have 3+ sessions.
    """
    query = text("""
    WITH TrackRaw AS (
        SELECT
            q.driver_id,
            ra.circuit_name,
            AVG(q.position) as raw_track_avg,
            COUNT(*) as n_sessions
        FROM qualifying q
        JOIN races ra ON q.race_id = ra.race_id
        WHERE ra.year >= 2023
        GROUP BY q.driver_id, ra.circuit_name
    ),
    GlobalAvg AS (
        SELECT
            driver_id,
            AVG(position) as global_avg
        FROM qualifying q
        JOIN races ra ON q.race_id = ra.race_id
        WHERE ra.year >= 2023
        GROUP BY driver_id
    )
    SELECT
        t.driver_id,
        t.circuit_name,
        t.raw_track_avg,
        t.n_sessions,
        g.global_avg,
        -- SHRINKAGE FORMULA: weighted avg toward global mean
        -- Prior weight = 3 (acts like "3 imaginary sessions at global avg")
        ((t.raw_track_avg * t.n_sessions) + (g.global_avg * 3)) / (t.n_sessions + 3)
            as shrunk_quali_track_affinity
    FROM TrackRaw t
    JOIN GlobalAvg g ON t.driver_id = g.driver_id
    """)

    return pd.read_sql(query, engine)


def _get_shap_top_factors(shap_values: np.ndarray, feature_names: list, top_n: int = 5) -> str:
    """
    Extract top N contributing factors from SHAP values and return as JSON.
    """
    shap_dict = dict(zip(feature_names, shap_values))
    # Sort by absolute value to get most influential factors
    top_factors = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n])
    # Round values for cleaner output
    top_factors = {k: round(float(v), 3) for k, v in top_factors.items()}
    return json.dumps(top_factors)


def predict_quali_outcomes():
    print("Fetching data (Safe Mode)...")

    # 2. FETCH DATA
    query = text("""
      -- 1. FUNCTIONAL QUERY 1: Best Practice Time
      WITH PracticeBest AS (
          SELECT
              race_id, driver_id,
              MIN(lap_time) as best_practice_time
          FROM practice_laps
          GROUP BY race_id, driver_id
      ),

      -- 2. FUNCTIONAL QUERY 2: Weather
      WeatherPrep AS (
          SELECT
              race_id,
              AVG(air_temp) as air_temp
          FROM weather
          WHERE session_type LIKE 'FP%' OR session_type LIKE 'Practice%'
          GROUP BY race_id
      )

      -- 3. THE MAIN SELECT
      SELECT
          -- Core IDs
          r.race_id, r.year, r.round, r.circuit_name,
          q.driver_id,
          res.constructor_id,

          -- Target Variable (Quali Position)
          q.position AS target_quali_pos,

          -- Feature: Pace (from Function 1)
          p.best_practice_time,

          -- Feature: Weather (from Function 2) - air_temp only for Friday runs
          w.air_temp

      -- 4. JOINING IT ALL TOGETHER
      FROM qualifying q

      -- Join standard info
      INNER JOIN races r ON q.race_id = r.race_id

      -- Join Results (to get constructor info)
      INNER JOIN results res
          ON q.race_id = res.race_id AND q.driver_id = res.driver_id

      -- Join Function 1 (Practice Pace)
      LEFT JOIN PracticeBest p
          ON q.race_id = p.race_id AND q.driver_id = p.driver_id

      -- Join Function 2 (Weather)
      LEFT JOIN WeatherPrep w
          ON q.race_id = w.race_id

      WHERE r.year >= 2023
      ORDER BY r.year ASC, r.round ASC
  """)

    try:
        df = pd.read_sql(query, engine)
        print(f"Data Fetched: {df.shape}")
    except Exception as e:
        print(f"Database Error: {e}")
        return

    # 3. FEATURE ENGINEERING
    df['best_practice_time'] = df['best_practice_time'].fillna(200)
    session_best = df.groupby('race_id')['best_practice_time'].transform('min')
    df['gap_to_best'] = (df['best_practice_time'] - session_best).clip(upper=2.5)

    df['practice_rank'] = df.groupby('race_id')['best_practice_time'].rank(ascending=True)
    df['practice_rank'] = df['practice_rank'].fillna(20)

    team_pace = df.groupby(['race_id', 'constructor_id'])['best_practice_time'].transform('mean')
    df['teammate_delta'] = df['best_practice_time'] - team_pace

    df['season_avg_pos'] = df.groupby(['year', 'driver_id'])['target_quali_pos'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df['season_avg_pos'] = df['season_avg_pos'].fillna(10.0)

    df = df.sort_values(by=['year', 'driver_id', 'round'])

    for i in range(1, 6):
        col_name = f'quali_lag_{i}'
        df[col_name] = df.groupby(['year', 'driver_id'])['target_quali_pos'].shift(i)
        df[col_name] = df[col_name].fillna(10.0)

    df['constructor_recent_form'] = df.groupby(['year', 'constructor_id'])['target_quali_pos'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df['constructor_recent_form'] = df['constructor_recent_form'].fillna(10)

    # TRACK-SPECIFIC AFFINITY WITH BAYESIAN SHRINKAGE
    # Replaces naive driver_track_avg to prevent overfitting on small samples
    print("Computing track-specific qualifying affinity with shrinkage...")
    quali_track_affinity_df = _compute_quali_track_affinity_with_shrinkage(engine)
    df = df.merge(
        quali_track_affinity_df[['driver_id', 'circuit_name', 'shrunk_quali_track_affinity']],
        on=['driver_id', 'circuit_name'],
        how='left'
    )
    # Fill missing (new driver-track combos) with global median (~10)
    df['shrunk_quali_track_affinity'] = df['shrunk_quali_track_affinity'].fillna(10.0)

    # Retype certain features for predictive compatibility
    df['constructor_id'] = df['constructor_id'].astype('category')
    df['round_cat'] = df['round'].astype('category')
    df['driver_id'] = df['driver_id'].astype('category')
    df['circuit_name'] = df['circuit_name'].astype('category')

    feature_cols = [
        'gap_to_best',
        'season_avg_pos',
        'constructor_id',
        'round_cat',
        'quali_lag_1',
        'quali_lag_2',
        'quali_lag_3',
        'quali_lag_4',
        'quali_lag_5',
        'air_temp',  # Only air_temp - track_temp not forecastable 48h out (Friday runs)
        'practice_rank',
        'teammate_delta',
        'constructor_recent_form',
        'shrunk_quali_track_affinity',  # Bayesian-shrunk track-specific performance
        'driver_id',
        'circuit_name'
    ]

    # ===== EMPIRICAL CONFIDENCE BANDS FROM HISTORICAL DATA =====
    print("Computing empirical confidence bands from historical residuals...")
    hist_train_mask = (df['year'] < 2025) & (df['target_quali_pos'].notna())
    X_hist = df.loc[hist_train_mask, feature_cols]
    Y_hist = df.loc[hist_train_mask, 'target_quali_pos']

    # Use 80/20 split for residual estimation
    split_idx = int(len(X_hist) * 0.8)
    X_cal_train, X_cal_val = X_hist.iloc[:split_idx], X_hist.iloc[split_idx:]
    Y_cal_train, Y_cal_val = Y_hist.iloc[:split_idx], Y_hist.iloc[split_idx:]

    # Train calibration model
    cal_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        enable_categorical=True,
        early_stopping_rounds=50,
        objective='reg:squarederror',
        n_jobs=-1,
    )
    cal_model.fit(X_cal_train, Y_cal_train, eval_set=[(X_cal_val, Y_cal_val)], verbose=False)

    # Compute residuals on validation set
    cal_preds = cal_model.predict(X_cal_val)
    residuals = Y_cal_val.values - cal_preds

    # Get 5th and 95th percentile of residuals for 90% CI
    ci_lower_bound = np.percentile(residuals, 5)
    ci_upper_bound = np.percentile(residuals, 95)
    print(f"  Empirical CI bounds: [{ci_lower_bound:.1f}, {ci_upper_bound:.1f}] positions")

    store_predictions = []

    for r in sorted(df[df['year'] == 2025]['round'].unique()):
        # Training on all previous results
        train_mask = (
            ((df['year'] < 2025) | ((df['year'] == 2025) & (df['round'] < r))) &
            (df['target_quali_pos'].notna())
        )

        # Testing on only the current race data
        test_mask = (df['year'] == 2025) & (df['round'] == r)

        # Fill weather data for future races using weather_source
        race_ids_test = df.loc[test_mask, 'race_id'].unique()
        for rid in race_ids_test:
            weather = get_race_weather(rid)
            # Fill air_temp for rows where it's missing (future races)
            mask = (df['race_id'] == rid) & (df['air_temp'].isna())
            df.loc[mask, 'air_temp'] = weather['air_temp']

        X_train = df.loc[train_mask, feature_cols]
        Y_train = df.loc[train_mask, 'target_quali_pos']

        X_test = df.loc[test_mask, feature_cols]

        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=5,
            enable_categorical=True,
            early_stopping_rounds=50,
            objective='reg:squarederror',
            n_jobs=-1,
        )

        model.fit(
            X_train, Y_train,
            eval_set=[(X_train, Y_train)],
            verbose=False
        )

        preds = model.predict(X_test)

        # Compute SHAP values for explainability
        print(f"  Computing SHAP values for Round {r}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        race_preds = df.loc[test_mask].copy()
        race_preds['predicted_score'] = preds
        race_preds['predicted_pos'] = race_preds['predicted_score'].rank(method='first').astype(int)

        # Apply empirical residual-based confidence bands
        race_preds['pred_quali_pos_p10'] = (race_preds['predicted_score'] + ci_lower_bound).clip(1, 20).round().astype(int)
        race_preds['pred_quali_pos_p90'] = (race_preds['predicted_score'] + ci_upper_bound).clip(1, 20).round().astype(int)

        # Ensure p10 <= p90
        race_preds['pred_quali_pos_p10'], race_preds['pred_quali_pos_p90'] = (
            race_preds[['pred_quali_pos_p10', 'pred_quali_pos_p90']].min(axis=1),
            race_preds[['pred_quali_pos_p10', 'pred_quali_pos_p90']].max(axis=1)
        )

        # Add SHAP explanations
        feature_names = [str(f) for f in feature_cols]
        race_preds['quali_factors_json'] = [
            _get_shap_top_factors(shap_values[i], feature_names)
            for i in range(len(shap_values))
        ]

        store_predictions.append(race_preds[[
            'race_id', 'year', 'round', 'driver_id',
            'target_quali_pos', 'predicted_pos',
            'pred_quali_pos_p10', 'pred_quali_pos_p90',
            'quali_factors_json'
        ]])
        print(f"Round {r} Prediction Complete.")

    # 5. RESULTS & ANALYTICS
    if store_predictions:
        final_results = pd.concat(store_predictions)

        # Calculate diffs for stats
        final_results['diff'] = final_results['predicted_pos'] - final_results['target_quali_pos']
        final_results['abs_diff'] = final_results['diff'].abs()

        print("\n" + "="*60)
        print("QUALIFYING PREDICTION LOG")
        print("="*60)

        grouped = final_results.groupby(['round'], sort=True)

        for round_num, race_df in grouped:

            print(f"\nROUND {round_num}")
            print(f"{'Pred':<6} {'Driver':<12} {'Actual':<8} {'CI':<10} {'Diff'}")
            print("-" * 55)

            race_df_sorted = race_df.sort_values('predicted_pos')

            for _, row in race_df_sorted.iterrows():
                pred = int(row['predicted_pos'])
                p10 = int(row['pred_quali_pos_p10'])
                p90 = int(row['pred_quali_pos_p90'])
                ci_str = f"[{p10}-{p90}]"

                if pd.notna(row['target_quali_pos']):
                    actual = int(row['target_quali_pos'])
                    diff = int(row['diff'])

                    if diff == 0:
                        diff_str = "OK"
                    elif abs(diff) <= 2:
                        diff_str = f"{diff:+d}"
                    else:
                        diff_str = f"! {diff:+d}"
                else:
                    actual = "-"
                    diff_str = "?"

                print(f"{pred:<6} {row['driver_id']:<12} {actual:<8} {ci_str:<10} {diff_str}")

        # --- DETAILED ANALYTICS ---
        valid_preds = final_results.dropna(subset=['target_quali_pos'])

        if not valid_preds.empty:
            mae_overall = valid_preds['abs_diff'].mean()
            exact_matches = len(valid_preds[valid_preds['abs_diff'] == 0])
            close_calls = len(valid_preds[valid_preds['abs_diff'] <= 2])
            total = len(valid_preds)

            # Segmented Performance
            top_3_mae = valid_preds[valid_preds['predicted_pos'] <= 3]['abs_diff'].mean()
            top_10_mae = valid_preds[valid_preds['predicted_pos'] <= 10]['abs_diff'].mean()

            # Confidence interval coverage
            valid_preds = valid_preds.copy()
            valid_preds['in_ci'] = (
                (valid_preds['target_quali_pos'] >= valid_preds['pred_quali_pos_p10']) &
                (valid_preds['target_quali_pos'] <= valid_preds['pred_quali_pos_p90'])
            )
            ci_coverage = valid_preds['in_ci'].mean()

            print("\n" + "="*60)
            print("MODEL PERFORMANCE SUMMARY")
            print("="*60)
            print(f"Global MAE       : {mae_overall:.2f} positions off on average")
            print(f"Top 3 Precision  : {top_3_mae:.2f} MAE (Front row accuracy)")
            print(f"Top 10 Precision : {top_10_mae:.2f} MAE (Q3 accuracy)")
            print("-" * 60)
            print(f"Exact Matches    : {exact_matches}/{total} ({exact_matches/total:.1%})")
            print(f"Close Calls      : {close_calls}/{total} ({close_calls/total:.1%}) (within +/- 2)")
            print(f"90% CI Coverage  : {ci_coverage:.1%} (target: ~90%)")
            print("="*60)

    # Database Upload
    if store_predictions:
        print("\nUploading to Database...")
        final_results = pd.concat(store_predictions)
        db_df = final_results[[
            'race_id', 'driver_id', 'predicted_pos',
            'pred_quali_pos_p10', 'pred_quali_pos_p90',
            'quali_factors_json'
        ]].copy()

        # Convert numpy types to native Python types for psycopg2 compatibility
        for col in ['race_id', 'predicted_pos', 'pred_quali_pos_p10', 'pred_quali_pos_p90']:
            db_df[col] = db_df[col].astype(int)

        data_to_upload = db_df.to_dict(orient='records')

        try:
            with engine.begin() as conn:
                query = text("""
                    UPDATE simulate_predictions
                    SET pred_quali_pos = :predicted_pos,
                        pred_quali_pos_p10 = :pred_quali_pos_p10,
                        pred_quali_pos_p90 = :pred_quali_pos_p90,
                        quali_factors_json = :quali_factors_json
                    WHERE race_id = :race_id AND driver_id = :driver_id
                """)
                for row in data_to_upload:
                    conn.execute(query, row)
            print(f"Upload Success! {len(data_to_upload)} predictions stored.")
        except Exception as e:
            print(f"Upload Failed: {e}")


if __name__ == "__main__":
    predict_quali_outcomes()
