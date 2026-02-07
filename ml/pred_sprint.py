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


def _compute_sprint_track_affinity_with_shrinkage(engine) -> pd.DataFrame:
    """
    Compute track-specific sprint affinity with Bayesian shrinkage.

    For sprints, we measure: positions gained/lost at each track.
    This captures track-specific racecraft (street circuits vs high-speed).

    IMPORTANT: Sprints have even fewer data points than races (~6 tracks/year
    since 2021), making shrinkage MORE critical here.

    Prior weight of 3 means we trust track-specific data once we have 3+ sprints.
    """
    query = text("""
    WITH TrackRaw AS (
        SELECT
            sr.driver_id,
            ra.circuit_name,
            AVG(sr.grid_position - sr.finish_position) as raw_track_avg,
            COUNT(*) as n_sprints
        FROM sprint_results sr
        JOIN races ra ON sr.race_id = ra.race_id
        WHERE ra.year >= 2023
        GROUP BY sr.driver_id, ra.circuit_name
    ),
    GlobalAvg AS (
        SELECT
            driver_id,
            AVG(grid_position - finish_position) as global_avg
        FROM sprint_results sr
        JOIN races ra ON sr.race_id = ra.race_id
        WHERE ra.year >= 2023
        GROUP BY driver_id
    )
    SELECT
        t.driver_id,
        t.circuit_name,
        t.raw_track_avg,
        t.n_sprints,
        g.global_avg,
        -- SHRINKAGE FORMULA: weighted avg toward global mean
        -- Prior weight = 3 (critical for sprints with sparse data)
        ((t.raw_track_avg * t.n_sprints) + (g.global_avg * 3)) / (t.n_sprints + 3)
            as shrunk_sprint_track_affinity
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


def predict_sprint_outcomes():
    print("Fetching data (Safe Mode)...")

    # Fetch Data from DB - only air_temp for Friday runs (no track_temp)
    query = text("""
      WITH PracticePrep AS (
        SELECT
          race_id, driver_id,
          ARRAY_AGG(lap_time ORDER BY lap_time) as best_laps
        FROM practice_laps
        GROUP BY race_id, driver_id
      ),

      WeatherPrep AS (
        SELECT
          race_id,
          AVG(air_temp) as air_temp
        FROM weather
        WHERE session_type LIKE 'FP%'
        GROUP BY race_id
      ),
      MainResults AS (
        SELECT race_id, driver_id, constructor_id, grid_position, finish_position as finish_pos, 'main' as race_type
        FROM results
      ),
      SprintResults AS (
        SELECT race_id, driver_id, constructor_id, grid_position, finish_position as finish_pos, 'sprint' as race_type
        FROM sprint_results
      ),
      Combined AS (
        SELECT * FROM MainResults UNION ALL SELECT * FROM SprintResults
      )
      SELECT
        r.race_id, r.year, r.round, r.circuit_name, c.race_type,
        c.driver_id, c.constructor_id, c.finish_pos, (c.grid_position - c.finish_pos) AS net_pos_gained,

        COALESCE(pred.pred_quali_pos, c.grid_position) as grid,
        p.best_laps,

        w.air_temp

      FROM Combined c
      INNER JOIN races r ON c.race_id = r.race_id

      LEFT JOIN simulate_predictions pred
        ON c.race_id = pred.race_id AND c.driver_id = pred.driver_id

      LEFT JOIN PracticePrep p
        ON c.race_id = p.race_id AND c.driver_id = p.driver_id

      LEFT JOIN WeatherPrep w
        ON c.race_id = w.race_id

      WHERE r.year >= 2023
      ORDER BY r.year, r.round
    """)

    try:
        df = pd.read_sql(query, engine)
        print(f"Data Fetched: {df.shape}")
    except Exception as e:
        print(f"Database Error: {e}")
        return

    # ==================FEATURE ENGINEERING===============
    print("Engineering Features...")

    # 1. PRACTICE LAPS
    df['best_laps'] = df['best_laps'].apply(lambda x: x if isinstance(x, list) else [])
    laps_expanded = pd.DataFrame(df['best_laps'].to_list(), index=df.index)
    laps_expanded = laps_expanded.iloc[:, :7]
    laps_expanded.columns = [f"best_lap_{i+1}" for i in range(laps_expanded.shape[1])]
    df = pd.concat([df, laps_expanded], axis=1)

    new_cols = [f"best_lap_{i+1}" for i in range(7)]
    existing_cols = [c for c in new_cols if c in df.columns]
    df[existing_cols] = df[existing_cols].fillna(120)

    session_best = df.groupby('race_id')['best_lap_1'].transform('min')
    for i in range(1, 8):
        col_name = f"best_lap_{i}"
        if col_name in df.columns:
            df[col_name] = df[col_name] - session_best

    # 2. TEAM MOMENTUM
    team_stats = df.groupby(['race_id', 'constructor_id', 'year', 'round'])['finish_pos'].mean().reset_index()
    team_stats = team_stats.sort_values(by=['year', 'round'])
    team_stats.rename(columns={'finish_pos': 'team_score_current'}, inplace=True)

    for i in range(1, 3):
        col_name = f'constructor_prev_{i}'
        team_stats[col_name] = team_stats.groupby('constructor_id')['team_score_current'].shift(i)
        team_stats[col_name] = team_stats[col_name].fillna(10.0)

    cols_to_merge = ['race_id', 'constructor_id', 'constructor_prev_1', 'constructor_prev_2']
    df = df.merge(team_stats[cols_to_merge], on=['race_id', 'constructor_id'], how='left')

    # 3. TEAMMATE GAP
    teammate_df = df[['race_id', 'constructor_id', 'driver_id', 'finish_pos']].copy()
    teammate_df.columns = ['race_id', 'constructor_id', 'teammate_id', 'teammate_pos']

    df_merged = df.merge(teammate_df, on=['race_id', 'constructor_id'], how='left')
    df_merged = df_merged[df_merged['driver_id'] != df_merged['teammate_id']]
    df_merged = df_merged.drop_duplicates(subset=['race_id', 'driver_id'])

    df_merged['teammate_gap_pos'] = df_merged['finish_pos'] - df_merged['teammate_pos']

    gap_col = df_merged[['race_id', 'driver_id', 'teammate_gap_pos']]
    df = df.merge(gap_col, on=['race_id', 'driver_id'], how='left')
    df['teammate_gap_pos'] = df['teammate_gap_pos'].fillna(0)

    # 4. LAPTIME CONSISTENCY
    lap_time_consistency_query = text("""
      SELECT
        l.race_id,
        l.driver_id,
        STDDEV(l.lap_time) as stdev_laps
      FROM laps l
      INNER JOIN races r ON l.race_id = r.race_id
      GROUP BY l.race_id, l.driver_id
      """)
    try:
        laps_df = pd.read_sql(lap_time_consistency_query, engine)
        races_lookup = df[['race_id', 'year', 'round']].drop_duplicates()
        laps_df = laps_df.merge(races_lookup, on='race_id')
        laps_df = laps_df.sort_values(by=['driver_id', 'year', 'round'])
    except Exception as e:
        print(f"Database Error (Consistency): {e}")
        return

    cols_to_merge = ['race_id', 'driver_id']
    for i in range(1, 4):
        col_name = f'stdev_lag_{i}'
        laps_df[col_name] = laps_df.groupby('driver_id')['stdev_laps'].shift(i)
        laps_df[col_name] = laps_df[col_name].fillna(5.0)
        cols_to_merge.append(col_name)

    df = df.merge(laps_df[cols_to_merge], on=['race_id', 'driver_id'], how='left')

    for i in range(1, 4):
        df[f'stdev_lag_{i}'] = df[f'stdev_lag_{i}'].fillna(5.0)

    # 5. OVERTAKES
    query_overtakes = text("""
      SELECT
        race_id, driver_id, overtakes, SUM(overtakes) OVER (PARTITION BY race_id) as total_overtakes
      FROM driver_fantasy_results
    """)
    query_df = pd.read_sql(query_overtakes, engine)

    for i in range(1, 4):
        col_name = f'prev_overtakes_{i}'
        query_df[col_name] = query_df.groupby('driver_id')['overtakes'].shift(i)
        query_df[col_name] = query_df[col_name].fillna(0)

    df = df.merge(query_df, on=['race_id', 'driver_id'], how="left")
    df['total_overtakes'] = df['total_overtakes'].fillna(0)
    df['overtakes'] = df['overtakes'].fillna(0)

    # 6. DRIVER MOMENTUM
    df = df.sort_values(by=['year', 'round'])
    for i in range(1, 4):
        col_name = f'prev_finish_{i}'
        df[col_name] = df.groupby('driver_id')['finish_pos'].shift(i)
        df[col_name] = df[col_name].fillna(10)

    # 7. TRACK-SPECIFIC AFFINITY WITH BAYESIAN SHRINKAGE
    # Critical for sprints: very few data points per driver-track combo
    print("Computing track-specific sprint affinity with shrinkage...")
    sprint_track_affinity_df = _compute_sprint_track_affinity_with_shrinkage(engine)
    df = df.merge(
        sprint_track_affinity_df[['driver_id', 'circuit_name', 'shrunk_sprint_track_affinity']],
        on=['driver_id', 'circuit_name'],
        how='left'
    )
    # Fill missing (new driver-track combos) with 0 (neutral - no expected gain/loss)
    df['shrunk_sprint_track_affinity'] = df['shrunk_sprint_track_affinity'].fillna(0.0)

    feature_cols = [
        'best_lap_1', 'best_lap_2',
        'constructor_prev_1',
        'prev_finish_1', 'prev_finish_2',
        'teammate_gap_pos',
        'stdev_lag_1', 'stdev_lag_2',
        'prev_overtakes_1',
        'grid',
        'constructor_id',
        'air_temp',  # Only air_temp - track_temp not forecastable 48h out (Friday runs)
        'driver_id',
        'circuit_name',
        'race_type',
        'shrunk_sprint_track_affinity',  # Bayesian-shrunk track-specific performance
    ]

    # ==================PREDICTIVE MODEL===============

    df['constructor_id'] = df['constructor_id'].astype('category')
    df['driver_id'] = df['driver_id'].astype('category')
    df['circuit_name'] = df['circuit_name'].astype('category')
    df['race_type'] = df['race_type'].astype('category')

    # ===== EMPIRICAL CONFIDENCE BANDS FROM HISTORICAL DATA =====
    print("Computing empirical confidence bands from historical residuals...")
    hist_train_mask = (df['year'] < 2025) & (df['finish_pos'].notna())
    X_hist = df.loc[hist_train_mask, feature_cols]
    Y_hist = df.loc[hist_train_mask, 'finish_pos']

    # Use 80/20 split for residual estimation
    split_idx = int(len(X_hist) * 0.8)
    X_cal_train, X_cal_val = X_hist.iloc[:split_idx], X_hist.iloc[split_idx:]
    Y_cal_train, Y_cal_val = Y_hist.iloc[:split_idx], Y_hist.iloc[split_idx:]

    # Train calibration model
    cal_model = xgb.XGBRegressor(
        tree_method='hist',
        enable_categorical=True,
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
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
    sprint_rounds = df[(df['year'] == 2025) & (df['race_type'] == 'sprint')]['round'].unique()

    for r in sorted(sprint_rounds):
        train_mask = (
            ((df['year'] < 2025) | ((df['year'] == 2025) & (df['round'] < r))) &
            (df['finish_pos'].notna())
        )

        test_mask = (df['year'] == 2025) & (df['round'] == r) & (df['race_type'] == 'sprint')

        # Fill weather data for future races using weather_source
        race_ids_test = df.loc[test_mask, 'race_id'].unique()
        for rid in race_ids_test:
            weather = get_race_weather(int(rid))
            # Fill air_temp for rows where it's missing (future races)
            mask = (df['race_id'] == rid) & (df['air_temp'].isna())
            df.loc[mask, 'air_temp'] = weather['air_temp']

        X_train = df.loc[train_mask, feature_cols]
        Y_train = df.loc[train_mask, 'finish_pos']
        X_test = df.loc[test_mask, feature_cols]

        model = xgb.XGBRegressor(
            tree_method='hist',
            enable_categorical=True,
            n_estimators=500,
            learning_rate=0.02,
            max_depth=5,
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
        race_preds['pred_sprint_pos_p10'] = (race_preds['predicted_score'] + ci_lower_bound).clip(1, 20).round().astype(int)
        race_preds['pred_sprint_pos_p90'] = (race_preds['predicted_score'] + ci_upper_bound).clip(1, 20).round().astype(int)

        # Ensure p10 <= p90
        race_preds['pred_sprint_pos_p10'], race_preds['pred_sprint_pos_p90'] = (
            race_preds[['pred_sprint_pos_p10', 'pred_sprint_pos_p90']].min(axis=1),
            race_preds[['pred_sprint_pos_p10', 'pred_sprint_pos_p90']].max(axis=1)
        )

        # Add SHAP explanations
        feature_names = [str(f) for f in feature_cols]
        race_preds['sprint_factors_json'] = [
            _get_shap_top_factors(shap_values[i], feature_names)
            for i in range(len(shap_values))
        ]

        store_predictions.append(race_preds[[
            'race_id', 'year', 'round', 'driver_id', 'grid',
            'predicted_pos', 'pred_sprint_pos_p10', 'pred_sprint_pos_p90',
            'finish_pos', 'sprint_factors_json'
        ]])
        print(f"Round {r} Prediction Complete.")

    # ==================RESULTS & ANALYTICS===============
    if store_predictions:
        final_results = pd.concat(store_predictions)

        # Calculate diffs for stats
        final_results['diff'] = final_results['predicted_pos'] - final_results['finish_pos']
        final_results['abs_diff'] = final_results['diff'].abs()

        print("\n" + "="*60)
        print("SPRINT PREDICTION LOG")
        print("="*60)

        grouped = final_results.groupby(['round'], sort=True)

        for round_num, race_df in grouped:

            print(f"\nROUND {round_num}")
            print(f"{'Pred':<6} {'Driver':<12} {'Grid':<6} {'Actual':<8} {'CI':<10} {'Diff'}")
            print("-" * 65)

            race_df_sorted = race_df.sort_values('predicted_pos')

            for _, row in race_df_sorted.iterrows():
                pred = int(row['predicted_pos'])
                grid = int(row['grid']) if pd.notna(row['grid']) else "-"
                p10 = int(row['pred_sprint_pos_p10'])
                p90 = int(row['pred_sprint_pos_p90'])
                ci_str = f"[{p10}-{p90}]"

                if pd.notna(row['finish_pos']):
                    actual = int(row['finish_pos'])
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

                print(f"{pred:<6} {row['driver_id']:<12} {grid:<6} {actual:<8} {ci_str:<10} {diff_str}")

        # --- DETAILED ANALYTICS ---
        valid_preds = final_results.dropna(subset=['finish_pos'])

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
                (valid_preds['finish_pos'] >= valid_preds['pred_sprint_pos_p10']) &
                (valid_preds['finish_pos'] <= valid_preds['pred_sprint_pos_p90'])
            )
            ci_coverage = valid_preds['in_ci'].mean()

            print("\n" + "="*60)
            print("MODEL PERFORMANCE SUMMARY")
            print("="*60)
            print(f"Global MAE       : {mae_overall:.2f} positions off on average")
            print(f"Top 3 Precision  : {top_3_mae:.2f} MAE (Podium accuracy)")
            print(f"Top 10 Precision : {top_10_mae:.2f} MAE (Points accuracy)")
            print("-" * 60)
            print(f"Exact Matches    : {exact_matches}/{total} ({exact_matches/total:.1%})")
            print(f"Close Calls      : {close_calls}/{total} ({close_calls/total:.1%}) (within +/- 2)")
            print(f"90% CI Coverage  : {ci_coverage:.1%} (target: ~90%)")
            print("="*60)

            sprint_df = df[df['race_type'] == 'sprint']
            valid_finishers = sprint_df[sprint_df['finish_pos'].notna()]

            if not valid_finishers.empty:
                baseline_error = (valid_finishers['grid'] - valid_finishers['finish_pos']).abs().mean()
                print(f"Simply using the Grid gives an error of: {baseline_error:.2f}")

        # 2. Database Upload
        print("\nUploading to Database...")
        db_df = final_results[[
            'race_id', 'driver_id', 'predicted_pos',
            'pred_sprint_pos_p10', 'pred_sprint_pos_p90',
            'sprint_factors_json'
        ]].copy()

        # Convert numpy types to native Python types for psycopg2 compatibility
        for col in ['race_id', 'predicted_pos', 'pred_sprint_pos_p10', 'pred_sprint_pos_p90']:
            db_df[col] = db_df[col].astype(int)

        data_to_upload = db_df.to_dict(orient='records')

        try:
            with engine.begin() as conn:
                query = text("""
                    UPDATE simulate_predictions
                    SET pred_sprint_pos = :predicted_pos,
                        pred_sprint_pos_p10 = :pred_sprint_pos_p10,
                        pred_sprint_pos_p90 = :pred_sprint_pos_p90
                    WHERE race_id = :race_id AND driver_id = :driver_id
                """)
                for row in data_to_upload:
                    conn.execute(query, row)
            print(f"Upload Success! {len(data_to_upload)} predictions stored.")
        except Exception as e:
            print(f"Upload Failed: {e}")

    else:
        print("No predictions were generated.")


if __name__ == "__main__":
    predict_sprint_outcomes()
