import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import xgboost as xgb
import shap
import json


# 1. SETUP DATABASE
DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

# Import weather source for decoupled weather handling
from weather_source import get_race_weather


def _compute_track_affinity_with_shrinkage(engine) -> pd.DataFrame:
    """
    Compute track-specific affinity with Bayesian shrinkage.

    Uses shrinkage toward global mean to handle small sample sizes
    (only ~2-3 races per driver-track combination in 2023-2025 data).

    Prior weight of 3 means we trust track-specific data once we have 3+ races.
    """
    query = text("""
    WITH TrackRaw AS (
        SELECT
            r.driver_id,
            ra.circuit_name,
            AVG(r.grid_position - r.finish_position) as raw_track_avg,
            COUNT(*) as n_races
        FROM results r
        JOIN races ra ON r.race_id = ra.race_id
        WHERE ra.year >= 2023
        GROUP BY r.driver_id, ra.circuit_name
    ),
    GlobalAvg AS (
        SELECT
            driver_id,
            AVG(grid_position - finish_position) as global_avg
        FROM results
        GROUP BY driver_id
    )
    SELECT
        t.driver_id,
        t.circuit_name,
        t.raw_track_avg,
        t.n_races,
        g.global_avg,
        -- SHRINKAGE FORMULA: weighted avg toward global mean
        -- Prior weight = 3 (acts like "3 imaginary races at global avg")
        ((t.raw_track_avg * t.n_races) + (g.global_avg * 3)) / (t.n_races + 3) as shrunk_track_affinity
    FROM TrackRaw t
    JOIN GlobalAvg g ON t.driver_id = g.driver_id
    """)

    return pd.read_sql(query, engine)


def _compute_tire_deg_proxy(engine) -> pd.DataFrame:
    """
    Compute tire degradation proxy from lap data.

    Measures: AVG(last 10 laps) - AVG(first 10 laps)
    Only for races with >= 40 laps (filters out DNFs and short races).
    """
    query = text("""
    WITH LapStats AS (
        SELECT
            l.race_id,
            l.driver_id,
            MAX(l.lap_number) as total_laps,
            AVG(CASE WHEN l.lap_number <= 10 THEN l.lap_time END) as early_stint_avg,
            AVG(CASE
                WHEN l.lap_number > (
                    SELECT MAX(lap_number) - 10
                    FROM laps l2
                    WHERE l2.race_id = l.race_id AND l2.driver_id = l.driver_id
                )
                THEN l.lap_time
            END) as late_stint_avg
        FROM laps l
        GROUP BY l.race_id, l.driver_id
    )
    SELECT
        race_id,
        driver_id,
        total_laps,
        CASE
            WHEN total_laps >= 40 AND early_stint_avg IS NOT NULL AND late_stint_avg IS NOT NULL
            THEN late_stint_avg - early_stint_avg
            ELSE NULL
        END as tire_deg_proxy
    FROM LapStats
    """)

    return pd.read_sql(query, engine)


def _compute_risk_flags(df: pd.DataFrame, engine) -> pd.DataFrame:
    """
    Compute risk flags for LLM explainability.

    Flags computed:
    - rain_sensitive: Position delta in wet vs dry races (null if <2 wet races)
    - first_lap_risk: Avg(lap1_pos - grid_pos) over last 10 races
    - tire_deg_weakness: Percentile rank of tire deg vs field
    - dnf_prone: Rolling 10-race DNF rate
    """
    # Get wet race performance
    wet_query = text("""
    WITH WetRaces AS (
        SELECT DISTINCT w.race_id
        FROM weather w
        WHERE w.rainfall = true AND w.session_type = 'Race'
    ),
    DryRaces AS (
        SELECT DISTINCT r.race_id
        FROM races r
        WHERE r.race_id NOT IN (SELECT race_id FROM WetRaces)
        AND r.year >= 2023
    ),
    WetPerformance AS (
        SELECT
            res.driver_id,
            AVG(res.grid_position - res.finish_position) as wet_delta,
            COUNT(*) as wet_count
        FROM results res
        JOIN WetRaces wr ON res.race_id = wr.race_id
        GROUP BY res.driver_id
    ),
    DryPerformance AS (
        SELECT
            res.driver_id,
            AVG(res.grid_position - res.finish_position) as dry_delta
        FROM results res
        JOIN DryRaces dr ON res.race_id = dr.race_id
        GROUP BY res.driver_id
    )
    SELECT
        w.driver_id,
        w.wet_delta,
        d.dry_delta,
        w.wet_count,
        CASE
            WHEN w.wet_count >= 2 THEN w.wet_delta - d.dry_delta
            ELSE NULL
        END as rain_sensitive
    FROM WetPerformance w
    LEFT JOIN DryPerformance d ON w.driver_id = d.driver_id
    """)

    try:
        rain_df = pd.read_sql(wet_query, engine)
    except Exception:
        rain_df = pd.DataFrame(columns=['driver_id', 'rain_sensitive'])

    # Get first lap risk (from laps table - position change on lap 1)
    first_lap_query = text("""
    WITH FirstLapPositions AS (
        SELECT
            l.race_id,
            l.driver_id,
            l.position as lap1_pos
        FROM laps l
        WHERE l.lap_number = 1
    ),
    GridPositions AS (
        SELECT race_id, driver_id, grid_position
        FROM results
    )
    SELECT
        f.driver_id,
        AVG(f.lap1_pos - g.grid_position) as first_lap_risk
    FROM FirstLapPositions f
    JOIN GridPositions g ON f.race_id = g.race_id AND f.driver_id = g.driver_id
    GROUP BY f.driver_id
    """)

    try:
        first_lap_df = pd.read_sql(first_lap_query, engine)
    except Exception:
        first_lap_df = pd.DataFrame(columns=['driver_id', 'first_lap_risk'])

    # Get DNF rate
    dnf_query = text("""
    SELECT
        driver_id,
        AVG(CASE
            WHEN status ~ '(Retire|Withdraw|Accident|Collision|Damage|Spun)' THEN 1.0
            ELSE 0.0
        END) as dnf_prone
    FROM results
    GROUP BY driver_id
    """)

    try:
        dnf_df = pd.read_sql(dnf_query, engine)
    except Exception:
        dnf_df = pd.DataFrame(columns=['driver_id', 'dnf_prone'])

    # Merge all risk flags
    risk_df = df[['driver_id']].drop_duplicates()
    risk_df = risk_df.merge(rain_df[['driver_id', 'rain_sensitive']], on='driver_id', how='left')
    risk_df = risk_df.merge(first_lap_df, on='driver_id', how='left')
    risk_df = risk_df.merge(dnf_df, on='driver_id', how='left')

    # Tire deg weakness will be computed in main function after tire_deg_proxy is available

    return risk_df


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


def predict_race_outcomes():
    print("Fetching data (Safe Mode)...")

    # Fetch Data from DB
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
          AVG(track_temp) as track_temp,
          AVG(air_temp) as air_temp
        FROM weather
        WHERE session_type LIKE 'FP%'
        GROUP BY race_id
      )

      SELECT
        r.race_id, r.year, r.round, r.circuit_name,
        res.driver_id, res.constructor_id, res.finish_position as finish_pos, (res.grid_position - res.finish_position) AS net_pos_gained,

        COALESCE(pred.pred_quali_pos, res.grid_position) as grid,
        p.best_laps,

        w.track_temp, w.air_temp

      FROM results res
      INNER JOIN races r ON res.race_id = r.race_id
      LEFT JOIN simulate_predictions pred
        ON res.race_id = pred.race_id AND res.driver_id = pred.driver_id

      LEFT JOIN PracticePrep p
        ON res.race_id = p.race_id AND res.driver_id = p.driver_id

      LEFT JOIN WeatherPrep w
        ON res.race_id = w.race_id

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

    # 7. TRACK-SPECIFIC AFFINITY (NEW - with Bayesian shrinkage)
    print("Computing track-specific affinity with shrinkage...")
    track_affinity_df = _compute_track_affinity_with_shrinkage(engine)
    df = df.merge(
        track_affinity_df[['driver_id', 'circuit_name', 'shrunk_track_affinity']],
        on=['driver_id', 'circuit_name'],
        how='left'
    )
    df['shrunk_track_affinity'] = df['shrunk_track_affinity'].fillna(0.0)  # New driver-track combo

    # 8. TIRE DEGRADATION PROXY (NEW)
    print("Computing tire degradation proxy...")
    tire_deg_df = _compute_tire_deg_proxy(engine)
    races_lookup = df[['race_id', 'year', 'round']].drop_duplicates()
    tire_deg_df = tire_deg_df.merge(races_lookup, on='race_id')
    tire_deg_df = tire_deg_df.sort_values(by=['driver_id', 'year', 'round'])

    # Create lagged features for tire deg
    for i in range(1, 4):
        col_name = f'tire_deg_lag_{i}'
        tire_deg_df[col_name] = tire_deg_df.groupby('driver_id')['tire_deg_proxy'].shift(i)
        # Fill with field median (~1.2s is typical)
        tire_deg_df[col_name] = tire_deg_df[col_name].fillna(1.2)

    df = df.merge(
        tire_deg_df[['race_id', 'driver_id', 'tire_deg_lag_1', 'tire_deg_lag_2', 'tire_deg_lag_3']],
        on=['race_id', 'driver_id'],
        how='left'
    )
    for i in range(1, 4):
        df[f'tire_deg_lag_{i}'] = df[f'tire_deg_lag_{i}'].fillna(1.2)

    # 9. RISK FLAGS (for LLM explainability)
    print("Computing risk flags...")
    risk_df = _compute_risk_flags(df, engine)

    # Compute tire deg weakness (percentile rank vs field)
    if 'tire_deg_lag_1' in df.columns:
        tire_deg_ranks = df.groupby('race_id')['tire_deg_lag_1'].rank(pct=True)
        driver_tire_weakness = df.groupby('driver_id').apply(
            lambda x: tire_deg_ranks.loc[x.index].mean()
        ).reset_index()
        driver_tire_weakness.columns = ['driver_id', 'tire_deg_weakness']
        risk_df = risk_df.merge(driver_tire_weakness, on='driver_id', how='left')

    # Store risk flags for later use (will be added to predictions)
    risk_flags_lookup = risk_df.set_index('driver_id').to_dict('index')

    # Fix 4: Add grid-position bucket feature for interaction effects
    # High-grid drivers tend to regress, low-grid drivers tend to gain
    df['grid_bucket'] = pd.cut(
        df['grid'].fillna(10),
        bins=[0, 5, 10, 15, 20],
        labels=['front', 'q2', 'q3_out', 'back']
    ).astype('category')

    feature_cols = [
        'best_lap_1', 'best_lap_2', 'best_lap_3', 'best_lap_4',
        'constructor_prev_1', 'constructor_prev_2',
        'prev_finish_1', 'prev_finish_2',
        'teammate_gap_pos',
        'stdev_lag_1', 'stdev_lag_2', 'stdev_lag_3',
        'prev_overtakes_1', 'prev_overtakes_2', 'prev_overtakes_3',
        'grid',
        'constructor_id',
        'air_temp',  # Only air_temp - track_temp not forecastable 48h out
        'driver_id',
        'circuit_name',
        # NEW FEATURES
        'shrunk_track_affinity',
        'tire_deg_lag_1', 'tire_deg_lag_2', 'tire_deg_lag_3',
        'grid_bucket',  # Fix 4: Grid position interaction
    ]

    # ==================PREDICTIVE MODEL===============

    df['constructor_id'] = df['constructor_id'].astype('category')
    df['driver_id'] = df['driver_id'].astype('category')
    df['circuit_name'] = df['circuit_name'].astype('category')

    store_predictions = []

    # ===== FIX 1: Compute empirical residual bands from historical data =====
    # Train on 2023-2024 to get calibrated CI bounds
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
        early_stopping_rounds=50,
        objective='reg:squarederror',
        n_jobs=-1,
    )
    cal_model.fit(X_cal_train, Y_cal_train, eval_set=[(X_cal_val, Y_cal_val)], verbose=False)

    # Compute residuals on validation set
    cal_preds = cal_model.predict(X_cal_val)
    residuals = Y_cal_val.values - cal_preds

    # Get 5th and 95th percentile of residuals for 90% CI
    ci_lower_bound = np.percentile(residuals, 5) #lower range of prediction 
    ci_upper_bound = np.percentile(residuals, 95) #upper range of prediction
    print(f"  Empirical CI bounds: [{ci_lower_bound:.1f}, {ci_upper_bound:.1f}] positions")

    for r in sorted(df[df['year'] == 2025]['round'].unique()):
        train_mask = (
            ((df['year'] < 2025) | ((df['year'] == 2025) & (df['round'] < r))) &
            (df['finish_pos'].notna())
        )

        test_mask = (df['year'] == 2025) & (df['round'] == r)

        X_train = df.loc[train_mask, feature_cols]
        Y_train = df.loc[train_mask, 'finish_pos']
        X_test = df.loc[test_mask, feature_cols]

        # Get weather source info for this race
        race_ids_test = df.loc[test_mask, 'race_id'].unique()
        weather_sources = {}
        for rid in race_ids_test:
            weather = get_race_weather(rid)
            weather_sources[rid] = 'forecast' if weather['is_forecast'] else 'historical'

        # Fix 3: Single model with improved hyperparameters (no more broken quantile regression)
        model = xgb.XGBRegressor(
            tree_method='hist',
            enable_categorical=True,
            n_estimators=500,
            learning_rate=0.02,  # Fix 3: faster convergence
            max_depth=5,         # Fix 3: more complex interactions
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

        # Build predictions dataframe
        race_preds = df.loc[test_mask].copy()
        race_preds['predicted_score'] = preds
        race_preds['predicted_pos'] = race_preds['predicted_score'].rank(method='first').astype(int)

        # Fix 1: Apply empirical residual-based confidence bands
        # CI bounds are relative to predicted position
        race_preds['pred_race_pos_p10'] = (race_preds['predicted_score'] + ci_lower_bound).clip(1, 20).round().astype(int)
        race_preds['pred_race_pos_p90'] = (race_preds['predicted_score'] + ci_upper_bound).clip(1, 20).round().astype(int)

        # Ensure p10 <= p90 (swap if needed due to negative lower bound)
        race_preds['pred_race_pos_p10'], race_preds['pred_race_pos_p90'] = (
            race_preds[['pred_race_pos_p10', 'pred_race_pos_p90']].min(axis=1),
            race_preds[['pred_race_pos_p10', 'pred_race_pos_p90']].max(axis=1)
        )

        # Add SHAP explanations
        feature_names = [str(f) for f in feature_cols]  # Convert to string for JSON
        race_preds['race_factors_json'] = [
            _get_shap_top_factors(shap_values[i], feature_names)
            for i in range(len(shap_values))
        ]

        # Add risk flags
        race_preds['risk_flags_json'] = race_preds['driver_id'].apply(
            lambda d: json.dumps({
                k: round(float(v), 3) if v is not None and not pd.isna(v) else None
                for k, v in risk_flags_lookup.get(d, {}).items()
            })
        )

        # Add weather source
        race_preds['weather_source'] = race_preds['race_id'].map(weather_sources)

        # Fix 2: DNF-adjusted expected value
        # Weight prediction by survival probability (DNF = effective P20)
        race_preds['dnf_prob'] = race_preds['driver_id'].apply(
            lambda d: risk_flags_lookup.get(d, {}).get('dnf_prone', 0.15)  # Default 15% DNF rate
        )
        race_preds['dnf_prob'] = race_preds['dnf_prob'].fillna(0.15)
        race_preds['ev_race_pos'] = (
            race_preds['predicted_pos'] * (1 - race_preds['dnf_prob']) +
            20 * race_preds['dnf_prob']  # DNF = effective P20
        ).round(1)

        store_predictions.append(race_preds[[
            'race_id', 'year', 'round', 'driver_id', 'grid',
            'predicted_pos', 'pred_race_pos_p10', 'pred_race_pos_p90',
            'ev_race_pos', 'dnf_prob',
            'race_factors_json', 'risk_flags_json', 'weather_source',
            'finish_pos'
        ]])
        print(f"Round {r} Prediction Complete.")

    # ==================RESULTS & ANALYTICS===============
    if store_predictions:
        final_results = pd.concat(store_predictions)

        # Calculate diffs for stats
        final_results['diff'] = final_results['predicted_pos'] - final_results['finish_pos']
        final_results['abs_diff'] = final_results['diff'].abs()

        print("\n" + "="*60)
        print("SEASON PREDICTION LOG")
        print("="*60)

        grouped = final_results.groupby(['round'], sort=True)

        for round_num, race_df in grouped:

            print(f"\nROUND {round_num}")
            print(f"{'Pred':<6} {'Driver':<12} {'Grid':<6} {'Actual':<6} {'CI':<10} {'EV':<6} {'Diff'}")
            print("-" * 75)

            race_df_sorted = race_df.sort_values('predicted_pos')

            for _, row in race_df_sorted.iterrows():
                pred = int(row['predicted_pos'])
                grid = int(row['grid']) if pd.notna(row['grid']) else "-"
                p10 = int(row['pred_race_pos_p10'])
                p90 = int(row['pred_race_pos_p90'])
                ci_str = f"[{p10}-{p90}]"
                ev = row['ev_race_pos']

                if pd.notna(row['finish_pos']):
                    actual = int(row['finish_pos'])
                    diff = int(row['diff'])

                    if diff == 0: diff_str = "OK"
                    elif abs(diff) <= 2: diff_str = f"{diff:+d}"
                    else: diff_str = f"! {diff:+d}"
                else:
                    actual = "-"
                    diff_str = "?"

                print(f"{pred:<6} {row['driver_id']:<12} {grid:<6} {actual:<6} {ci_str:<10} {ev:<6.1f} {diff_str}")

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
            valid_preds['in_ci'] = (
                (valid_preds['finish_pos'] >= valid_preds['pred_race_pos_p10']) &
                (valid_preds['finish_pos'] <= valid_preds['pred_race_pos_p90'])
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

            valid_finishers = df[df['finish_pos'].notna()]
            baseline_error = (valid_finishers['grid'] - valid_finishers['finish_pos']).abs().mean()
            print(f"Simply using the Grid gives an error of: {baseline_error:.2f}")

        # 2. Database Upload
        print("\nUploading to Database...")
        db_df = final_results[[
            'race_id', 'driver_id', 'predicted_pos',
            'pred_race_pos_p10', 'pred_race_pos_p90',
            'ev_race_pos',
            'race_factors_json', 'risk_flags_json', 'weather_source'
        ]].copy()

        # Convert numpy types to native Python types for psycopg2 compatibility
        for col in ['race_id', 'predicted_pos', 'pred_race_pos_p10', 'pred_race_pos_p90']:
            db_df[col] = db_df[col].astype(int)
        db_df['ev_race_pos'] = db_df['ev_race_pos'].astype(float)

        data_to_upload = db_df.to_dict(orient='records')

        try:
            with engine.begin() as conn:
                query = text("""
                    UPDATE simulate_predictions
                    SET pred_race_pos = :predicted_pos,
                        pred_race_pos_p10 = :pred_race_pos_p10,
                        pred_race_pos_p90 = :pred_race_pos_p90,
                        ev_race_pos = :ev_race_pos,
                        race_factors_json = :race_factors_json,
                        risk_flags_json = :risk_flags_json,
                        weather_source = :weather_source
                    WHERE race_id = :race_id AND driver_id = :driver_id
                """)
                for row in data_to_upload:
                    conn.execute(query, row)
            print(f"Upload Success! {len(data_to_upload)} predictions stored.")
        except Exception as e:
            print(f"Upload Failed: {e}")

    else:
        print("No predictions were generated.")


def predict_race_overtakes():
    query = text("""
    WITH PracticeStats AS (
        SELECT 
            race_id, 
            driver_id, 
            AVG(lap_time) as avg_fp2_lap_time,
            MIN(lap_time) as best_fp2_lap_time
        FROM practice_laps 
        WHERE session_type in ('FP1', 'FP2')
        GROUP BY race_id, driver_id
    ),
    OvertakesTarget AS (
        SELECT race_id, driver_id, overtakes 
        FROM driver_fantasy_results
    ),
    WeatherPrep AS (
        SELECT 
            race_id, 
            AVG(track_temp) as avg_track_temp
        FROM weather 
        WHERE session_type LIKE 'FP%' 
        GROUP BY race_id
    ),
    TrackStats AS (
        SELECT race_id, SUM(overtakes) as total_race_overtakes
        FROM driver_fantasy_results 
        GROUP BY race_id
    ),
    GridInfo AS (
        SELECT 
            race.year,
            race.round,
            race.circuit_name,
            
            -- Must COALESCE keys so 2026 rows (which have no 'r' data) get IDs
            COALESCE(r.race_id, sp.race_id) as race_id,
            COALESCE(r.driver_id, sp.driver_id) as driver_id,
            COALESCE(r.constructor_id, sp.constructor_id) as constructor_id,
            CASE 
                WHEN race.year > 2025 THEN sp.pred_quali_pos
                ELSE r.grid_position 
            END as start_grid,

            r.status,
            r.finish_position as finish_pos

        FROM results r
        FULL OUTER JOIN simulate_predictions sp 
            ON r.race_id = sp.race_id AND r.driver_id = sp.driver_id
            
        INNER JOIN races race 
            ON COALESCE(r.race_id, sp.race_id) = race.race_id
            
        WHERE race.year >= 2023
    )
    SELECT
        gi.year,
        gi.round,
        p.driver_id,
        p.race_id,
        ot.overtakes,
        gi.start_grid,
        gi.finish_pos,
        gi.constructor_id,
        gi.circuit_name,
        p.avg_fp2_lap_time,
        p.best_fp2_lap_time,
        wp.avg_track_temp,
        ts.total_race_overtakes
    FROM PracticeStats p
    JOIN GridInfo gi ON p.race_id = gi.race_id AND p.driver_id = gi.driver_id
    LEFT JOIN OvertakesTarget ot ON p.driver_id = ot.driver_id AND p.race_id = ot.race_id
    LEFT JOIN WeatherPrep wp ON p.race_id = wp.race_id
    LEFT JOIN TrackStats ts ON p.race_id = ts.race_id
""")
    
    try:
        print("🚀 Script started...")
        df = pd.read_sql(query, engine)
        print(df.head())
    except Exception as e:
        print(f"❌ Error: {e}")

    # FEATURE ENGINEERING
    team_pace_stats = df.groupby(['race_id', 'constructor_id'])['avg_fp2_lap_time'].mean().reset_index()
    team_pace_stats.rename(columns={'avg_fp2_lap_time': 'team_avg_pace'}, inplace=True)
    df_merged = df.merge(team_pace_stats, on=['race_id', 'constructor_id'], how='left')
    
    df_merged['practice_pace_gap'] = df_merged['avg_fp2_lap_time'] - df_merged['team_avg_pace']
    
    df_merged['practice_pace_gap'] = df_merged['practice_pace_gap'].fillna(0)

    # 2. Rolling Features
    df_merged = df_merged.sort_values(by=['year', 'round'])
    
    for i in range(1, 4):
        col_name = f"rolling_overtakes_{i}"
        col_name_2 = f"aggresion_score_{i}"
        
        # Shift within driver history
        df_merged[col_name] = df_merged.groupby(['driver_id'])['overtakes'].shift(i)
        
        # Calculate Aggression Score
        denom = (df_merged['start_grid'] - df_merged['finish_pos'] + 1)
        denom = denom.replace(0, 1)
        
        df_merged[col_name_2] = df_merged[col_name] / denom
        
        df_merged[col_name] = df_merged[col_name].fillna(0.0)
        df_merged[col_name_2] = df_merged[col_name_2].fillna(0.0)
        df_merged[col_name_2] = df_merged[col_name_2].replace([np.inf, -np.inf], 0.0)
      
    # 3. Track Baseline
    track_baseline = df_merged.groupby('circuit_name')['total_race_overtakes'].mean().reset_index()
    track_baseline.columns = ['circuit_name', 'avg_circuit_overtakes']
    df_merged = df_merged.merge(track_baseline, on='circuit_name', how='left')
    
    # Fill missing track data for new/renamed tracks
    global_avg = df_merged['total_race_overtakes'].mean()
    df_merged['avg_circuit_overtakes'] = df_merged['avg_circuit_overtakes'].fillna(global_avg)
    
    
    feature_cols = [
        'rolling_overtakes_1', 'rolling_overtakes_2', 'rolling_overtakes_3',
        'aggresion_score_1', 'aggresion_score_2', 'aggresion_score_3',
        'practice_pace_gap',
        'avg_fp2_lap_time',
        'best_fp2_lap_time',
        'avg_track_temp', 
        'start_grid',
        'constructor_id',
        'driver_id',
        'avg_circuit_overtakes',
                
    ]

# ==================PREDICTIVE MODEL===============
    df_merged['constructor_id'] = df_merged['constructor_id'].astype('category')
    df_merged['driver_id'] = df_merged['driver_id'].astype('category')
    df_merged['circuit_name'] = df_merged['circuit_name'].astype('category')
    
    store_predictions = []
    
    for r in sorted(df_merged[df_merged['year'] == 2025]['round'].unique()):
        train_mask = (
            ((df_merged['year'] < 2025) | ((df_merged['year'] == 2025) & (df_merged['round'] < r))) & 
            (df_merged['overtakes'].notna())
        )
        
        test_mask = (df_merged['year'] == 2025) & (df_merged['round'] == r)
        
        X_train = df_merged.loc[train_mask, feature_cols]
        Y_train = df_merged.loc[train_mask, 'overtakes']
        X_test = df_merged.loc[test_mask, feature_cols]
      
        model = xgb.XGBRegressor(
            objective='count:poisson', #good for count data such as # of occurence        
            enable_categorical=True,  
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=4
        )
      
        model.fit(
            X_train, Y_train,
            eval_set=[(X_train, Y_train)],
            verbose=False
        )
      
        preds = model.predict(X_test)
      
        overtakes_preds = df_merged.loc[test_mask].copy()
        overtakes_preds['predicted_overtakes'] = np.round(np.maximum(preds, 0)).astype(int)
        store_predictions.append(overtakes_preds[['race_id', 'year', 'round', 'driver_id', 'predicted_overtakes']])
    
    final_results = pd.concat(store_predictions)

    final_results = final_results.merge(
        df[['race_id', 'driver_id', 'overtakes']], 
        on=['race_id', 'driver_id'], 
        how='left'
    )

    final_results = final_results.rename(columns={'overtakes': 'actual_overtakes'})
    
    valid_preds = final_results.dropna(subset=['actual_overtakes'])
    
    if not valid_preds.empty:
        valid_preds['abs_diff'] = abs(valid_preds['predicted_overtakes'] - valid_preds['actual_overtakes'])
        
        mae_overall = valid_preds['abs_diff'].mean()
        exact_matches = len(valid_preds[valid_preds['abs_diff'] == 0])
        close_calls = len(valid_preds[valid_preds['abs_diff'] <= 2]) # Within 2 overtakes is solid
        total = len(valid_preds)
        
        low_action_mae = valid_preds[valid_preds['predicted_overtakes'] <= 2]['abs_diff'].mean()
        high_action_mae = valid_preds[valid_preds['predicted_overtakes'] >= 5]['abs_diff'].mean()
        
        print("\n" + "="*60)
        print("📊 OVERTAKE MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Global MAE       : {mae_overall:.2f} overtakes off on average")
        print("-" * 60)
        print(f"Low Action MAE   : {low_action_mae:.2f} (When predicting <= 2 moves)")
        print(f"High Action MAE  : {high_action_mae:.2f} (When predicting >= 5 moves)")
        print("-" * 60)
        print(f"✅ Exact Matches : {exact_matches}/{total} ({exact_matches/total:.1%})")
        print(f"🎯 Close Calls   : {close_calls}/{total} ({close_calls/total:.1%}) (within +/- 2)")
        print("="*60)

    print("\nUploading to Database...")
    
    db_df = final_results[['race_id', 'driver_id', 'predicted_overtakes']].copy()
    data_to_upload = db_df.to_dict(orient='records')
    
    try:
        with engine.begin() as conn:  
            query = text("""
                UPDATE simulate_predictions
                SET pred_race_overtakes = :predicted_overtakes
                WHERE race_id = :race_id AND driver_id = :driver_id
            """)
        
            for row in data_to_upload:
                conn.execute(query, row)
                
        print(f"✅ Upload Success! {len(data_to_upload)} predictions stored in 'pred_race_overtakes'.")
        
    except Exception as e:
        print(f"❌ Upload Failed: {e}")

def predict_race_dnfs():
    query = text("""
    WITH 
    -- 1. Get Target Data (Overtakes)
    OvertakesTarget AS (
        SELECT race_id, driver_id, constructor_id, overtakes 
        FROM driver_fantasy_results
    ),
    -- 2. Get Features (Grid, Status, etc.)
    GridInfo AS (
        SELECT 
            race.year,
            race.round,
            race.circuit_name,
            race.race_id,
            COALESCE(r.driver_id, sp.driver_id) as driver_id,
            COALESCE(r.constructor_id, sp.constructor_id) as constructor_id,
            
            -- GRID LOGIC: Use Prediction if Future, Real Grid if Past
            CASE 
                WHEN race.year > 2025 THEN sp.pred_quali_pos
                ELSE r.grid_position 
            END as start_grid,

            -- CRITICAL: We need this column for your Python 'is_dnf' logic!
            r.status 

        FROM races race
        LEFT JOIN results r ON race.race_id = r.race_id
        LEFT JOIN simulate_predictions sp 
            ON race.race_id = sp.race_id 
            AND (r.driver_id IS NULL OR r.driver_id = sp.driver_id)
        WHERE race.year >= 2021
    )

    -- 3. Select Final Columns
    SELECT 
        g.race_id,
        g.year,
        g.circuit_name,
        g.round,
        g.driver_id,
        g.start_grid,
        g.status, 
        ot.constructor_id,
        COALESCE(ot.overtakes, 0) as overtakes
        
    FROM GridInfo g
    LEFT JOIN OvertakesTarget ot 
           ON g.race_id = ot.race_id 
          AND g.driver_id = ot.driver_id
    WHERE g.driver_id IS NOT NULL
    ORDER BY g.year, g.round
    """)

    
    try:
        print("🚀 Script started...")
        df = pd.read_sql(query, engine)
        print(df.head())
    except Exception as e:
        print(f"❌ Error: {e}")

    #Feature Engineering
    keywords = ["Retire", "Withdraw", "Accident", "Collision", "Damage", "Spun"]
    pattern = '|'.join(keywords)
    df['is_dnf'] = df['status'].str.contains(pattern, case=False, na=False).astype(int)
    
    df = df.sort_values(by=['year', 'round'])
    
    for i in range(1, 11):
        col_name = f"dnf_{i}_races_ago"
        df[col_name] = df.groupby('driver_id')['is_dnf'].shift(i)
        df[col_name] = df[col_name].fillna(0.0)
    
    for i in range(1, 4):
        col_name = f"rolling_overtakes_{i}"
        df[col_name] = df.groupby('driver_id')['overtakes'].shift(i)
        df[col_name] = df[col_name].fillna(0.0)
    
    feature_cols = [
        'constructor_id',
        'circuit_name',
        'start_grid',
        'dnf_1_races_ago',
        'dnf_2_races_ago',
        'dnf_3_races_ago',
        'rolling_overtakes_1',
        'rolling_overtakes_2',
        'rolling_overtakes_3'
    ]
    df['driver_id'] = df['driver_id'].astype('category')
    df['circuit_name'] = df['circuit_name'].astype('category')
    df['constructor_id'] = df['constructor_id'].astype('category')
    
    store_predictions = []

    for r in sorted(df[df['year'] == 2025]['round'].unique()):
        train_mask = (
            ((df['year'] < 2025) | ((df['year'] == 2025) & (df['round'] < r))) & 
            (df['is_dnf'].notna())
        )

        test_mask = (df['year'] == 2025) & (df['round'] == r)
          
        X_train = df.loc[train_mask, feature_cols]
        Y_train = df.loc[train_mask, 'is_dnf']
        X_test = df.loc[test_mask, feature_cols]
      
        model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=5,
            scale_pos_weight=2,
            eval_metric='logloss',
            enable_categorical=True,
            tree_method="hist"
        )
      
        model.fit(
            X_train, Y_train,
            eval_set=[(X_train, Y_train)],
            verbose=False
        )
        
        probs = model.predict_proba(X_test)[:, 1]
        race_preds = df.loc[test_mask].copy()
        race_preds['dnf_probability'] = probs
        race_preds['predicted_dnf'] = (probs > 0.6).astype(int)
    
        store_predictions.append(race_preds[[
            'race_id', 'year', 'round', 'driver_id', 'start_grid', 'dnf_probability', 'predicted_dnf', 'is_dnf'
        ]])
        print(f"✅ Round {r} Prediction Complete.")

    final_results = pd.concat(store_predictions)
    
    # 1. FIX THE TYPO HERE: Add 'final_results' before the column name
    final_results['error'] = (final_results['dnf_probability'] - final_results['is_dnf']).abs()
    
    # "Brier Score" (Average Error) - Lower is better
    avg_error = final_results['error'].mean()
    
    # specialized metric: How good are we at spotting crashes?
    # Filter for the races where a driver ACTUALLY DNF'd
    actual_dnfs = final_results[final_results['is_dnf'] == 1]
    avg_risk_assigned_to_crashes = actual_dnfs['dnf_probability'].mean()
    
    # Filter for races where a driver FINISHED
    actual_finishers = final_results[final_results['is_dnf'] == 0]
    avg_risk_assigned_to_finishers = actual_finishers['dnf_probability'].mean()

    print("\n" + "="*60)
    print("🚑 DNF MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Global Error (Brier Score): {avg_error:.4f} (Lower is better)")
    print("-" * 60)
    print(f"Avg Risk Assigned to VICTIMS    : {avg_risk_assigned_to_crashes:.1%} (Should be High)")
    print(f"Avg Risk Assigned to SURVIVORS  : {avg_risk_assigned_to_finishers:.1%} (Should be Low)")
    print("-" * 60)
    
    # "Top Risk" check: Did the person with the highest risk actually crash?
    riskiest_predictions = final_results.sort_values('dnf_probability', ascending=False).head(20)
    correct_calls = riskiest_predictions['is_dnf'].sum()
    print(f"🎯 Out of the top 20 riskiest predictions, {int(correct_calls)} actually crashed.")
    print("="*60)

    print("\nUploading to Database...")
    
    # Prepare for upload
    db_df = final_results[['race_id', 'driver_id', 'dnf_probability']].copy()
    data_to_upload = db_df.to_dict(orient='records')
    
    importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Create a simple table
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    
    print(feat_imp)
    print("="*60)
    
    try:
        with engine.begin() as conn:  
            # Make sure your 'simulate_predictions' table has a 'pred_dnf_prob' column!
            # If not, run: ALTER TABLE simulate_predictions ADD COLUMN pred_dnf_prob FLOAT;
            query = text("""
                UPDATE simulate_predictions
                SET pred_dnf_prob = :dnf_probability
                WHERE race_id = :race_id AND driver_id = :driver_id
            """)
        
            for row in data_to_upload:
                conn.execute(query, row)
                
        print(f"✅ Upload Success! {len(data_to_upload)} probabilities stored.")
        
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
    
    
    
if __name__ == "__main__":
    predict_race_outcomes()
    predict_race_overtakes()
    predict_race_dnfs()