import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import xgboost as xgb

# 1. SETUP DATABASE
DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

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
        
        pred.pred_quali_pos as grid, 
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
        print(f"✅ Data Fetched: {df.shape}")
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return

    # ==================FEATURE ENGINEERING===============
    print("Engineering Features...")
    
    field_avg_pace = df['avg_fp2_lap_time'].mean()
    df['avg_fp2_lap_time'] = df['avg_fp2_lap_time'].fillna(field_avg_pace)
    df['best_fp2_lap_time'] = df['best_fp2_lap_time'].fillna(df['best_fp2_lap_time'].mean())
    
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
        print(f"❌ Database Error (Consistency): {e}")
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
      
    feature_cols = [
        'best_lap_1', 'best_lap_2', 'best_lap_3', 'best_lap_4', 
        'constructor_prev_1', 'constructor_prev_2', 
        'prev_finish_1', 'prev_finish_2',       
        'teammate_gap_pos',    
        'stdev_lag_1', 'stdev_lag_2', 'stdev_lag_3',        
        'prev_overtakes_1', 'prev_overtakes_2', 'prev_overtakes_3',       
        'grid',               
        'constructor_id',
        'track_temp',
        'air_temp',
        'driver_id' ,
        'circuit_name'       
    ]

    # ==================PREDICTIVE MODEL===============
    
    df['constructor_id'] = df['constructor_id'].astype('category')
    df['driver_id'] = df['driver_id'].astype('category')
    df['circuit_name'] = df['circuit_name'].astype('category')
    
    store_predictions = []

    for r in sorted(df[df['year'] == 2025]['round'].unique()):
        train_mask = (
            ((df['year'] < 2025) | ((df['year'] == 2025) & (df['round'] < r))) & 
            (df['finish_pos'].notna())
        )

        test_mask = (df['year'] == 2025) & (df['round'] == r)
          
        X_train = df.loc[train_mask, feature_cols]
        Y_train = df.loc[train_mask, 'finish_pos']
        X_test = df.loc[test_mask, feature_cols]
      
        model = xgb.XGBRegressor(
            tree_method='hist',        
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
      
        race_preds = df.loc[test_mask].copy()
        race_preds['predicted_score'] = preds
        race_preds['predicted_pos'] = race_preds['predicted_score'].rank(method='first').astype(int) 
      
        store_predictions.append(race_preds[['race_id','year', 'round', 'driver_id', 'grid', 'predicted_pos', 'finish_pos']])
        print(f"✅ Round {r} Prediction Complete.")

    # ==================RESULTS & ANALYTICS===============
    if store_predictions:
        final_results = pd.concat(store_predictions)
        
        # Calculate diffs for stats
        # We compare predicted_pos vs finish_pos (Actual Race Result)
        final_results['diff'] = final_results['predicted_pos'] - final_results['finish_pos']
        final_results['abs_diff'] = final_results['diff'].abs()
        
        print("\n" + "="*60)
        print("🏁 SEASON PREDICTION LOG")
        print("="*60)

        grouped = final_results.groupby(['round'], sort=True)

        for round_num, race_df in grouped:
            
            print(f"\n📍 ROUND {round_num}")
            print(f"{'Pred':<6} {'Driver':<12} {'Grid':<6} {'Actual':<6} {'Diff'}")
            print("-" * 55)
            
            race_df_sorted = race_df.sort_values('predicted_pos')
            
            for _, row in race_df_sorted.iterrows():
                pred = int(row['predicted_pos'])
                grid = int(row['grid']) if pd.notna(row['grid']) else "-"
                
                if pd.notna(row['finish_pos']):
                    actual = int(row['finish_pos'])
                    diff = int(row['diff'])
                    
                    if diff == 0: diff_str = "✅"
                    elif abs(diff) <= 2: diff_str = f"{diff:+d}"
                    else: diff_str = f"⚠️ {diff:+d}"
                else:
                    actual = "-"
                    diff_str = "🔮"
                
                print(f"{pred:<6} {row['driver_id']:<12} {pred:<6} {actual:<6} {diff_str}")

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
            
            print("\n" + "="*60)
            print("📊 MODEL PERFORMANCE SUMMARY")
            print("="*60)
            print(f"Global MAE       : {mae_overall:.2f} positions off on average")
            print(f"Top 3 Precision  : {top_3_mae:.2f} MAE (Podium accuracy)")
            print(f"Top 10 Precision : {top_10_mae:.2f} MAE (Points accuracy)")
            print("-" * 60)
            print(f"✅ Exact Matches : {exact_matches}/{total} ({exact_matches/total:.1%})")
            print(f"🎯 Close Calls   : {close_calls}/{total} ({close_calls/total:.1%}) (within +/- 2)")
            print("="*60)

        # 2. Database Upload
        print("\nUploading to Database...")
        db_df = final_results[['race_id', 'driver_id', 'predicted_pos']].copy()
        data_to_upload = db_df.to_dict(orient='records')
        
        try:
            with engine.begin() as conn:  
                query = text("""
                    UPDATE simulate_predictions
                    SET pred_race_pos = :predicted_pos
                    WHERE race_id = :race_id AND driver_id = :driver_id
                """)
                for row in data_to_upload:
                    conn.execute(query, row)
            print("✅ Upload Success! Predictions stored in 'pred_race_pos'.")
        except Exception as e:
            print(f"❌ Upload Failed: {e}")
            
    else:
        print("⚠️ No predictions were generated.")


def predict_overtakes():
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
            COALESCE(r.constructor_id, sp.constructor_id) as constructor_id,
            COALESCE(sp.race_id, r.race_id) as race_id,
            COALESCE(sp.driver_id, r.driver_id) as driver_id,
            COALESCE(sp.pred_quali_pos, r.grid_position) as start_grid,
            r.finish_position as finish_pos
        FROM results r
        FULL OUTER JOIN simulate_predictions sp 
            ON r.race_id = sp.race_id AND r.driver_id = sp.driver_id
        INNER JOIN races race ON COALESCE(r.race_id, sp.race_id) = race.race_id
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
    # 1. ROBUST PACE GAP (The Fix)
    # Calculate the average pace for the TEAM for that specific race
    team_pace_stats = df.groupby(['race_id', 'constructor_id'])['avg_fp2_lap_time'].mean().reset_index()
    team_pace_stats.rename(columns={'avg_fp2_lap_time': 'team_avg_pace'}, inplace=True)
    
    # Merge back to the main dataframe
    df_merged = df.merge(team_pace_stats, on=['race_id', 'constructor_id'], how='left')
    
    # Calculate gap: (My Pace) - (Team Average)
    # If I am faster than average, this will be negative (Good). Slower = positive.
    df_merged['practice_pace_gap'] = df_merged['avg_fp2_lap_time'] - df_merged['team_avg_pace']
    
    # Fill any remaining NaNs (e.g. if the whole team has no times) with 0
    df_merged['practice_pace_gap'] = df_merged['practice_pace_gap'].fillna(0)

    # 2. Rolling Features
    df_merged = df_merged.sort_values(by=['year', 'round'])
    
    for i in range(1, 4):
        col_name = f"rolling_overtakes_{i}"
        col_name_2 = f"aggresion_score_{i}"
        
        # Shift within driver history
        df_merged[col_name] = df_merged.groupby(['driver_id'])['overtakes'].shift(i)
        
        # Calculate Aggression Score
        # Add safety to denominator to prevent DivisionByZero
        denom = (df_merged['start_grid'] - df_merged['finish_pos'] + 1)
        denom = denom.replace(0, 1) # Safety check
        
        df_merged[col_name_2] = df_merged[col_name] / denom
        
        # Clean up
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

    # 1. Re-attach 'actual_overtakes' from the main df so we can check accuracy
    #    (We assume 'df' still has the original data)
    final_results = final_results.merge(
        df[['race_id', 'driver_id', 'overtakes']], 
        on=['race_id', 'driver_id'], 
        how='left'
    )
    
    # Rename for clarity
    final_results = final_results.rename(columns={'overtakes': 'actual_overtakes'})
    
    valid_preds = final_results.dropna(subset=['actual_overtakes'])
    
    if not valid_preds.empty:
        # Calculate the error (absolute difference)
        valid_preds['abs_diff'] = abs(valid_preds['predicted_overtakes'] - valid_preds['actual_overtakes'])
        
        mae_overall = valid_preds['abs_diff'].mean()
        exact_matches = len(valid_preds[valid_preds['abs_diff'] == 0])
        close_calls = len(valid_preds[valid_preds['abs_diff'] <= 2]) # Within 2 overtakes is solid
        total = len(valid_preds)
        
        # Segmented Performance: How good are we at predicting "Boring" vs "Exciting" races?
        # Low Action (Predicted <= 2 overtakes)
        low_action_mae = valid_preds[valid_preds['predicted_overtakes'] <= 2]['abs_diff'].mean()
        # High Action (Predicted >= 5 overtakes)
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

    # 2. Database Upload (Targeting 'pred_race_overtakes')
    print("\nUploading to Database...")
    
    # Prepare the data for upload
    db_df = final_results[['race_id', 'driver_id', 'predicted_overtakes']].copy()
    data_to_upload = db_df.to_dict(orient='records')
    
    try:
        with engine.begin() as conn:  
            query = text("""
                UPDATE simulate_predictions
                SET pred_race_overtakes = :predicted_overtakes
                WHERE race_id = :race_id AND driver_id = :driver_id
            """)
            
            # Execute in bulk is faster, but loop works fine for small datasets
            for row in data_to_upload:
                conn.execute(query, row)
                
        print(f"✅ Upload Success! {len(data_to_upload)} predictions stored in 'pred_race_overtakes'.")
        
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
    
    
    
    
    


if __name__ == "__main__":
    # predict_race_outcomes()
    predict_overtakes()