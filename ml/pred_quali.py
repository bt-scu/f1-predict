import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import xgboost as xgb

# 1. SETUP DATABASE
DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

def predict_safe_mode():
    print("Fetching data (Safe Mode)...")
    
    # 2. FETCH DATA
    query = text("""
        SELECT
            r.race_id,
            r.year,
            r.round,
            r.circuit_name,
            q.driver_id,
            q.position AS target_quali_pos,
            res.constructor_id,
            p.best_practice_time,
            COALESCE(w.is_wet, 0) as is_wet,
            w.track_temp,
            w.air_temp
        FROM qualifying AS q
        INNER JOIN races AS r ON q.race_id = r.race_id
        INNER JOIN results AS res ON q.race_id = res.race_id AND q.driver_id = res.driver_id
        
        -- WEATHER JOIN (STRICTLY RACE_ID + SESSION_TYPE)
        LEFT JOIN (
            SELECT 
                race_id, 
                MAX(CASE WHEN rainfall = TRUE THEN 1 ELSE 0 END) as is_wet,
                AVG(track_temp) as track_temp,
                AVG(air_temp) as air_temp
            FROM weather
            WHERE session_type = 'Qualifying'
            GROUP BY race_id
        ) AS w ON r.race_id = w.race_id

        -- PRACTICE LAPS JOIN
        LEFT JOIN (
            SELECT
                race_id, driver_id,
                MIN(lap_time) as best_practice_time
            FROM practice_laps
            GROUP BY race_id, driver_id
        ) AS p ON q.race_id = p.race_id AND q.driver_id = p.driver_id
        
        WHERE r.year >= 2023
        ORDER BY r.year ASC, r.round ASC
    """)
    
    try:
        df = pd.read_sql(query, engine)
        #print(df.head())
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return

    # # 3. FEATURE ENGINEERING
    df['best_practice_time'] = df['best_practice_time'].fillna(200) #fill not available
    session_best = df.groupby('race_id')['best_practice_time'].transform('min')
    df['gap_to_best'] = (df['best_practice_time'] - session_best).clip(upper=2.5) # normalize the simulated quali laps based to the best lap (no higher than 3 seconds)
    
    df['practice_rank'] = df.groupby('race_id')['best_practice_time'].rank(ascending=True)
    df['practice_rank'] = df['practice_rank'].fillna(20)
    
    team_pace = df.groupby(['race_id', 'constructor_id'])['best_practice_time'].transform('mean')
    df['teammate_delta'] = df['best_practice_time'] - team_pace

    # print("Calculating Seasonal AVG")
    df['season_avg_pos'] = df.groupby(['year', 'driver_id'])['target_quali_pos'].transform(
      lambda x: x.expanding().mean().shift(1) 
    )
    # df['season_avg_pos'] = df['season_avg_pos'].fillna(10.0)
    
    df = df.sort_values(by=['year', 'driver_id', 'round'])
    #preserve 5 past finishes in the given season
    
    for i in range(1,6):
      col_name = f'quali_lag_{i}'
      df[col_name] = df.groupby(['year', 'driver_id'])['target_quali_pos'].shift(i)
      df[col_name] = df[col_name].fillna(10.0)
    
    
    df['constructor_recent_form'] = df.groupby(['year', 'constructor_id'])['target_quali_pos'].transform(
    lambda x: x.expanding().mean().shift(1) # Simple expanding mean (or use .rolling(3) for stricter recency)
    )
    df['constructor_recent_form'] = df['constructor_recent_form'].fillna(10)
    
    
    df['driver_track_avg'] = df.groupby(['driver_id', 'circuit_name'])['target_quali_pos'].transform(
      lambda x: x.expanding().mean().shift(1)
    )
    df['driver_track_avg'] = df['driver_track_avg'].fillna(10) # Default if new track/driver
  
    #retype certain features for predictive compatability
    df['constructor_id'] = df['constructor_id'].astype('category')
    df['is_wet'] = df['is_wet'].astype('int')
    df['round_cat'] = df['round'].astype('category') 
    
    print(df.head())

    feature_cols = [
        'gap_to_best', 
        'season_avg_pos',     
        'constructor_id', 
        'round_cat',
        'is_wet',
        'quali_lag_1',
        'quali_lag_2',
        'quali_lag_3',
        'quali_lag_4',
        'quali_lag_5',
        'track_temp',
        'air_temp',
        'practice_rank', 
        'teammate_delta',
        'constructor_recent_form', # <--- NEW
        'driver_track_avg'
    ]


    store_predictions = []
    
    for r in sorted(df[df['year'] == 2025]['round'].unique()):
      #we are training on all previous results
      train_mask = (
          ((df['year'] < 2025) | ((df['year'] == 2025) & (df['round'] < r))) & 
          ( df['target_quali_pos'].notna())
      )
      
      #we are testing on only the current race data
      test_mask = (df['year'] == 2025) & (df['round'] == r)
      
      X_train = df.loc[train_mask, feature_cols]
      Y_train = df.loc[train_mask, 'target_quali_pos']
        
      X_test = df.loc[test_mask, feature_cols]
      
      model = xgb.XGBRegressor (
        n_estimators=2000, # of trees
        learning_rate=0.005, # how fast does model learn; variable to control overfittting
        max_depth = 4,
        enable_categorical=True,
        early_stopping_rounds=50, # if you don't improve after # trees stop
        objective='reg:absoluteerror', # the default method is squared error but abs error is less prone to skews
        n_jobs=-1, # use all cpu avaiable
      )
      
      model.fit(
        X_train, Y_train,
        eval_set=[(X_train, Y_train)],
        verbose=False
      )
      
      preds = model.predict(X_test)# should be a list of predictions?
      
      race_preds = df.loc[test_mask].copy()
      race_preds['predicted_score'] = preds
      race_preds['predicted_pos'] = race_preds['predicted_score'].rank(method='first').astype(int) # cool python method to sort the float outputs into integers
      
      store_predictions.append(race_preds[['year', 'round', 'driver_id', 'target_quali_pos', 'predicted_pos', 'is_wet']])
      print(f"Round {r} Prediction Complete.")


    final_results = pd.concat(store_predictions)
      
    
    # 5. RESULTS & ANALYTICS
    if store_predictions:
        final_results = pd.concat(store_predictions)
        
        # Calculate diffs for stats
        final_results['diff'] = final_results['predicted_pos'] - final_results['target_quali_pos']
        final_results['abs_diff'] = final_results['diff'].abs()
        
        print("\n" + "="*60)
        print("🏁 2025 SEASON PREDICTION LOG")
        print("="*60)

        grouped = final_results.groupby(['round'], sort=True)

        for round_num, race_df in grouped:
            race_is_wet = race_df['is_wet'].max() == 1
            weather_icon = "🌧️" if race_is_wet else "☀️"
            
            print(f"\n📍 ROUND {round_num} {weather_icon}")
            print(f"{'P':<4} {'Driver':<8} {'Pred':<6} {'Actual':<6} {'Diff'}")
            print("-" * 45)
            
            race_df_sorted = race_df.sort_values('predicted_pos')
            
            for _, row in race_df_sorted.iterrows():
                pred = int(row['predicted_pos'])
                if pd.notna(row['target_quali_pos']):
                    actual = int(row['target_quali_pos'])
                    diff = int(row['diff'])
                    
                    if diff == 0: diff_str = "✅"
                    elif abs(diff) <= 2: diff_str = f"{diff:+d}"
                    else: diff_str = f"⚠️ {diff:+d}"
                else:
                    actual = "-"
                    diff_str = "🔮"
                
                print(f"{pred:<4} {row['driver_id']:<8} {pred:<6} {actual:<6} {diff_str}")

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
            
            print("\n" + "="*60)
            print("📊 MODEL PERFORMANCE SUMMARY")
            print("="*60)
            print(f"Global MAE       : {mae_overall:.2f} positions off on average")
            print(f"Top 3 Precision  : {top_3_mae:.2f} MAE (Front row accuracy)")
            print(f"Top 10 Precision : {top_10_mae:.2f} MAE (Q3 accuracy)")
            print("-" * 60)
            print(f"✅ Exact Matches : {exact_matches}/{total} ({exact_matches/total:.1%})")
            print(f"🎯 Close Calls   : {close_calls}/{total} ({close_calls/total:.1%}) (within +/- 2)")
            print("="*60)

if __name__ == "__main__":
    predict_safe_mode()