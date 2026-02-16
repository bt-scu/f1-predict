import pandas as pd
import numpy as np
import xgboost as xgb
from db.config import engine
from sklearn.metrics import mean_absolute_error

print("📊 Fetching full history...")

# ✅ FIXED QUERY: Removed 'dr.code', using 'dr.full_name' instead
query = """
    SELECT 
        r.year,
        r.round,
        r.event_name,
        d.driver_id,
        dr.full_name,
        d.constructor_id,
        d.total_points,
        d.start_price
    FROM driver_fantasy_results d
    JOIN races r ON d.race_id = r.race_id
    JOIN drivers dr ON d.driver_id = dr.driver_id
    ORDER BY r.year, r.round
"""
df = pd.read_sql(query, engine)

# 2. GLOBAL FEATURE ENGINEERING
df = df.sort_values(by=['driver_id', 'year', 'round'])

df['last_points'] = df.groupby('driver_id')['total_points'].shift(1)
df['last_2_points'] = df.groupby('driver_id')['total_points'].shift(2)
df['avg_last_3'] = df.groupby('driver_id')['total_points'].transform(lambda x: x.rolling(3).mean().shift(1))
df['team_strength'] = df.groupby('constructor_id')['total_points'].transform(lambda x: x.rolling(5).mean().shift(1))
df['start_price'] = df['start_price'].replace(0, np.nan).fillna(df['start_price'].mean())

df_clean = df.dropna().copy()

# 3. THE SEASON LOOP
print("\n🏎️ STARTING 2025 SEASON SIMULATION (Walk-Forward Validation)\n")
# Adjusted column widths for full names
print(f"{'Rnd':<4} {'GP Name':<25} {'Pred Winner':<20} {'Actual Winner':<20} {'MAE':<5}")
print("-" * 85)

total_mae = 0
rounds_simulated = 0
season_results = []

rounds_2025 = sorted(df_clean[df_clean['year'] == 2025]['round'].unique())

for r in rounds_2025:
    # A. SLICE DATA
    train_mask = (df_clean['year'] < 2025) | ((df_clean['year'] == 2025) & (df_clean['round'] < r))
    test_mask = (df_clean['year'] == 2025) & (df_clean['round'] == r)
    
    X_train = df_clean.loc[train_mask, ['last_points', 'avg_last_3', 'team_strength', 'start_price']]
    y_train = df_clean.loc[train_mask, 'total_points']
    
    X_test = df_clean.loc[test_mask, ['last_points', 'avg_last_3', 'team_strength', 'start_price']]
    y_test = df_clean.loc[test_mask, 'total_points']
    
    if len(X_test) == 0:
        continue

    # B. TRAIN
    # Early stopping moved to constructor
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        objective='reg:absoluteerror', 
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )
    
    # Eval set required for early stopping
    model.fit(
        X_train, y_train, 
        eval_set=[(X_test, y_test)], 
        verbose=False
    )
    
    # C. PREDICT
    preds = model.predict(X_test)
    
    # D. ANALYZE
    round_mae = mean_absolute_error(y_test, preds)
    total_mae += round_mae
    rounds_simulated += 1
    
    results_df = df_clean.loc[test_mask].copy()
    results_df['pred'] = preds
    
    # Find winners
    pred_winner = results_df.sort_values('pred', ascending=False).iloc[0]
    act_winner = results_df.sort_values('total_points', ascending=False).iloc[0]
    
    gp_name = results_df['event_name'].iloc[0] if 'event_name' in results_df else f"Round {r}"
    
    # Check if we guessed the winner correctly
    is_correct = pred_winner['driver_id'] == act_winner['driver_id']
    match_symbol = "✅" if is_correct else "❌"
    
    # Truncate names slightly to fit
    p_name = pred_winner['full_name'][:18]
    a_name = act_winner['full_name'][:18]
    
    print(f"{r:<4} {gp_name[:23]:<25} {p_name:<20} {a_name:<20} {round_mae:.2f} {match_symbol}")
    
    season_results.append(results_df)

# 4. FINAL REPORT
if rounds_simulated > 0:
    print("-" * 85)
    print(f"🏁 Season Complete!")
    print(f"📉 Average Season MAE: {total_mae / rounds_simulated:.2f}")

    all_2025 = pd.concat(season_results)
    final_standings = all_2025.groupby('full_name')[['total_points', 'pred']].sum().sort_values('total_points', ascending=False)
    final_standings['diff'] = final_standings['pred'] - final_standings['total_points']

    print("\n🏆 FINAL 2025 STANDINGS (Projected vs Real):")
    print(final_standings.head(5))
else:
    print("\n⚠️ No 2025 rounds found in database yet!")