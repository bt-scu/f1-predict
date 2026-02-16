import pandas as pd
import os
import sys
from sqlalchemy import text
from db.config import engine
from db.ingestion.helper.get_race_id_map import get_race_id_map
from collections import defaultdict, deque

# --- MAPPINGS ---
# Maps CSV abbreviations to Database Constructor IDs
TEAM_NAME_MAP = {
    'MCL': 'McLaren', 'McLaren': 'McLaren', 
    'RED': 'Red Bull Racing', 'Red Bull': 'Red Bull Racing', 'RBR': 'Red Bull Racing',
    'MER': 'Mercedes', 'Mercedes': 'Mercedes',
    'FER': 'Ferrari', 'Ferrari': 'Ferrari',
    'WIL': 'Williams', 'Williams': 'Williams',
    'HAA': 'Haas F1 Team', 'Haas': 'Haas F1 Team',
    'VRB': 'RB', 'RB': 'RB', 'Racing Bulls': 'RB', 'AlphaTauri': 'RB', 'AT': 'RB',
    'AST': 'Aston Martin', 'Aston Martin': 'Aston Martin',
    'KCK': 'Kick Sauber', 'Kick Sauber': 'Kick Sauber', 'Sauber': 'Kick Sauber', 
    'ALP': 'Alpine', 'Alpine': 'Alpine'
}

# Maps CSV abbreviations to Database Driver IDs
DRIVER_CODE_MAP = {
    'Ver': 'VER', 'Nor': 'NOR', 'Pia': 'PIA', 'Lec': 'LEC', 'Ham': 'HAM',
    'Rus': 'RUS', 'Law': 'LAW', 'Ant': 'ANT', 'Tsu': 'TSU', 'Alb': 'ALB',
    'Str': 'STR', 'Sai': 'SAI', 'Oco': 'OCO', 'Bea': 'BEA', 'Hul': 'HUL',
    'Gas': 'GAS', 'Alo': 'ALO', 'Had': 'HAD', 'Doo': 'DOO', 'Col': 'COL',
    'Bor': 'BOR', 'Zho': 'ZHO', 'Bot': 'BOT', 'Mag': 'MAG', 'Ric': 'RIC', 
    'Sar': 'SAR', 'Dev': 'DEV', 'Per': 'PER', 'Sai': 'SAI'
}

def ingest_2025_prices(driver_file, constructor_file):
    print(f"\n🔵 Processing 2025 Exact Prices...")
    
    with engine.begin() as conn:
        race_map = get_race_id_map(conn)
        try:
            d_df = pd.read_csv(driver_file)
            print(f"   ... Processing Drivers ({len(d_df)} rows)")
            
            for _, row in d_df.iterrows():
                try:
                    rnd = int(row['Round'])
                    price = float(row['Start_Price'])
                    raw_code = str(row['Driver']).strip().capitalize()
                    d_code = DRIVER_CODE_MAP.get(raw_code, raw_code.upper())
                    
                    race_id = race_map.get((2025, rnd))
                    if race_id:
                        conn.execute(text("""
                            UPDATE driver_fantasy_results 
                            SET start_price = :p 
                            WHERE race_id = :rid AND driver_id = :d
                        """), {"p": price, "rid": race_id, "d": d_code})
                except Exception: continue
        except Exception as e:
            print(f"   ⚠️ Driver file skipped: {e}")
        
        try:
            c_df = pd.read_csv(constructor_file)
            print(f"   ... Processing Constructors ({len(c_df)} rows)")
            
            for _, row in c_df.iterrows():
                try:
                    rnd = int(row['Round'])
                    price = float(row['Start_Price'])
                    raw_team = row['Constructor'].strip()
                    c_name = TEAM_NAME_MAP.get(raw_team, TEAM_NAME_MAP.get(raw_team.upper(), raw_team))

                    race_id = race_map.get((2025,rnd))
                    if race_id:
                         conn.execute(text("""
                            UPDATE constructor_fantasy_results 
                            SET start_price = :p 
                            WHERE race_id = :rid AND constructor_id = :c
                        """), {"p": price, "rid": race_id, "c": c_name})
                except Exception: continue
        except Exception as e:
             print(f"   ⚠️ Constructor file skipped: {e}")
    
    print("   ✅ 2025 Data Ingested.")

# --- HELPER: Dynamic Thresholds (The "Ramp Up") ---
def get_price_thresholds(round_num):
    if round_num == 1:
        return (0.2, 0.3, 0.4) 
    
    elif round_num == 2:
        return (0.4, 0.6, 0.8) 
    else:
        return (0.6, 0.9, 1.2)

def calculate_change(avg_ppm, current_price, thresholds):
    low, mid, high = thresholds
    
    is_tier_a = current_price >= 19.0
    
    # 2. Determine Rating
    if avg_ppm >= high:
        rating = "GREAT"
    elif avg_ppm >= mid:
        rating = "GOOD"
    elif avg_ppm >= low:
        rating = "POOR"
    else:
        rating = "TERRIBLE"
        
    # 3. Lookup Change in Matrix
    if is_tier_a:
        if rating == "GREAT": return 0.3
        if rating == "GOOD": return 0.1
        if rating == "POOR": return -0.1
        return -0.3
    else: 
        if rating == "GREAT": return 0.6
        if rating == "GOOD": return 0.2
        if rating == "POOR": return -0.2
        return -0.6

def simulate_historical_prices(driver_start_file, constructor_start_file):
    print("🔄 Starting 2025 Price Algorithm Simulation...")
    
    # --- 1. SETUP: LOAD STATE ---
    d_start_df = pd.read_csv(driver_start_file)
    c_start_df = pd.read_csv(constructor_start_file)
    
    # Replacement Logic (The "Seat Map")
    replacements = {}
    if 'replaces_driver_code' in d_start_df.columns:
        for _, row in d_start_df.iterrows():
            if pd.notna(row['replaces_driver_code']):
                # Key: (Driver, Team) -> Value: Parent Driver
                key = (row['driver_code'].strip().upper(), row['team_id'].strip())
                replacements[key] = row['replaces_driver_code'].strip().upper()

    # Price Ledger (Tracks current price in memory)
    current_driver_prices = {}
    for _, row in d_start_df.iterrows():
        current_driver_prices[row['driver_code'].strip().upper()] = float(row['start_price'])

    current_team_prices = {}
    for _, row in c_start_df.iterrows():
        t_id = row['constructor_id'].strip()
        # Safe map for team names if you have a mapping dict, otherwise use ID
        c_name = TEAM_NAME_MAP.get(t_id, t_id) if 'TEAM_NAME_MAP' in globals() else t_id
        current_team_prices[c_name] = float(row['start_price'])

    # Points Ledger (The Rolling Memory)
    # Stores [Score1, Score2, Score3]. Starts with [0.0, 0.0] for the "Ramp Up".
    points_ledger = defaultdict(lambda: [0.0, 0.0]) 

    # --- 2. THE TIME LOOP ---
    with engine.begin() as conn:
        race_map = get_race_id_map(conn)
        sorted_keys = sorted(race_map.keys())

        for key in sorted_keys:
            year, round_num = key
            rid = race_map[key]
            
            # Get Dynamic Thresholds
            thresholds = get_price_thresholds(round_num)
            print(f"   📍 Processing {year} Round {round_num} [Thresholds: {thresholds}]...")

            # Fetch Results
            drivers_in_race = conn.execute(text("SELECT driver_id, constructor_id, total_points FROM driver_fantasy_results WHERE race_id = :rid"), {"rid": rid}).fetchall()
            teams_in_race = conn.execute(text("SELECT constructor_id, total_points FROM constructor_fantasy_results WHERE race_id = :rid"), {"rid": rid}).fetchall()

            # ===================================================
            # STEP A: PRE-RACE SYNC (Handle Replacements)
            # ===================================================
            for d in drivers_in_race:
                seat_key = (d.driver_id, d.constructor_id)
                
                # 1. Inherit Price & History if this is a substitute driver
                if seat_key in replacements:
                    parent = replacements[seat_key]
                    
                    # Inherit Price
                    if parent in current_driver_prices:
                        current_driver_prices[d.driver_id] = current_driver_prices[parent]
                        
                    # Inherit History (Critical for accurate pricing)
                    if parent in points_ledger:
                        points_ledger[d.driver_id] = list(points_ledger[parent])

                # Safety check for missing drivers (e.g. new mid-season entries)
                if d.driver_id not in current_driver_prices:
                    current_driver_prices[d.driver_id] = 5.0

            # ===================================================
            # STEP B: SAVE START PRICES (DB Update for Current Race)
            # ===================================================
            # This ensures the DB records what the price WAS at the start of this race
            for d in drivers_in_race:
                conn.execute(text("UPDATE driver_fantasy_results SET start_price = :p WHERE race_id = :rid AND driver_id = :d"), 
                             {"p": round(current_driver_prices[d.driver_id], 1), "rid": rid, "d": d.driver_id})
            
            for t in teams_in_race:
                conn.execute(text("UPDATE constructor_fantasy_results SET start_price = :p WHERE race_id = :rid AND constructor_id = :c"), 
                             {"p": round(current_team_prices.get(t.constructor_id, 10.0), 1), "rid": rid, "c": t.constructor_id})

            # ===================================================
            # STEP C: CALCULATE & APPLY PRICE CHANGE
            # ===================================================
            
            # Temporary dicts to hold prices for the NEXT race
            next_race_driver_prices = {}
            next_race_team_prices = {}

            # --- DRIVERS ---
            for d in drivers_in_race:
                d_code = d.driver_id
                pts = d.total_points or 0
                
                # Identify "Seat Owner" (Price changes stick to the Seat, not the temp driver)
                seat_owner = d_code
                seat_key = (d_code, d.constructor_id)
                if seat_key in replacements:
                    seat_owner = replacements[seat_key]
                
                price = current_driver_prices.get(seat_owner, 5.0)
                
                # 1. Update Ledger with POINTS (not PPM)
                points_ledger[seat_owner].append(float(pts))
                recent_pts = points_ledger[seat_owner][-3:] # Keep last 3
                
                # 2. Calculate Average Points
                avg_pts = sum(recent_pts) / 3.0
                
                # 3. Calculate PPM against CURRENT PRICE
                current_ppm_stat = (avg_pts / price) if price > 0 else 0
                
                # 4. Get Price Delta
                change = calculate_change(current_ppm_stat, price, thresholds)
                
                # 5. Apply Change
                new_price = max(4.5, price + change)
                
                # Update Memory
                current_driver_prices[seat_owner] = new_price
                current_driver_prices[d_code] = new_price # Sync sub driver too
                next_race_driver_prices[d_code] = new_price # Store for DB write

            # --- CONSTRUCTORS ---
            for t in teams_in_race:
                t_id = t.constructor_id
                pts = t.total_points or 0
                price = current_team_prices.get(t_id, 10.0)
                
                # 1. Update Ledger
                points_ledger[t_id].append(float(pts))
                recent_pts = points_ledger[t_id][-3:]
                
                # 2. Avg Points
                avg_pts = sum(recent_pts) / 3.0
                
                # 3. PPM Stat
                current_ppm_stat = (avg_pts / price) if price > 0 else 0
                
                # 4. Change
                change = calculate_change(current_ppm_stat, price, thresholds)
                
                # 5. Apply
                new_price = max(1.0, price + change)
                current_team_prices[t_id] = new_price
                next_race_team_prices[t_id] = new_price

            # ===================================================
            # STEP D: WRITE PRICES TO *NEXT* RACE
            # ===================================================
            
            # Find the ID of the NEXT race to populate its start prices
            next_race_order = round_num + 1
            next_race_query = text("SELECT race_id FROM races WHERE year = :y AND round = :r")
            next_rid_result = conn.execute(next_race_query, {"y": year, "r": next_race_order}).fetchone()
            
            if next_rid_result:
                next_rid = next_rid_result[0]
                # print(f"   💾 Saving Start Prices for Next Race (ID: {next_rid})...")
                
                # Update Drivers
                for d_id, p_val in next_race_driver_prices.items():
                    conn.execute(text("""
                        UPDATE driver_fantasy_results 
                        SET start_price = :p 
                        WHERE race_id = :rid AND driver_id = :d
                    """), {"p": round(p_val, 1), "rid": next_rid, "d": d_id})
                    
                # Update Constructors
                for c_id, p_val in next_race_team_prices.items():
                    conn.execute(text("""
                        UPDATE constructor_fantasy_results 
                        SET start_price = :p 
                        WHERE race_id = :rid AND constructor_id = :c
                    """), {"p": round(p_val, 1), "rid": next_rid, "c": c_id})
            else:
                print("   🏁 End of Season (or no next race found).")

    print("✅ 2025 Price Simulation Complete.")
    
    
def main():
    print("🚀 STARTING PRICE INGESTION...")

    # 2. 2023/2024 Data (Simulation)
    historical_driver_file = '2023-data/startprice_drivers.csv'
    historical_team_file = '2023-data/startprice_constructors.csv'
    simulate_historical_prices(historical_driver_file, historical_team_file)
    
    ingest_2025_prices(
        '2025-data/driver_prices.csv', 
        '2025-data/constructor_prices.csv'
    )

    print("🏁 ALL PRICES INGESTED.")

if __name__ == "__main__":
    main()