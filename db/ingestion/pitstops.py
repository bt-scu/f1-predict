import os
import sys
import pandas as pd
import json
from sqlalchemy import create_engine, text

from db.ingestion.helper.get_race_id_map import get_race_id_map

DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

# Ensure this map matches your DB 'constructor_id' exactly
TEAM_MAP = {
    'RED': 'Red Bull Racing', 'RBR': 'Red Bull Racing', 'Red Bull': 'Red Bull Racing',
    'MER': 'Mercedes', 'Mercedes': 'Mercedes',
    'FER': 'Ferrari', 'Ferrari': 'Ferrari',
    'MCL': 'McLaren', 'McLaren': 'McLaren',
    'AST': 'Aston Martin', 'AMR': 'Aston Martin', 'Aston Martin': 'Aston Martin',
    'ALP': 'Alpine', 'Alpine': 'Alpine',
    'WIL': 'Williams', 'Williams': 'Williams',
    'HAA': 'Haas F1 Team', 'HAS': 'Haas F1 Team', 'Haas': 'Haas F1 Team',
    'KCK': 'Kick Sauber', 'SAU': 'Kick Sauber', 'ALF': 'Kick Sauber', 'Kick Sauber': 'Kick Sauber',
    'VRB': 'RB', 'RB': 'RB', 'Racing Bulls': 'RB',
    'AT': 'AlphaTauri', 'AlphaTauri': 'AlphaTauri',
    'ALT': 'AlphaTauri'
}

def calculate_pit_points(pit_time, is_fastest, is_wr):
    points = 0
    
    # --- BASE TIER POINTS ---
    if pit_time < 2.00:
        points = 20
    elif pit_time <= 2.19: 
        points = 10
    elif pit_time <= 2.49:
        points = 5
    elif pit_time <= 2.99:
        points = 2
    else:
        points = 0

    # --- BONUSES ---
    if is_fastest:
        points += 5  
    if is_wr:
        points += 15 
        
    return points

def ingest_pitstops_2023_2024(season_directory):
    csv_files = [f for f in os.listdir(season_directory) if f.endswith('.csv')]
    if not csv_files: return

    print(f"🔧 Processing {len(csv_files)} CSV files for Pit Stops (2023-24)...")

    with engine.connect() as conn:
        race_map = get_race_id_map(conn)

    for filename in csv_files:
        try:
            file_path = os.path.join(season_directory, filename)
            df = pd.read_csv(file_path)
            
            # Extract Season/Round from the first row
            first_row = df.iloc[0]
            season = int(first_row.get('season', 0))
            round_num = int(first_row.get('round', 0))

            race_id = race_map.get((season, round_num))

            # --- PROCESS BEST STOPS ---
            with engine.begin() as conn:
                for code, row in df.iterrows():
                    db_constructor_id = TEAM_MAP[row['constructor_code']]

                    pit_time = float(row['pit_time'])
                    
                    # Handle boolean conversion safely
                    fastest_pit = row['fastest_pit']
                    world_record = row['world_record']

                    new_points = calculate_pit_points(pit_time, fastest_pit, world_record)
                    
                    # Update DB
                    conn.execute(text("""
                        UPDATE constructor_fantasy_results
                        SET 
                            pitstop_points = :pts,
                            total_points = (total_points - COALESCE(pitstop_points, 0) + :pts)
                        WHERE race_id = :rid AND constructor_id = :cid
                    """), {
                        "pts": new_points, 
                        "rid": race_id, 
                        "cid": db_constructor_id 
                    })
                    
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

def ingest_pitstops_2025(season_directory):
    json_files = [f for f in os.listdir(season_directory) if f.endswith('.json')]
    if not json_files: return

    print(f"🚀 Processing {len(json_files)} JSON files for Pit Stops (2025)...")

    with engine.connect() as conn:
        race_map = get_race_id_map(conn)

    for filename in json_files:
        try:
            with open(os.path.join(season_directory, filename), 'r') as f:
                data = json.load(f)
            
            abbrev = data.get('abbreviation', '').strip().upper()
            db_constructor_id = TEAM_MAP.get(abbrev)
        
            for race in data.get('races'):
                round_num = int(race['round'])
                race_id = race_map.get((2025, round_num))

                r_data = race.get('race')
                
                total_pts = r_data.get('fastestPitStop', 0) + r_data.get('pitStopBonus', 0 ) + r_data.get('worldRecordBonus', 0)
                
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE constructor_fantasy_results
                        SET 
                            pitstop_points = :pts,
                            total_points = (total_points - COALESCE(pitstop_points, 0) + :pts)
                        WHERE race_id = :rid AND constructor_id = :cid
                    """), {
                        "pts": total_pts, 
                        "rid": race_id, 
                        "cid": db_constructor_id
                    })

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

def main():
    print("🌟 STARTING PITSTOP INGESTION...")
    ingest_pitstops_2023_2024("2023-data/constructors")
    ingest_pitstops_2023_2024("2024-data/constructors")
    ingest_pitstops_2025("2025-data/constructor_data")
    print("🏁 PITSTOP INGESTION COMPLETE.")

if __name__ == "__main__":
    main()