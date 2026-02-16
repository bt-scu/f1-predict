import os
import sys
import pandas as pd
import json
from sqlalchemy import text
from db.config import engine
from db.ingestion.helper.get_race_id_map import get_race_id_map

def check_dsq(conn, driver_id, race_id, session='race'):
    """
    Returns True if the driver was disqualified in the specified session.
    session: 'race' (default) or 'sprint'
    """
    # Select the correct table based on the session
    table = "sprint_results" if session == 'sprint' else "results"
    
    query = text(f"SELECT status FROM {table} WHERE race_id = :rid AND driver_id = :did")
    status = conn.execute(query, {"rid": race_id, "did": driver_id}).scalar()
    
    # Return True only if status is found and contains "Disqualified"
    return status is not None and "Disqualified" in str(status)

def ingest_overtakes_2023_2024(season_directory):
    csv_files = [f for f in os.listdir(season_directory) if f.endswith('.csv')]
    if not csv_files: return
    
    print(f"🏎️  Processing {len(csv_files)} CSV files for Overtakes (2023-24)...")
    
    with engine.connect() as conn:
        race_map = get_race_id_map(conn)
    
    for filename in csv_files:
        try:
            with engine.begin() as conn:
                file_path = os.path.join(season_directory, filename)
                df = pd.read_csv(file_path)
                
                df.columns = df.columns.str.lower().str.strip()
                for index, row in df.iterrows():
                    driver_code = row.get('driver_code')
                    if pd.isna(driver_code): continue
                    driver_code = str(driver_code).strip().upper()

                    ot = row['overtakes']
                    season = int(row['season'])
                    round_num = int(row['round'])
                    race_id = race_map.get((season, round_num))
                    
                    is_dsq = check_dsq(conn, driver_code, race_id, session='race')
                    if is_dsq:
                            ot = 0
                    if not race_id: continue

                    # Update Main Race Overtakes
                    # Logic: Remove OLD points, add NEW points (1 point per overtake)
                    query = text("""
                        UPDATE driver_fantasy_results
                        SET 
                            total_points = COALESCE(total_points, 0)
                                         - COALESCE(overtakes, 0)
                                         + :new_overtakes,
                            overtakes = :new_overtakes
                        WHERE race_id = :race_id AND driver_id = :driver_code
                    """)

                    conn.execute(query, {
                        "new_overtakes": ot,
                        "race_id": race_id, 
                        "driver_code": driver_code
                    })

                    # --- SPRINT OVERTAKES ---
                    if int(row.get('sprint', 0)) == 1:
                        s_ot = row['sprint_overtakes']
                        is_dsq = check_dsq(conn, driver_code, race_id, session='sprint')
                        if is_dsq:
                            s_ot = 0

                        sprint_query = text("""
                            UPDATE driver_fantasy_results
                            SET 
                                total_points = COALESCE(total_points, 0)
                                             - COALESCE(sprint_overtakes, 0)
                                             + :new_s_overtakes,
                                sprint_overtakes = :new_s_overtakes
                            WHERE race_id = :race_id AND driver_id = :driver_code
                        """)
                        
                        conn.execute(sprint_query, {
                            "new_s_overtakes": s_ot,
                            "race_id": race_id, 
                            "driver_code": driver_code
                        })

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

def ingest_overtakes_2025(season_directory):
    json_files = [f for f in os.listdir(season_directory) if f.endswith('.json')]
    if not json_files: return

    print(f"🚀 Processing {len(json_files)} JSON files for Overtakes (2025)...")

    with engine.connect() as conn:
        race_map = get_race_id_map(conn)
    
    for filename in json_files:
        try:
            with engine.begin() as conn:
                file_path = os.path.join(season_directory, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                driver_abbr = data.get('abbreviation', '').strip().upper()

                for race in data.get('races', []):
                    round_num = int(race.get('round'))

                    race_id = race_map.get((2025, round_num))

                    # --- MAIN RACE OVERTAKES ---
                    race_data = race.get('race')
                    
                    overtakes = race_data.get('overtakeBonus')
                    
                    is_dsq = check_dsq(conn, driver_abbr, race_id, session='race')
                    
                    if is_dsq:
                        overtakes = 0
                         
                    
                    query = text("""
                        UPDATE driver_fantasy_results
                        SET 
                            total_points = COALESCE(total_points, 0)
                                         - COALESCE(overtakes, 0)
                                         + :new_overtakes,
                            overtakes = :new_overtakes
                        WHERE race_id = :race_id AND driver_id = :driver_code
                    """)
                    
                    # Try with Abbreviation
                    result = conn.execute(query, {
                        "new_overtakes": overtakes,
                        "race_id": race_id, 
                        "driver_code": driver_abbr
                    })
                    
                    # --- SPRINT OVERTAKES ---
                    sprint_data = race.get('sprint')
                    if sprint_data:
                        s_overtakes = sprint_data.get('overtakeBonus', 0)
                        is_dsq = check_dsq(conn, driver_abbr, race_id, session='sprint')
                        if is_dsq:
                            s_overtakes = 0
                        s_query = text("""
                            UPDATE driver_fantasy_results
                            SET 
                                total_points = COALESCE(total_points, 0)
                                             - COALESCE(sprint_overtakes, 0)
                                             + :new_s_overtakes,
                                sprint_overtakes = :new_s_overtakes
                            WHERE race_id = :race_id AND driver_id = :driver_code
                        """)
                        # Try with Abbr
                        conn.execute(s_query, {
                            "new_s_overtakes": s_overtakes,
                            "race_id": race_id, 
                            "driver_code": driver_abbr
                        })

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

def main():
    print("🌟 STARTING OVERTAKE INGESTION...")
    ingest_overtakes_2023_2024("2023-data/drivers")
    ingest_overtakes_2023_2024("2024-data/drivers")
    ingest_overtakes_2025("2025-data/driver_data")
    print("🏁 OVERTAKE INGESTION COMPLETE.")
 
if __name__ == "__main__":
    main()