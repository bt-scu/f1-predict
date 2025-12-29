import os
import pandas as pd
import json
from sqlalchemy import create_engine, text
from db.ingestion.helper.get_race_id_map import get_race_id_map

DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

def ingest_dotd_2023_2024(season_directory):
    csv_files = [f for f in os.listdir(season_directory) if f.endswith('.csv')]
    if not csv_files:
        print(f"⚠️ No CSV files found in {season_directory}")
        return
    
    with engine.connect() as conn:
        race_map = get_race_id_map(conn)
    
    print(f"📂 Processing {len(csv_files)} CSV files for DotD...")

    for filename in csv_files:
        try:
            with engine.begin() as conn:
                file_path = os.path.join(season_directory, filename)
                df = pd.read_csv(file_path)
                
                for index, row in df.iterrows():
                    driver_code = row.get('driver_code')
                    driver_code = str(driver_code).strip().upper()
                    is_dotd = bool(row['dotd'])

                    season = int(row['season'])
                    round_num = int(row['round'])
                    race_id = race_map.get((season, round_num))
                    
                    if not race_id: continue
                    query = text("""
                        UPDATE driver_fantasy_results
                        SET 
                            total_points = COALESCE(total_points, 0)
                                         - (CASE WHEN COALESCE(dotd, false) THEN 10 ELSE 0 END) 
                                         + (CASE WHEN :new_dotd THEN 10 ELSE 0 END),            
                            dotd = :new_dotd
                        WHERE race_id = :race_id AND driver_id = :driver_code
                    """)

                    conn.execute(query, {
                        "new_dotd": is_dotd,
                        "race_id": race_id,
                        "driver_code": driver_code
                    })

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")


def ingest_dotd_2025(season_directory):
    json_files = [f for f in os.listdir(season_directory) if f.endswith('.json')]
    if not json_files: return

    print(f"🚀 Processing {len(json_files)} JSON files for DotD...")

    with engine.connect() as conn:
        race_map = get_race_id_map(conn)
    
    for filename in json_files:
        try:
            with engine.begin() as conn:
                file_path = os.path.join(season_directory, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # ID Lookup
                driver_abbr = data.get('abbreviation', '').strip().upper()
                for race in data.get('races', []):
                    try: round_num = int(race.get('round', 0))
                    except: continue 

                    race_id = race_map.get((2025, round_num))

                    race_data = race.get('race')
                    raw_dotd = race_data.get('dotd')
                    is_dotd = False
                      
                    if (float(raw_dotd) == 10):
                      is_dotd=True

                    query = text("""
                        UPDATE driver_fantasy_results
                        SET 
                            total_points = COALESCE(total_points, 0)
                                         - (CASE WHEN COALESCE(dotd, false) THEN 10 ELSE 0 END)
                                         + (CASE WHEN :new_dotd THEN 10 ELSE 0 END),
                            dotd = :new_dotd
                        WHERE race_id = :race_id AND driver_id = :driver_code
                    """)
                    params = {
                        "new_dotd": is_dotd,
                        "race_id": race_id, 
                        "driver_code": driver_abbr
                    }
                    result = conn.execute(query, params)
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

def main():
    print("🌟 STARTING DRIVER OF THE DAY INGESTION...")
    ingest_dotd_2023_2024("2023-data/drivers")
    ingest_dotd_2023_2024("2024-data/drivers")
    ingest_dotd_2025("2025-data/driver_data")
    print("🏁 DOTD INGESTION COMPLETE.")

if __name__ == "__main__":
    main()