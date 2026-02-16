import pandas as pd
from sqlalchemy import text
from db.config import engine
from db.ingestion.helper.extract_races import extract_races

# Point Maps
SPRINT_PT_MAP = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
RACE_PT_MAP   = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
QUALI_PT_MAP  = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}

def calculate_constructor_scores(race_id, race_year, race_event_name):
    print(f"🏎️  Processing Constructors for: {race_event_name} ({race_year})")
    
    with engine.begin() as conn:
        # 0. CLEANUP: Delete existing scores for this race to allow re-runs
        conn.execute(text("DELETE FROM constructor_fantasy_results WHERE race_id = :rid"), {"rid": race_id})

        # 1. Map Constructors to Drivers
        query = text("""
            SELECT constructor_id, driver_id 
            FROM results 
            WHERE race_id = :race_id AND constructor_id IS NOT NULL
        """)
        df = pd.read_sql(query, conn, params={"race_id": race_id})
        
        if df.empty:
            print(f"   ⚠️ No driver results found for Race {race_id}. Skipping.")
            return

        driver_map = df.groupby('constructor_id')['driver_id'].apply(list).to_dict()
        
        # 2. Check if Sprint Data exists
        s_check_query = text("SELECT COUNT(*) FROM sprint_results WHERE race_id = :rid")
        is_sprint_weekend = conn.execute(s_check_query, {"rid": race_id}).scalar() > 0

        results_to_insert = []

        # 3. Calculate Scores per Constructor
        for constructor_id, driver_ids in driver_map.items():
            if not constructor_id: continue # Skip None/Null constructors

            # A. Qualifying Score (Includes "Both Cars in Q3" bonus)
            q_score = calculate_qualifying_points(conn, driver_ids, race_id)
            
            # B. Race Score (Includes DNF penalties)
            r_score = calculate_race_points(conn, driver_ids, race_id)
            
            # C. Sprint Score
            s_score = 0
            if is_sprint_weekend:
                s_score = calculate_sprint_points(conn, driver_ids, race_id)
            
            # D. Overtakes (Fetch from Driver Fantasy table as these are hard to calc raw)
            overtakes = ingest_constructor_overtakes(conn, driver_ids, race_id)
            
            total_points = q_score + r_score + s_score + overtakes["race_overtakes"] + overtakes["sprint_overtakes"]
            
            results_to_insert.append({
                "race_id": race_id,
                "constructor_id": constructor_id,
                "quali_score": q_score,
                "race_score": r_score,
                "sprint_score": s_score,
                "total_points": total_points
            })

        # 4. Batch Insert
        if results_to_insert:
            insert_query = text("""
                INSERT INTO constructor_fantasy_results (
                    race_id, constructor_id, 
                    quali_score, race_score, sprint_score, total_points,
                    start_price
                ) VALUES (
                    :race_id, :constructor_id, 
                    :quali_score, :race_score, :sprint_score, :total_points,
                    0
                )
            """)
            conn.execute(insert_query, results_to_insert)
            print(f"   ✅ Saved scores for {len(results_to_insert)} constructors.")

def calculate_qualifying_points(conn, driver_ids, race_id):
    # Safety: Ensure list isn't empty
    if not driver_ids: return 0
    
    query = text("""
        SELECT driver_id, position, status
        FROM qualifying 
        WHERE race_id = :rid AND driver_id IN :dids
    """)
    rows = conn.execute(query, {"rid": race_id, "dids": tuple(driver_ids)}).fetchall()
    driver_data = {row.driver_id: row for row in rows}
        
    base_points = 0
    q2_reached = 0
    q3_reached = 0
    
    for did in driver_ids:
        # CRITICAL FIX: Handle if driver missing from Qualifying data (e.g. pit lane start)
        row = driver_data.get(did)
        if not row:
            # If they didn't participate in Quali, usually -5 (NC) or 0 depending on your rules.
            # We'll treat as NC (-5) to be safe.
            base_points += -5
            continue
            
        pos = row.position
        status = str(row.status).lower() if row.status else ""
        
        if 'disqualified' in status:
            base_points += -15
            continue

        if pos is None or 'nc' in status:
            base_points += -5
            continue

        base_points += QUALI_PT_MAP.get(pos, 0)
        
        if pos <= 15: q2_reached += 1
        if pos <= 10: q3_reached += 1

    # Constructor Bonuses
    bonus = 0
    if q3_reached == 2: bonus = 10
    elif q3_reached == 1: bonus = 5
    elif q2_reached == 2: bonus = 3
    elif q2_reached == 1: bonus = 1
    else: bonus = -1 # Assuming -1 if NO cars reach Q2? Or maybe 0? Leaving as your logic.
    
    return base_points + bonus

def calculate_race_points(conn, driver_ids, race_id):
    if not driver_ids: return 0

    query = text("""
        SELECT driver_id, grid_position, finish_position, laps_completed, status, fastest_lap
        FROM results 
        WHERE race_id = :rid AND driver_id IN :dids
    """)
    rows = conn.execute(query, {"rid": race_id, "dids": tuple(driver_ids)}).fetchall()
    driver_data = {row.driver_id: row for row in rows}
    
    max_laps_query = text("SELECT MAX(laps_completed) FROM results WHERE race_id = :rid")
    max_laps = conn.execute(max_laps_query, {"rid": race_id}).scalar() or 0
    laps_threshold = max_laps * 0.9
    
    total_points = 0
    
    for did in driver_ids:
        row = driver_data.get(did)
        if not row: continue # Should not happen if Step 1 works, but safe to keep

        g_pos = row.grid_position if row.grid_position is not None else 0
        f_pos = row.finish_position
        laps = row.laps_completed if row.laps_completed is not None else 0
        status = str(row.status).lower()
        
        # Robust Fastest Lap Check
        # Accepts: 1, '1', True, 'True'
        raw_fl = str(row.fastest_lap).lower()
        is_fastest = raw_fl in ['1', 'true', 't', '1.0']
        
        if 'disqualified' in status:
            total_points += -30 
            continue 
            
        # DNF / 90% Rule
        if (laps < laps_threshold):
            total_points += -20
            if is_fastest: total_points += 10 # Retain FL point if they set it before DNF? (Rare but possible)
            continue
            
        # Classified Scoring
        points_pos = RACE_PT_MAP.get(f_pos, 0)
        pos_gained = (g_pos - f_pos)
        
        # Cap positions gained/lost if needed (e.g. max +10 or -10), or keep raw
        total_points += points_pos + pos_gained
        
        if is_fastest: total_points += 10
            
    return total_points

def calculate_sprint_points(conn, driver_ids, race_id):
    if not driver_ids: return 0

    query = text("""
        SELECT driver_id, grid_position, finish_position, laps_completed, status, fastest_lap
        FROM sprint_results 
        WHERE race_id = :rid AND driver_id IN :dids
    """)
    rows = conn.execute(query, {"rid": race_id, "dids": tuple(driver_ids)}).fetchall()
    driver_data = {row.driver_id: row for row in rows}
    
    max_laps_query = text("SELECT MAX(laps_completed) FROM sprint_results WHERE race_id = :rid")
    max_laps = conn.execute(max_laps_query, {"rid": race_id}).scalar() or 0
    laps_threshold = max_laps * 0.9
    
    total_points = 0
    
    for did in driver_ids:
        row = driver_data.get(did)
        if not row: continue
            
        g_pos = row.grid_position if row.grid_position is not None else 0
        f_pos = row.finish_position
        laps = row.laps_completed if row.laps_completed is not None else 0
        status = str(row.status).lower()

        raw_fl = str(row.fastest_lap).lower()
        is_fastest = raw_fl in ['1', 'true', 't', '1.0']
        
        if 'disqualified' in status:
            total_points += -30 
            continue 
            
        if (laps < laps_threshold):
            total_points += -20
            continue
            
        total_points += SPRINT_PT_MAP.get(f_pos, 0)
        total_points += (g_pos - f_pos)
        
        if is_fastest: total_points += 5
            
    return total_points

def ingest_constructor_overtakes(conn, driver_ids, race_id):
    if not driver_ids: return {"race_overtakes": 0, "sprint_overtakes": 0}

    # Fetch pre-calculated overtakes from the driver_fantasy table
    # This assumes driver_fantasy_results is ALREADY POPULATED for this race
    race_query = text("""
        SELECT COALESCE(SUM(overtakes), 0)
        FROM driver_fantasy_results
        WHERE race_id = :rid AND driver_id IN :dids
    """)
    
    sprint_query = text("""
        SELECT COALESCE(SUM(sprint_overtakes), 0)
        FROM driver_fantasy_results
        WHERE race_id = :rid AND driver_id IN :dids
    """)
    
    race_overtakes_count = conn.execute(race_query, {"rid": race_id, "dids": tuple(driver_ids)}).scalar()
    sprint_overtakes_count = conn.execute(sprint_query, {"rid": race_id, "dids": tuple(driver_ids)}).scalar()
    
    return {
        "race_overtakes": race_overtakes_count or 0, # Handle None
        "sprint_overtakes": sprint_overtakes_count or 0
    }
    
def main():
    races = extract_races()
    print(f"📊 Found {len(races)} races to process.")
    for r in races:
        calculate_constructor_scores(r["id"], r["year"], r["event_name"])

if __name__ == "__main__":
    main()