import sys
import os
from sqlalchemy import create_engine, text
from db.ingestion.helper.extract_races import extract_races


DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

SPRINT_PT_MAP = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
RACE_PT_MAP   = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
QUALI_PT_MAP  = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}


def parse_lap_time(time_str):
    if not time_str: return None
    try:
        if isinstance(time_str, (int, float)): return float(time_str)
        
        # Clean string
        time_str = str(time_str).strip()
        if time_str == '' or time_str.lower() == 'none': return None
        if ':' in time_str:
            parts = time_str.split(':')
            minutes = float(parts[0])
            seconds = float(parts[1])
            return (minutes * 60) + seconds
        return float(time_str)
    except Exception:
        return None

def calculate_base_results(race_id, race_year, race_event_name):
    print(f"  🏁 Calculating Base Points: {race_year} {race_event_name}...")

    sprint_map = {}
    
    
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # 1. FETCH QUALI DATA
            q_query = text("SELECT driver_id, position, q1_time FROM qualifying WHERE race_id = :rid")
            q_rows = conn.execute(q_query, {"rid": race_id}).fetchall()
            
            quali_map = {}
            valid_q1_times = []
            for row in q_rows:
                t = parse_lap_time(row.q1_time)
                
                if t is not None:
                    valid_q1_times.append(t)
            
                quali_map[row.driver_id] = {
                    'pos': row.position,
                    'q1_time': t
                }
            if valid_q1_times:
                fastest_q1 = min(valid_q1_times)
                threshold_quali = fastest_q1 * 1.05
            
            else:
                threshold_quali = float('inf')

            # 2. FETCH SPRINT DATA
            s_query = text("""
                SELECT driver_id, finish_position, grid_position, status, fastest_lap, laps_completed
                FROM sprint_results 
                WHERE race_id = :rid
            """)
            s_rows = conn.execute(s_query, {"rid": race_id}).fetchall()

            #Map the driver to their stats in the given sprint
            for row in s_rows:
                sprint_map[row.driver_id] = {
                    'finish': row.finish_position,
                    'grid': row.grid_position,
                    'status': row.status,
                    'fastest_lap': row.fastest_lap,
                    'laps_completed': row.laps_completed
                }

            # 3. FETCH RACE DATA
            r_query = text("""
                SELECT driver_id, constructor_id, grid_position, finish_position, fastest_lap, status, laps_completed AS laps
                FROM results 
                WHERE race_id = :rid
            """)
            r_rows = conn.execute(r_query, {"rid": race_id}).fetchall()

            # Calculate Max Laps for 90% rule
            max_laps = 0
            if r_rows:
                max_laps = max((row.laps for row in r_rows if row.laps is not None), default=0)
            threshold_laps = max_laps * 0.84
            
            max_sprint_laps = 0
            if r_rows:
                # Safely check for sprint_laps (defaults to 0 if column is missing/None)
                max_sprint_laps = max((getattr(row, 'sprint_laps', 0) for row in r_rows if getattr(row, 'sprint_laps', None) is not None), default=0)
            
            
            sprint_threshold_laps = max_sprint_laps * 0.80
            
            driver_inserts = []

            for row in r_rows:
                curr_driver = row.driver_id
                curr_constructor = row.constructor_id
                status = row.status
                
                q_data = quali_map.get(curr_driver, {})
                
                q_pos = q_data.get('pos')
                q_time = q_data.get('q1_time')

                # --- QUALI CALC ---
                if q_time is None or q_time > threshold_quali:
                    quali_pts = -5
                elif q_pos is not None:
                     quali_pts = QUALI_PT_MAP.get(q_pos, 0)

                # --- SPRINT CALC ---
                sprint_pts = 0
                if len(sprint_map) > 0 and curr_driver in sprint_map:
                    s_data = sprint_map[curr_driver]
                    s_fin = s_data['finish']
                    s_grid = s_data['grid'] if s_data['grid'] else 20
                    s_stat = str(s_data['status'])
                    s_fl = s_data['fastest_lap']
                    s_lc = s_data['laps_completed']

                    # Finish Points
                    if s_fin in SPRINT_PT_MAP:
                        sprint_pts += SPRINT_PT_MAP[s_fin]
                    
                    # Position Gain (Grid - Finish)
                    if s_fin:
                        sprint_pts += (s_grid - s_fin)

                    # Sprint Fastest Lap (+5 Bonus)
                    if s_fl:
                        sprint_pts += 5

                    
                    if 'Disqualified' in s_stat:
                        sprint_pts = -20
                    elif s_lc < sprint_threshold_laps:
                        sprint_pts = -20 
                    
                # --- RACE CALC ---
                race_pts = 0
                pos_gain_pts = 0
                fl_pts = 0

                fin_pos = row.finish_position
                r_grid = row.grid_position if row.grid_position else 20
                safe_fin_pos = fin_pos if fin_pos else 20
                
                # Finish Points
                race_pts += RACE_PT_MAP.get(safe_fin_pos, 0)

                # Position Gain
                if fin_pos and fin_pos > 0:
                    pos_gain_pts = r_grid - safe_fin_pos

                # Fastest Lap Bonus (+10) 
                if row.fastest_lap:
                    fl_pts = 10
                    race_pts += 10
                
                # 90% Distance Rule & DNF
                driver_laps = row.laps if row.laps else 0
                is_valid_finish = driver_laps >= threshold_laps
                
                if 'Disqualified' in str(status):
                    race_pts = -20
                    pos_gain_pts = 0
                    if row.fastest_lap:
                        fl_pts = 0
                elif not is_valid_finish:
                    race_pts = -20
                    pos_gain_pts = 0

                # --- TOTAL CALCULATION ---
                total_pts = quali_pts + sprint_pts + race_pts + pos_gain_pts

                driver_inserts.append({
                    "race_id": race_id,
                    "driver_id": curr_driver,
                    "constructor_id": curr_constructor,
                    "quali_points": quali_pts,
                    "sprint_points": sprint_pts,
                    "race_points": race_pts + pos_gain_pts,
                    "fl_points": fl_pts,
                    "total_points": total_pts
                })

            # --- INSERT INTO DB ---
            conn.execute(text("DELETE FROM driver_fantasy_results WHERE race_id = :rid"), {"rid": race_id})

            if driver_inserts:
                conn.execute(text("""
                    INSERT INTO driver_fantasy_results 
                    (race_id, driver_id, constructor_id, quali_points, sprint_points, race_points, fl_points, total_points)
                    VALUES (:race_id, :driver_id, :constructor_id, :quali_points, :sprint_points, :race_points, :fl_points, :total_points)
                """), driver_inserts)

            trans.commit()
            print(f"   ✅ Inserted {len(driver_inserts)} driver results.")

        except Exception as e:
            trans.rollback()
            print(f"   ❌ Error calculating {race_event_name}: {e}")

def main():
    print("🚀 Starting Base Result Ingestion...")
    races = extract_races()
    print(f"📊 Found {len(races)} races to process.")
    
    #for every race that occured calculate the results given the race_id, race_year, and race_name
    for row in races:
        calculate_base_results(row["id"], row["year"], row["event_name"])
    
    print("🏁 Base Ingestion Complete.")

if __name__ == "__main__":
    main()