from sqlalchemy import text
from db.config import engine
import fastf1
import pandas as pd
import time
import math
fastf1.Cache.set_disabled()

#function to return race id or create entry in DB and return race id
def get_or_create_race(year, round_num, session):
    with engine.connect() as conn:
        trans = conn.begin()
    
        try:
            #check to see if the race_id already exists given race year and round number
            existing_id = conn.execute(
                text("SELECT race_id FROM races WHERE year = :y AND round = :r"),
                {"y": year, "r": round_num}
            ).scalar()
            
            if existing_id:
                return existing_id
            
            print(f"     + New Race Detected: {session.event.EventName}")
            
            circuit = session.event['Location'] 
            name = session.event['EventName']
            date = session.event['EventDate']
            
            #insert some race data, having it serve as the foreign key
            new_id = conn.execute(
                text(
                """
                    INSERT INTO races (year, round, circuit_name, event_name, date)
                    VALUES (:y, :r, :c, :n, :d)
                    RETURNING race_id
                """
                ),
                {"y": year, "r": round_num, "c": circuit, "n": name, "d": date}
            ).scalar()
            
            trans.commit()
            return new_id
        
        except Exception as e:
            trans.rollback()
            raise e

#helper function to check if data exists in database
def check_data_exists(race_id, table_name, session_type=None):
    with engine.connect() as conn:
        try:
            #given the table name check if a single instance given the race _id exists
            if table_name in ['laps', 'results', 'qualifying', 'sprint_laps', 'sprint_results']:
                query = text(f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE race_id = :rid)")
                return conn.execute(query, {"rid": race_id}).scalar()
            
            # given table name check if race id exists specified by the session type
            elif table_name in ['practice_laps', 'weather']:
                query = text(f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE race_id = :rid AND session_type = :st)")
                return conn.execute(query, {"rid": race_id, "st": session_type}).scalar()
                
        except Exception as e:
            print(f"      ?? Error checking DB: {e}")
            return False
    return False

#function to handle construct a drivers foreign key table
def process_drivers(session):
    with engine.connect() as conn:
        trans = conn.begin()
        #extract current drivers from DB to memory
        query_drivers = conn.execute(text("SELECT driver_id, team_name FROM drivers"))
        driver_map = {row[0]: row[1] for row in query_drivers}
        
        #for all drivers in the current session
        for driver_id in session.drivers:
            d_info = session.get_driver(driver_id)
            
            if not d_info['Abbreviation']:
                continue
            
            d_id = d_info['Abbreviation']
            d_team = d_info['TeamName']
            d_num = d_info['DriverNumber']
            d_name = d_info['FullName']
            
            #if driver is not in memory, they are not in table so add them
            if d_id not in driver_map:
                conn.execute(
                    text("""
                    INSERT INTO drivers (driver_id, driver_number, full_name, team_name)
                    VALUES (:id, :num, :name, :team)
                    """), {"id": d_id, "num": d_num, "name": d_name, "team": d_team}
                )
                driver_map[d_id] = d_team
            #extra check to see if drivers changed team
            elif driver_map[d_id] != d_team:
                conn.execute(
                    text("UPDATE drivers SET team_name = :t WHERE driver_id = :id"), {"t": d_team, "id": d_id}
                )
                driver_map[d_id] = d_team
        trans.commit()

#helper function to ensure that the value read to integer values can be stored into DB
def safe_int(val):
    try:
        if val is None:
            return None
        # Check for NaN or Infinity
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        
        val = int(val)
        # Postgres INTEGER limit check (-2.1B to +2.1B)
        if abs(val) > 2147483647: 
            return None
            
        return val
    except (ValueError, TypeError):
        return None
    
def reassign_fastest_lap(engine, race_id, session_type='race'):
    """
    Opens its own connection to reassign the fastest lap.
    Call this from main() after ingestion is fully committed.
    """
    
    # Select dynamic table names
    if session_type == 'sprint':
        results_table = "sprint_results"
        laps_table = "sprint_laps" 
    else:
        results_table = "results"
        laps_table = "laps"

    print(f"🔄 Reassigning Fastest Lap for {session_type} (ID: {race_id})...")

    with engine.begin() as conn:
        # 1. Find the true fastest driver (ignoring DSQ)
        query_inplay_driver = text(f"""
            SELECT l.driver_id
            FROM {laps_table} l
            JOIN {results_table} r ON l.race_id = r.race_id AND l.driver_id = r.driver_id
            WHERE l.race_id = :rid
              AND r.status NOT LIKE '%Disqualified%' 
              AND l.lap_time IS NOT NULL 
            ORDER BY l.lap_time ASC
            LIMIT 1
        """)
        
        inplay_fastest_driver = conn.execute(query_inplay_driver, {"rid": race_id}).scalar()
        
        if not inplay_fastest_driver:
            print(f"   ⚠️ No valid fastest lap found for {session_type}.")
            return

        print(f"   ✨ Valid Fastest Driver identified: {inplay_fastest_driver}")

        # 2. Reset everyone
        reset_query = text(f"UPDATE {results_table} SET fastest_lap = false WHERE race_id = :rid")
        conn.execute(reset_query, {"rid": race_id})
        
        # 3. Award the point
        update_query = text(f"""
            UPDATE {results_table} 
            SET fastest_lap = true 
            WHERE race_id = :rid AND driver_id = :did
        """)
        conn.execute(update_query, {"rid": race_id, "did": inplay_fastest_driver})

def process_race_data(race_id, session):
    
    #construct drivers table based on the current drivers in a race
    process_drivers(session)
    
    #construct weather data table based on race conditions
    process_weather_data(race_id, session, 'Race')
    
    #check if we have the race data for a given race
    if check_data_exists(race_id, 'results'):
        print(f"      ⏭️  Skipping Results (Already in DB)")
        return
    
    results = session.results
    data_to_insert = []
    
    #for every driver add their race performance to the db
    for driver_label, row in results.iterrows():
        data_to_insert.append({
            "rid": race_id,
            "did": row['Abbreviation'],  
            "cid": row['TeamName'],        
            "grid": safe_int(row['GridPosition']),
            "fin": safe_int(row['Position']),
            "pts": float(row['Points']),
            "laps": safe_int(row['Laps']),
            "stat": str(row['Status'])  
        })
    #append driver results to the results table
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            if data_to_insert:
                conn.execute(text(
                """
                    INSERT INTO results (
                        race_id, driver_id, constructor_id, grid_position, 
                        finish_position, points, laps_completed, status
                    ) VALUES (
                        :rid, :did, :cid, :grid, 
                        :fin, :pts, :laps, :stat
                    )
                """
                ), data_to_insert)
                
            trans.commit()
            print(f"      > ✅ Saved {len(data_to_insert)} results")
        except Exception as e:
            trans.rollback()
            print(f"      !! Error saving results: {e}")
    #code section to store lap data to memory
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            laps = session.laps.pick_quicklaps()
            laps_to_insert = []
            for index, lap in laps.iterrows():
                # Time conversion
                l_time = lap['LapTime'].total_seconds() if pd.notnull(lap['LapTime']) else None
                
                laps_to_insert.append({
                    "rid": race_id,
                    "did": lap['Driver'],
                    "lnum": safe_int(lap['LapNumber']),
                    "ltime": l_time,
                    "pos": safe_int(lap['Position']),
                    "comp": lap['Compound'],
                    "tlife": safe_int(lap['TyreLife'])
                })
            # insert lap data to db
            if laps_to_insert:
                conn.execute(text("""
                    INSERT INTO laps (race_id, driver_id, lap_number, lap_time, position, compound, tyre_life)
                    VALUES (:rid, :did, :lnum, :ltime, :pos, :comp, :tlife)
                """), laps_to_insert)
            
            try:
                fastest = session.laps.pick_fastest() 
                f_driver = fastest['Driver']

                # Update the 'results' table to mark this driver
                conn.execute(text("""
                    UPDATE results 
                    SET fastest_lap = TRUE 
                    WHERE race_id = :rid AND driver_id = :did
                """), {"rid": race_id, "did": f_driver})
                
                print(f"      > 🏎️  Fastest Lap awarded to {f_driver}") 
            except Exception as fl_error:
                print(f"      ~ Warning: Could not set fastest lap: {fl_error}")
            
            trans.commit()
            print(f"      > ✅ Saved {len(laps_to_insert)} Race Laps")
            
        except Exception as e:
            trans.rollback()
            print(f"      !! Error saving results: {e}")
    return
        
           
def process_practice(year, round_num, race_id, practice_sessions):
    # iterate through each practice session
    for s_name in practice_sessions:
        print(f"      ⚙️ Processing {s_name}...")
        # check if data exists for the given session
        if check_data_exists(race_id, 'practice_laps', s_name):
            print(f"      ⏭️  Skipping {s_name} (Already in DB)")
            continue
        try:
            # extract session data
            session = fastf1.get_session(year, round_num, s_name)
            session.load(telemetry=False, weather=True, messages=False)
            
            #process race data for the given session
            process_weather_data(race_id, session, s_name)
        except Exception as e:
            print(f"      !! API Error loading {s_name}: {e}")
            continue
        
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                #extract lap data and store API session results to memory
                laps = session.laps.pick_quicklaps()
                laps_to_insert = []
                for index, lap in laps.iterrows():
                # Safely convert times to floats (handle NaNs/None)
                    l_time = lap['LapTime'].total_seconds() if pd.notnull(lap['LapTime']) else None
                    s1 = lap['Sector1Time'].total_seconds() if pd.notnull(lap['Sector1Time']) else None
                    s2 = lap['Sector2Time'].total_seconds() if pd.notnull(lap['Sector2Time']) else None
                    s3 = lap['Sector3Time'].total_seconds() if pd.notnull(lap['Sector3Time']) else None
                                
                    laps_to_insert.append({
                        "rid": race_id,
                        "did": lap['Driver'],
                        "stype": s_name,
                        "lnum": safe_int(lap['LapNumber']),
                        "ltime": l_time,
                        "s1": s1, "s2": s2, "s3": s3,
                        "comp": lap['Compound'],
                        "tlife": safe_int(lap['TyreLife']), 
                        "stint": safe_int(lap['Stint'])
                    })
                            
                # Bulk Insert to DB
                if laps_to_insert:
                    conn.execute(
                    text(
                        """
                        INSERT INTO practice_laps (
                            race_id, driver_id, session_type, lap_number, 
                            lap_time, sector_1, sector_2, sector_3, 
                            compound, tyre_life, stint
                        ) VALUES (
                            :rid, :did, :stype, :lnum, 
                            :ltime, :s1, :s2, :s3, 
                            :comp, :tlife, :stint
                        )
                        """
                    ),
                    laps_to_insert
                    )       
                trans.commit()
                print(f"      > ✅ Saved {len(laps_to_insert)} laps from {s_name}")
            except Exception as e:
                trans.rollback()
                print(f"      !! Error saving race laps: {e}")

def process_sprint_data(year, round_num, race_id):
    # Check if already done
    if check_data_exists(race_id, 'sprint_laps'):
        print(f"      ⏭️  Skipping Sprint (Already in DB)")
        return

    print(f"      🏎️  Processing Sprint...")
    try:
        #extract API data
        session = fastf1.get_session(year, round_num, 'Sprint')
        session.load(telemetry=False, weather=True, messages=False)
        
        #process weather data
        process_weather_data(race_id, session, 'Sprint')
        
        try:
            fastest_lap_data = session.laps.pick_fastest()
            fastest_driver_id = fastest_lap_data['Driver']
        except Exception:
            fastest_driver_id = None
        
        #extract results
        results = session.results
        res_insert = []
        #store driver data
        for _, row in results.iterrows():
            is_fastest = False
            if fastest_driver_id and row['Abbreviation'] == fastest_driver_id:
                is_fastest = True
            res_insert.append({
                "rid": race_id, 
                "did": row['Abbreviation'], 
                "cid": row['TeamName'],
                "grid": safe_int(row['GridPosition']), 
                "fin": safe_int(row['Position']), 
                "pts": row['Points'],
                "stat": str(row['Status']), 
                "laps": safe_int(row['Laps']),
                "fl": is_fastest
            })

        #
        laps = session.laps.pick_quicklaps()
        laps_insert = []
        for _, lap in laps.iterrows():
            ltime = lap['LapTime'].total_seconds() if pd.notnull(lap['LapTime']) else None
            laps_insert.append({
                "rid": race_id, 
                "did": lap['Driver'], 
                "lnum": safe_int(lap['LapNumber']),
                "ltime": ltime, 
                "pos": safe_int(lap['Position']), 
                "comp": lap['Compound'], 
                "tlife": safe_int(lap['TyreLife'])
            })

        #store driver data and lap data to db
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                # Insert Sprint Results
                if res_insert:
                    conn.execute(text("""
                        INSERT INTO sprint_results (
                            race_id, driver_id, constructor_id, grid_position, 
                            finish_position, points, status, laps_completed, fastest_lap
                        ) VALUES (:rid, :did, :cid, :grid, :fin, :pts, :stat, :laps, :fl)
                    """), res_insert)
                
                # Insert Sprint Laps
                if laps_insert:
                    conn.execute(text("""
                        INSERT INTO sprint_laps (
                            race_id, driver_id, lap_number, lap_time, 
                            position, compound, tyre_life
                        ) VALUES (:rid, :did, :lnum, :ltime, :pos, :comp, :tlife)
                    """), laps_insert)
                trans.commit()
                print(f"      > ✅ Saved Sprint Data")
            except Exception as e:
                trans.rollback()
                print(f"      !! DB Error saving Sprint: {e}")

    except Exception as e:
        print(f"      !! API Error loading Sprint: {e}")

def process_qualifying(year, round_num, race_id):
    #check if we already store qualifying data
    if check_data_exists(race_id, "qualifying"):
        print(f"      ⏭️  Skipping Qualifying (Already in DB)")
        return
    
    print(f"      ⏱️  Processing Qualifying...")
    try:
        #extract session data from API
        session = fastf1.get_session(year, round_num, 'Qualifying')
        session.load(telemetry=False, weather=True, messages=False)
        #process weather data for the given session
        process_weather_data(race_id, session, 'Qualifying')
    except Exception as e:
        print(f"      !! API Error loading qualifying: {e}")
        return
    # store results data
    results = session.results
    qual_data_to_insert = []
    
    for driver_code, row in results.iterrows():
        q1 = row['Q1'].total_seconds() if pd.notnull(row['Q1']) else None
        q2 = row['Q2'].total_seconds() if pd.notnull(row['Q2']) else None
        q3 = row['Q3'].total_seconds() if pd.notnull(row['Q3']) else None
            
        qual_data_to_insert.append({
            "rid": race_id,
            "did": row['Abbreviation'],
            "pos": safe_int(row['Position']),
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "status": str(row['Status'])
        })
    #store qualifying data to DB
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            if qual_data_to_insert:
                conn.execute(
                    text(
                    """
                    INSERT INTO qualifying (
                        race_id, driver_id, position, 
                        q1_time, q2_time, q3_time, status
                    ) VALUES (
                        :rid, :did, :pos, 
                        :q1, :q2, :q3,
                        :status
                    )
                    """
                    ), 
                    qual_data_to_insert
                )       
                trans.commit()
                print(f"      > ✅ Saved Qualifying Results")
        except Exception as e:
            trans.rollback()
            print(f"      !! Error saving race laps: {e}")
    return

#function to process weather data given the parameters
def process_weather_data(race_id, session, session_type):
    #does weather data for the session already exist?
    if check_data_exists(race_id, 'weather', session_type):
        print(f"      ⏭️  Skipping {session_type} Weather (Already in DB)")
        return
    print(f"      ☁️  Processing {session_type} Weather...")

    try:
        #fetch session data from APU
        w_data = session.weather_data
        weather_to_insert = []
        
        for index, row in w_data.iterrows():
            # Convert Timedelta to Seconds (Float) to match lap data (reads in intervals of 90 seconds)
            t_offset = row['Time'].total_seconds() if pd.notnull(row['Time']) else 0.0
            
            weather_to_insert.append({
                "rid": race_id,
                "stype": session_type,
                "time": t_offset,
                "air": float(row['AirTemp']),
                "track": float(row['TrackTemp']),
                "hum": float(row['Humidity']),
                "rain": bool(row['Rainfall']),
                "ws": float(row['WindSpeed']) if pd.notnull(row.get('WindSpeed')) else None,
                "wd": safe_int(row.get('WindDirection'))
            })
        #insert array of weather data into DB
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                if weather_to_insert:
                    conn.execute(text("""
                        INSERT INTO weather (
                            race_id, session_type, time_offset, 
                            air_temp, track_temp, humidity, rainfall, 
                            wind_speed, wind_direction
                        ) VALUES (
                            :rid, :stype, :time, 
                            :air, :track, :hum, :rain, 
                            :ws, :wd
                        )
                    """), weather_to_insert)
                trans.commit()
                print(f"      > ✅ Saved {len(weather_to_insert)} weather records")
                
            except Exception as e:
                trans.rollback()
                print(f"      !! DB Error saving weather: {e}")

    except Exception as e:
        print(f"      !! API Error extracting weather: {e}")
    
def main():
    years = [2023, 2024, 2025]
    for year in years:
        print(f"\n📅 Checking Schedule for {year}...")
        try:
            #extract all races that occured in agiven season
            schedule = fastf1.get_event_schedule(year)
            
            # print("\n--- RAW SCHEDULE OBJECT ---")
            # print(schedule.head(5))
            
            # print("\n--- INTERESTING COLUMNS ---")
            # print(schedule[['RoundNumber', 'EventName', 'Session5Date', 'F1ApiSupport']].head())
            
        except Exception as e:
            print(f"!! Error fetching schedule for {year}: {e}")
            continue
        
        #for all weekends in a given season
        for i, row in schedule.iterrows():
            round_num = row['RoundNumber']
            event_name = row['EventName']

            if round_num == 0:
                continue

            try:
                #extract race data
                race_session = fastf1.get_session(year, round_num, 'R')

                #store race id (check for existance or create)
                race_id = get_or_create_race(year, round_num, race_session)

                print(f"  > Processing Round {round_num}: {event_name}")

                #check if data exists if not than we can extract race lap data
                if check_data_exists(race_id, 'laps'):
                    print("     ⏭️  Skipping Race Data (Already in DB)")

                else:
                    print("     > Loading Race Telemetry...")
                    race_session.load(telemetry=False, weather=True, messages=False)
                    process_race_data(race_id, race_session)
                    reassign_fastest_lap(engine, race_id, 'race')

                #find out how many practice sessions there are for a race weekends
                practice_sessions = []
                if 'Practice' in str(row['Session1']): practice_sessions.append('FP1')
                if 'Practice' in str(row['Session2']): practice_sessions.append('FP2')
                if 'Practice' in str(row['Session3']): practice_sessions.append('FP3')

                #load practice data
                if practice_sessions:
                    process_practice(year, round_num, race_id, practice_sessions)

                #load sprint data if it exists
                if 'Sprint' in str(row.values):
                    process_sprint_data(year, round_num, race_id)
                    reassign_fastest_lap(engine, race_id, 'sprint')

                #load qualifying data
                process_qualifying(year, round_num, race_id)

                time.sleep(3)
                print(f"     ✨ Weekend Complete: {event_name}")

            except Exception as e:
                print(f"     ⚠️  Skipping Round {round_num} ({event_name}): {e}")
                continue
            
if __name__ == '__main__':
    main()