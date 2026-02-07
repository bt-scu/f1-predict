from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

def main():
    print("--- CREATING TABLES ---")
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # 1. DRIVERS
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS drivers (
                        driver_id TEXT PRIMARY KEY, 
                        driver_number INTEGER, 
                        full_name TEXT, 
                        team_name TEXT
                    )
                    """
                )
            )

            # 2. RACES
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS races (
                        race_id SERIAL PRIMARY KEY, 
                        year INTEGER, 
                        round INTEGER, 
                        circuit_name TEXT, 
                        event_name TEXT, 
                        date TIMESTAMP
                    )
                    """
                )
            )

            # 3. RACE RESULTS (Sunday)
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS results (
                        result_id SERIAL PRIMARY KEY,
                        race_id INTEGER, 
                        driver_id TEXT, 
                        constructor_id TEXT, 
                        grid_position INTEGER, 
                        finish_position INTEGER, 
                        points FLOAT, 
                        laps_completed INTEGER, 
                        status TEXT, 
                        fastest_lap BOOLEAN, 
                        driver_of_the_day BOOLEAN
                    )
                    """
                )
            )

            # 4. SPRINT RESULTS
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS sprint_results (
                        sprint_id SERIAL PRIMARY KEY,
                        race_id INTEGER,
                        driver_id TEXT,
                        constructor_id TEXT,
                        grid_position INTEGER,
                        finish_position INTEGER,
                        points FLOAT,
                        status TEXT,
                        laps_completed INTEGER,
                        fastest_lap BOOLEAN
                    )
                    """
                )
            )

            # 5. QUALIFYING
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS qualifying (
                        qualifying_id SERIAL PRIMARY KEY,
                        race_id INTEGER,
                        driver_id TEXT,
                        position INTEGER,
                        q1_time FLOAT,
                        q2_time FLOAT,
                        q3_time FLOAT,
                        status TEXT
                        
                    )
                    """
                )
            )

            # 6. PRACTICE LAPS
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS practice_laps (
                        practice_lap_id SERIAL PRIMARY KEY,
                        race_id INTEGER,
                        driver_id TEXT,
                        session_type TEXT,
                        lap_number INTEGER,
                        lap_time FLOAT,
                        sector_1 FLOAT,
                        sector_2 FLOAT,
                        sector_3 FLOAT,
                        compound TEXT,
                        tyre_life INTEGER,
                        stint INTEGER
                    )
                    """
                )
            )

            # 7. RACE LAPS
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS laps (
                        lap_id SERIAL PRIMARY KEY,
                        race_id INTEGER,
                        driver_id TEXT,
                        lap_number INTEGER,
                        lap_time FLOAT,
                        position INTEGER,
                        compound TEXT,
                        tyre_life INTEGER
                    )
                    """
                )
            )

            # 8. PIT STOPS
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS pit_stops (
                        pit_stop_id SERIAL PRIMARY KEY,
                        race_id INTEGER,
                        driver_id TEXT,
                        lap_number INTEGER,
                        duration FLOAT,
                        total_pit_time FLOAT
                    )
                    """
                )
            )
            
            # 9. WEATHER
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS weather (
                        weather_id SERIAL PRIMARY KEY,
                        race_id INTEGER,
                        session_type TEXT,
                        time_offset FLOAT,
                        air_temp FLOAT,
                        track_temp FLOAT,
                        humidity FLOAT,
                        rainfall BOOLEAN,
                        wind_speed FLOAT,
                        wind_direction INTEGER
                    )
                    """
                )
            )
            
            # 10. SPRINT LAPS
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS sprint_laps (
                        sprint_lap_id SERIAL PRIMARY KEY,
                        race_id INTEGER,
                        driver_id TEXT,
                        lap_number INTEGER,
                        lap_time FLOAT,
                        position INTEGER,
                        compound TEXT,
                        tyre_life INTEGER,
                        weather_data_id INTEGER
                    )
                    """
                )
            )
            
            # 11. DRIVER FANTASY RESULTS
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS driver_fantasy_results (
                        id SERIAL PRIMARY KEY,
                        driver_id TEXT,
                        race_id INTEGER,
                        constructor_id TEXT,
                        
                        quali_points INTEGER DEFAULT 0,
                        sprint_points INTEGER DEFAULT 0,
                        race_points INTEGER DEFAULT 0,
                        race_pos_gain_points INTEGER DEFAULT 0,
                        fl_points INTEGER DEFAULT 0,
                        
                        total_points INTEGER DEFAULT 0,
                        overtakes INTEGER DEFAULT 0,
                        sprint_overtakes INTEGER DEFAULT 0,
                        start_price NUMERIC(4, 1) DEFAULT 0.0,
                        dotd BOOLEAN DEFAULT NULL,
                        
                        FOREIGN KEY (race_id) REFERENCES races(race_id)
                    )
                    """
                )
            )
            
            # 12. CONSTRUCTOR FANTASY RESULTS
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS constructor_fantasy_results (
                        id SERIAL PRIMARY KEY,
                        race_id INTEGER,
                        constructor_id TEXT,
                        
                        quali_score INTEGER DEFAULT 0,  
                        sprint_score INTEGER DEFAULT 0,
                        race_score INTEGER DEFAULT 0,   
                        
                        pitstop_points INTEGER DEFAULT 0,
                        q_bonus_points INTEGER DEFAULT 0,
                        
                        total_points INTEGER DEFAULT 0,
                        start_price NUMERIC(4, 1) DEFAULT 0.0,
                        
                        FOREIGN KEY (race_id) REFERENCES races(race_id)
                    )
                    """
                )
            )
            
            # 13. DRIVER PREDICTIONS (The "Crystal Ball" Table)
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS driver_fantasy_results (
                        prediction_id SERIAL PRIMARY KEY,
                        race_id INTEGER NOT NULL,
                        driver_id TEXT NOT NULL,

                        -- ⏱️ QUALIFYING (GP Grid)
                        pred_quali_pos INTEGER,         -- The "Most Likely" outcome (e.g., P4)
                        ev_quali_pos FLOAT,
                        prob_quali_nc FLOAT,            -- Risk of "No Time Set"
                        ev_quali_points FLOAT,          -- Weighted Average (The "True Value")
                        quali_factors_json JSONB,       -- SHAP-based top 5 factors

                        -- 🏎️ SPRINT (Sprint Grid & Race)
                        pred_sprint_start_pos INTEGER,
                        pred_sprint_pos INTEGER,
                        pred_sprint_overtakes FLOAT,
                        prob_sprint_dnf FLOAT,
                        ev_sprint_pos FLOAT,
                        ev_sprint_points FLOAT,

                        -- 🏁 GRAND PRIX (The Main Race)
                        pred_race_pos INTEGER,          -- Most Likely Finish
                        pred_race_pos_p10 INTEGER,      -- 10th percentile (optimistic)
                        pred_race_pos_p90 INTEGER,      -- 90th percentile (pessimistic)
                        pred_race_overtakes FLOAT,      -- Expected count
                        prob_race_dnf FLOAT,            -- The "Portfolio Killer" risk
                        prob_race_fastest_lap FLOAT,    -- Probability (0.0 - 1.0)
                        prob_race_dotd FLOAT,           -- Probability (0.0 - 1.0)
                        ev_race_pos FLOAT,
                        ev_race_points FLOAT,           -- Weighted Average of everything above
                        race_factors_json JSONB,        -- SHAP-based top 5 factors

                        -- 🔮 RISK & EXPLAINABILITY
                        risk_flags_json JSONB,          -- {"rain_sensitive": 0.7, "first_lap_risk": -0.3, ...}
                        weather_source TEXT,            -- 'historical' or 'forecast'

                        -- 💰 TOTALS
                        total_projected_points FLOAT,   -- The Sum of all EVs (The Ranking Metric)

                        model_version TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),

                        FOREIGN KEY (race_id) REFERENCES races(race_id),
                        UNIQUE(race_id, driver_id)
                    )
                    """
                )
            )

            trans.commit()
            print("--- TABLES CREATED ---")
        except Exception as e:
            trans.rollback()
            print(f"Error: {e}")

main()