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
                    CREATE TABLE IF NOT EXISTS driver_predictions (
                        prediction_id SERIAL PRIMARY KEY,
                        race_id INTEGER NOT NULL,
                        driver_id TEXT NOT NULL,

                        pred_quali_pos INTEGER,
                        ev_quali_pos FLOAT,
                        prob_quali_nc FLOAT,
                        ev_quali_points FLOAT,
                        quali_factors_json JSONB,

                        pred_sprint_start_pos INTEGER,
                        pred_sprint_pos INTEGER,
                        pred_sprint_overtakes FLOAT,
                        prob_sprint_dnf FLOAT,
                        ev_sprint_pos FLOAT,
                        ev_sprint_points FLOAT,

                        pred_race_pos INTEGER,
                        pred_race_pos_p10 INTEGER,
                        pred_race_pos_p90 INTEGER,
                        pred_race_overtakes FLOAT,
                        prob_race_dnf FLOAT,
                        prob_race_fastest_lap FLOAT,
                        prob_race_dotd FLOAT,
                        ev_race_pos FLOAT,
                        ev_race_points FLOAT,
                        race_factors_json JSONB,

                        risk_flags_json JSONB,
                        weather_source TEXT,

                        total_projected_points FLOAT,

                        model_version TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),

                        FOREIGN KEY (race_id) REFERENCES races(race_id),
                        UNIQUE(race_id, driver_id)
                    )
                    """
                )
            )

            # ============================================================
            # RAG TABLES (pgvector-backed)
            # ============================================================

            # Enable pgvector (idempotent -- safe to run repeatedly)
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # 14. NEWS ARTICLES -- raw ingested content
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS news_articles (
                        article_id    SERIAL PRIMARY KEY,
                        source        TEXT NOT NULL,
                        url           TEXT UNIQUE NOT NULL,
                        title         TEXT,
                        author        TEXT,
                        content       TEXT NOT NULL,
                        published_at  TIMESTAMP WITH TIME ZONE,
                        ingested_at   TIMESTAMP DEFAULT NOW(),
                        driver_codes  TEXT[],
                        circuit_name  TEXT,
                        content_type  TEXT DEFAULT 'article',
                        word_count    INTEGER,
                        content_hash  TEXT UNIQUE
                    )
                    """
                )
            )

            # 15. NEWS CHUNKS -- chunked text with vector embeddings
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS news_chunks (
                        chunk_id      SERIAL PRIMARY KEY,
                        article_id    INTEGER REFERENCES news_articles(article_id) ON DELETE CASCADE,
                        chunk_index   INTEGER NOT NULL,
                        chunk_text    TEXT NOT NULL,
                        token_count   INTEGER,
                        embedding     vector(384),
                        driver_codes  TEXT[],
                        created_at    TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
            )

            # HNSW index for fast approximate nearest neighbor search
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                        ON news_chunks USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200)
                    """
                )
            )

            # Indexes for metadata-filtered searches
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_chunks_drivers ON news_chunks USING gin (driver_codes)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_chunks_article ON news_chunks (article_id)")
            )

            # 16. INGESTION LOG -- tracks each scraping run
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS ingestion_log (
                        log_id          SERIAL PRIMARY KEY,
                        source          TEXT NOT NULL,
                        run_at          TIMESTAMP DEFAULT NOW(),
                        articles_found  INTEGER DEFAULT 0,
                        articles_new    INTEGER DEFAULT 0,
                        chunks_created  INTEGER DEFAULT 0,
                        errors          JSONB,
                        duration_sec    FLOAT
                    )
                    """
                )
            )

            # 17. RETRIEVAL FEEDBACK -- logs queries for evaluation
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS retrieval_feedback (
                        feedback_id       SERIAL PRIMARY KEY,
                        query_text        TEXT,
                        query_embedding   vector(384),
                        retrieved_chunk_ids INTEGER[],
                        llm_response      TEXT,
                        user_rating       INTEGER CHECK (user_rating BETWEEN 1 AND 5),
                        created_at        TIMESTAMP DEFAULT NOW()
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