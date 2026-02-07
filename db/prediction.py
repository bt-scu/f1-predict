from sqlalchemy import create_engine, text

# 1. Setup DB Connection
DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

def populate_drivers_prediction_tables():
    with engine.connect() as conn:
        # 2. Get 2025 Races
        query_races = text("SELECT race_id, circuit_name FROM races WHERE year = 2025 ORDER BY date ASC")
        races = conn.execute(query_races).fetchall()
        
        if not races:
            print("⚠️ No 2025 races found! (Did you run the ingestion script?)")
            return

        # Prepared statement for insertion
        insert_query = text("""
            INSERT INTO simulate_predictions (race_id, driver_id, constructor_id)
            VALUES (:race_id, :driver_id, :constructor_id)
            ON CONFLICT (race_id, driver_id) DO NOTHING
        """)

        total_rows = 0
        for race in races:
            driver_query = text("SELECT driver_id, constructor_id FROM results WHERE race_id = :r_id")
            drivers = conn.execute(driver_query, {"r_id": race.race_id}).fetchall()
            
            for driver_row in drivers:
                conn.execute(insert_query, {
                    "race_id": race.race_id, 
                    "constructor_id": driver_row.constructor_id,
                    "driver_id": driver_row.driver_id
                })
                total_rows += 1
            conn.commit()
            print(f"📍 Race {race.circuit_name} ({race.race_id}): Seeded {len(drivers)} drivers.")

        print(f"✅ Done. Total prediction rows created: {total_rows}")

if __name__ == "__main__":
    populate_drivers_prediction_tables()
