from db.create_tables import main as create_tables
from db.ingestion.fastf1API import main as ingest_fastf1
from db.ingestion.positions import main as calc_positions
from db.ingestion.overtakes import main as calc_overtakes
from db.ingestion.pitstops import main as ingest_pitstops
from db.ingestion.dotd import main as ingest_dotd
from db.ingestion.constructors import main as calc_constructors
from db.ingestion.prices import main as ingest_prices
from db.prediction import populate_drivers_prediction_tables

def run_pipeline():
    print("🚀 Running F1 Data Pipeline...")
    #create_tables()
    #ingest_fastf1()
    calc_positions()
    calc_overtakes()
    ingest_dotd()
    calc_constructors()
    ingest_prices()
    ingest_pitstops()
    populate_drivers_prediction_tables()


    print("✅ Pipeline Complete.")

if __name__ == "__main__":
    run_pipeline()