import pandas as pd
from sqlalchemy import create_engine
import sys

# --- CONFIGURATION ---

# 1. Database Connection String (Your "address")
#    Make sure to use your port 6000!
DB_URL = 'postgresql://postgres:f1-pass@localhost:6000/postgres'

# 2. Filepath to your CSV
CSV_FILE = 'f1data/results.csv'

# 3. Name of the table you just created in SQL
#    (We used 'results', but change this if you named it 'f1_csv_results')
TABLE_NAME = 'results'

# --- SCRIPT ---
try:
    print("Connecting to database...")
    # Create the "engine" that SQLAlchemy uses to connect
    engine = create_engine(DB_URL)

    print(f"Reading CSV file: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    # CRITICAL CLEANING STEP:
    # Replace the CSV's '\N' text with Python's 'None' (which SQL understands as 'NULL')
    df = df.replace('\\N', None)

    print(f"Loading {len(df)} rows into '{TABLE_NAME}' table...")
    
    # This is the magic command!
    # - 'if_exists='append'' means "add this data, don't overwrite"
    # - 'index=False' means "don't save the pandas index as a column"
    df.to_sql(TABLE_NAME, engine, if_exists='append', index=False)

    print("---" * 10)
    print("✅ Success! Data has been loaded into your PostgreSQL database.")
    print("You can now see the data in your VS Code extension.")

except Exception as e:
    print(f"❌ An error occurred: {e}")
    sys.exit(1)