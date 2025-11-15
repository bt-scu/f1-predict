import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine  # <-- NEW: Import the engine

# --- NEW: DATABASE CONFIG ---
# 1. Database Connection String (Your "address")
DB_URL = 'postgresql://postgres:f1-pass@localhost:6000/postgres'

# 2. SQL Query to select our data
#    This query *is* our data loading AND part of our cleaning
SQL_QUERY = """
SELECT
    "grid",
    "position",
    "constructorId",
    "driverId",
    "rank",
    "statusId"
FROM
    results
WHERE
    "position" IS NOT NULL
    AND "grid" > 0
    AND "rank" IS NOT NULL;
"""

# --- SCRIPT ---

print("Connecting to database...")
# Create the "engine" that SQLAlchemy uses to connect
engine = create_engine(DB_URL)

# --- 1. Load the data ---
# We replace pd.read_csv with pd.read_sql_query
print("Loading data from PostgreSQL...")
df = pd.read_sql_query(SQL_QUERY, engine)

# --- 2. Clean the data ---
# The SQL query already handled the hardest parts (filtering NULLs and grid=0)!
# We just need to ensure the types are correct.
df['position'] = df['position'].astype(int)
df['grid'] = df['grid'].astype(int)
df['rank'] = df['rank'].astype(int)
df['statusId'] = df['statusId'].astype(int)

# 3. Define Features and Target
features = [
    'grid',
    'constructorId',
    'driverId',
    'rank',
    'statusId'
]
target = 'position'

X = df[features]
y = df[target]

# 4. Create the Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build and Train the V1.5 Model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

print("Training model... (This may take 10-20 seconds)")
model.fit(X_train, y_train)

# 6. Evaluate the Model
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print("---" * 10)
print(f"✅ V1.5 Model (Full Raw DB): R-squared: {score:.4f}")