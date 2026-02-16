from sqlalchemy import text
from db.config import engine

#a function to extract all races that occured within the given season
def extract_races():
  races = [] 
  with engine.connect() as conn:
    try:
      result = conn.execute(text("SELECT race_id, year, event_name FROM races ORDER BY date ASC"))
      for row in result:
        races.append({
          "id": row.race_id,
          "year": row.year,
          "event_name": row.event_name
        })
    except Exception as e:
      print(f"Error extracting races: {e}")
      return []
  return races