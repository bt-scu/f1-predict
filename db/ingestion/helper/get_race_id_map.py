from sqlalchemy import text

#function to create mappings from race_ids given round and year
def get_race_id_map(conn):
 mapping = {}
 try:
  query = text("SELECT race_id, year, round FROM races")
  result = conn.execute(query)
  
  for row in result:
   key = (int(row.year), int(row.round))
   mapping[key] = row.race_id
   
 except Exception as e:
  print(f"⚠️ Error building race map: {e}")
  
 return mapping