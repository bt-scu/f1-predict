import pandas as pd
from sqlalchemy import create_engine, text

def calculate_overtakes(laps_df):
  #sort values by driver_id alphabetical + by lap #
  laps_df = laps_df.sort_values(['driver_id', 'lap_number'])
  
  #dict to keep track of driver overtakes
  counts = {d: 0 for d in laps_df['driver_id'].unique()}
  
  if laps_df.empty:
    return counts
  
  max_lap = int(laps_df['lap_number'].max())
  
  # start on lap 1 since we check prev laps
  for lap in range(2, max_lap+1):
    prev = laps_df[laps_df['lap_number'] == lap - 1] #extracts lap data for the previous lap
    curr = laps_df[laps_df['lap_number'] == lap] #extracts lap data for the current lap
    
    merged = pd.merge(prev, curr, on='driver_id', suffixes=('_prev', '_curr')) # perform a pandas merge
    
    attackers = merged[merged['position_curr'] < merged['position_prev']] # if you are a driver whose current position is less than prev than you've performed an overtake within the lap
    
    # for all attackers
    for _, att in attackers.iterrows():
      attacker_id = att['driver_id']
      new_pos = att['position_curr']
      old_pos = att['position_prev']
            
      #identify victems that attacker passed
      victims = merged[
        (merged['position_prev'] >= new_pos) & # keeps drivers below the attacker
        (merged['position_prev'] < old_pos) & # keeps drivers who were originally ahead of attacker
        (merged['position_curr'] > new_pos) # is this victem now behind out attacker
      ]
      
      # check if victem got passed in the pits (does not count as on track overtake)
      for _, vic in victims.iterrows():
        victim_pitted = False
        t_curr = vic['tyre_life_curr']
        t_prev = vic['tyre_life_prev']
        
        if pd.notna(t_curr) and pd.notna(t_prev) and t_curr < t_prev:
          victim_pitted = True
              
        if vic['compound_curr'] != vic['compound_prev']:
          victim_pitted = True
                    
        if not victim_pitted:
          counts[attacker_id] += 1  
  return counts
  

def process_race_overtakes(conn):
  print("--- Checking for unprocessed races ---")
  #extract all race_id where overtakes  has not been calculated
  query = text("SELECT DISTINCT race_id FROM results where overtakes is NULL")
  pending_races = pd.read_sql(query, conn)['race_id'].tolist()
  
  if not pending_races:
    print("All races have been updated! No work to do.")
    return

  print(f"Found {len(pending_races)} races to process: {pending_races}")
  
  #for every race we need to process, select all drivers and perform the caculate overtake function on them
  for race_id in pending_races:
    query_laps = text(
      """
      SELECT driver_id, lap_number, position, tyre_life, compound 
      FROM laps 
      WHERE race_id = :rid
      """
    )
    
    laps_df = pd.read_sql(query_laps, conn, params={"rid": race_id})
    
    overtake_results = calculate_overtakes(laps_df)
    print(f"  -> Updating DB for Race {race_id}...")
    try:
        #update race results table with the drivers who made overtakes and ensure that null is not a value in the column
        for driver_id, count in overtake_results.items():
          update_query = text(
            """
            UPDATE results 
            SET overtakes = :count 
            WHERE race_id = :rid AND driver_id = :did
            """
          )
          conn.execute(update_query, {"count": count, "rid": race_id, "did": driver_id})
        cleanup_query = text(
          """
          UPDATE results 
          SET overtakes = 0 
          WHERE race_id = :rid AND overtakes IS NULL
          """
        )
        conn.execute(cleanup_query, {"rid": race_id})
        conn.commit()
        print("  -> Done.")
    except Exception as e:
      conn.rollback()
      print(f"  -> Error updating Race {race_id}: {e}")
  return

def process_sprint_overtakes(conn):
  print("--- Checking for unprocessed races ---")
  #extract all race_id where overtakes  has not been calculated
  query = text("SELECT DISTINCT race_id FROM sprint_results where overtakes is NULL")
  pending_races = pd.read_sql(query, conn)['race_id'].tolist()
  
  #if we have no races to process we are done
  if not pending_races:
    print("All races have been updated! No work to do.")
    return

  print(f"Found {len(pending_races)} races to process: {pending_races}")
  
  # for all races
  for race_id in pending_races:
    # select drivers within the race
    query_laps = text(
      """
      SELECT driver_id, lap_number, position, tyre_life, compound 
      FROM sprint_laps
      WHERE race_id = :rid
      """
    )
    laps_df = pd.read_sql(query_laps, conn, params={"rid": race_id})
    
    #calculate overtakes done
    overtake_results = calculate_overtakes(laps_df)
    
    
    print(f"  -> Updating DB for Race {race_id}...")
    try:
        # parse through the return dict
        for driver_id, count in overtake_results.items():
          # update sprint_results table
          update_query = text(
            """
            UPDATE sprint_results 
            SET overtakes = :count 
            WHERE race_id = :rid AND driver_id = :did
            """
          )
          conn.execute(update_query, {"count": count, "rid": race_id, "did": driver_id})
        #drivers who did not make overtakes set overtakes to 0
        cleanup_query = text(
          """
          UPDATE sprint_results 
          SET overtakes = 0 
          WHERE race_id = :rid AND overtakes IS NULL
          """
        )
        conn.execute(cleanup_query, {"rid": race_id})
        conn.commit()
        print("  -> Done.")
    except Exception as e:
      conn.rollback()
      print(f"  -> Error updating Race {race_id}: {e}")
  return



def main():
  engine = create_engine("postgresql://postgres:f1-pass@localhost:6000/postgres")
  with engine.connect() as conn:
    process_race_overtakes(conn)
    
    process_sprint_overtakes(conn)


if __name__ == "__main__":
  main()