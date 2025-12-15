from sqlalchemy import create_engine, text
import math

DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)

def create_fantasy_tables():
  print("--- CREATING FANTASY TABLE ---")
  with engine.connect() as conn:
    trans = conn.begin()
    try:
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
          overtake_points INTEGER DEFAULT 0,
          fl_points INTEGER DEFAULT 0,
          
          total_points INTEGER DEFAULT 0,
          
          FOREIGN KEY (race_id) REFERENCES races(race_id)
        )
        """
       )
      )
      
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
          
          FOREIGN KEY (race_id) REFERENCES races(race_id)
        )
        """
       )
      )
      trans.commit()
    except Exception as e:
      trans.rollback()
      print(f"   !! Error creating table: {e}")

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


def calculate_fantasy_results(race_id, race_year, race_event_name):
  #maps to map position finishes -> points attained
  sprint_pt_map = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
  race_pt_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
  quali_pt_map = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}

  print(f"  🏁 Calculating: {race_year} {race_event_name}...")

  sprint_map = {}
  with engine.connect() as conn:
    trans = conn.begin()
    try:
      # fetch query data
      q_query = text("SELECT driver_id, position FROM qualifying WHERE race_id = :rid")
      q_rows = conn.execute(q_query, {"rid": race_id}).fetchall()
      #organzie quali data in to a mapping
      quali_map = {row.driver_id: row.position for row in q_rows}

      # fetch sprint data
      s_query = text("SELECT driver_id, finish_position, grid_position, status, overtakes FROM sprint_results WHERE race_id = :rid")
      s_rows = conn.execute(s_query, {"rid": race_id}).fetchall()

      #organize sprint data into a dict
      for row in s_rows:
        sprint_map[row.driver_id] = {
          'finish': row.finish_position,
          'grid': row.grid_position,
          'status': row.status,
          'overtake': row.overtakes
        }

      # fetch race data
      r_query = text(
        """
        SELECT driver_id, constructor_id, grid_position, finish_position, fastest_lap, status, overtakes
        FROM results 
        WHERE race_id = :rid
        """
        
      )
      r_rows = conn.execute(r_query, {"rid": race_id}).fetchall()

      driver_inserts = []
      constructor_inserts = []
      constructor_tracker = {}

      #traverse race data to populate driver's fantasy result tables
      for row in r_rows:
        curr_driver = row.driver_id
        curr_constructor = row.constructor_id
        status = row.status

        #prep to store meta data for constructor point calculator
        if curr_constructor not in constructor_tracker:
          constructor_tracker[curr_constructor] = {
            'q_positions': [],
            'sprint_score': 0,
            'race_score': 0,
            'dq_count': 0
          }

        #calculating quali results
        quali_pos = quali_map.get(curr_driver)
        quali_pts = 0
        #case driver does not quali
        if quali_pos is not None:
          quali_pts = quali_pt_map.get(quali_pos, 0)
          constructor_tracker[curr_constructor]['q_positions'].append(quali_pos)
        else:
          quali_pts = -5
          constructor_tracker[curr_constructor]['q_positions'].append(20)

        # calculate sprint results
        sprint_pts = 0
        sprint_overtake_pts = 0
        if len(sprint_map) > 0:
          if curr_driver in sprint_map:
            s_data = sprint_map[curr_driver]
            s_fin = s_data['finish']
            s_grid = s_data['grid']
            s_stat = str(s_data['status'])
            s_overtakes = s_data['overtake']

            safe_s_grid = s_grid if s_grid is not None else 20
            safe_s_fin = s_fin if s_fin is not None else 20
            # check if driver finished in the pts
            if s_fin in sprint_pt_map:
              sprint_pts += sprint_pt_map[s_fin]
            
            # extra check to handle case where driver starts from pits
            calc_s_grid = safe_s_grid if safe_s_grid > 0 else 20

            # check to see if driver made overtakes
            if s_fin:
              sprint_pts += (calc_s_grid - s_fin)
            is_valid_finish = ('Finished' in s_stat) or ('Lapped' in s_stat) or ('+' in s_stat)
            
            sprint_overtake_pts = s_overtakes
            sprint_pts += sprint_overtake_pts
            # check to see if driver dnf
            if not is_valid_finish:
              sprint_pts += -20
            #check to see if driver disq
            if 'Disqualified' in str(s_stat):
              sprint_pts = -20 
              constructor_tracker[curr_constructor]['dq_count'] += 1

        race_pts = 0
        pos_gain_pts = 0
        race_overtake_pts = 0
        

        fin_pos = row.finish_position
        r_grid = row.grid_position

        safe_r_grid = r_grid if r_grid is not None else 20
        safe_fin_pos = fin_pos if fin_pos is not None else 20
        
        # check to see if driver finishes in the pts
        race_pts += race_pt_map.get(safe_fin_pos, 0)

        # ensure the handle case that driver starts from pitlane
        calc_r_grid = safe_r_grid if safe_r_grid > 0 else 20  


        if fin_pos and fin_pos > 0:
          pos_gain_pts = calc_r_grid - safe_fin_pos

        # Fastest Lap Bonus
        if row.fastest_lap:
          race_pts += 10
        
        r_overtakes = row.overtakes if row.overtakes else 0
        race_overtake_pts = r_overtakes
        
        total_overtake_pts = sprint_overtake_pts + race_overtake_pts

        is_valid_finish = ('Finished' in status) or ('Lapped' in status) or ('+' in status)
        
        #case driver is disqualified in race
        if 'Disqualified' in str(status):
          race_pts = -20
          pos_gain_pts = 0
          race_overtake_pts = 0
          constructor_tracker[curr_constructor]['dq_count'] += 1
          
        elif not is_valid_finish:
          race_pts = -20
          pos_gain_pts = 0
  

        # total points driver gained
        total_pts = quali_pts + sprint_pts + race_pts + pos_gain_pts + race_overtake_pts

        #store value in constructor
        constructor_tracker[curr_constructor]['sprint_score'] += sprint_pts
        constructor_tracker[curr_constructor]['race_score'] += (race_pts + pos_gain_pts)

        #prep to add driver
        driver_inserts.append({
            "race_id": race_id,
            "driver_id": curr_driver,
            "constructor_id": curr_constructor,
            "quali_points": quali_pts,
            "sprint_points": sprint_pts,
            "race_points": race_pts + pos_gain_pts,
            "overtake_points": total_overtake_pts,
            "total_points": total_pts
        })
      
      #Dealing with constructors
      for team_id, data in constructor_tracker.items():
          #see if constructor earned qualifying point bonuses
          q_bonus = 0
          q_list = data['q_positions']
          in_q3 = sum(1 for p in q_list if p <= 10)
          in_q2 = sum(1 for p in q_list if p <= 15)

          if in_q3 == 2: q_bonus = 10
          elif in_q3 == 1: q_bonus = 5
          elif in_q2 == 2: q_bonus = 3
          elif in_q2 == 1: q_bonus = 1
          else: q_bonus = -1

          #sum up driver qualifying sums
          driver_q_sum = sum(d['quali_points'] for d in driver_inserts if d['constructor_id'] == team_id)

          #add q bonus
          final_q_score = driver_q_sum + q_bonus

          # calculate penalties
          penalty_score = data['dq_count'] * -10

          #calculate sprints and race_score
          final_s_score = data['sprint_score']
          final_r_score = data['race_score'] + penalty_score

          #total point constructor acheived
          total_cons_pts = final_q_score + final_s_score + final_r_score

          # populate table
          constructor_inserts.append({
              "race_id": race_id,
              "constructor_id": team_id,
              "quali_score": final_q_score,
              "sprint_score": final_s_score,
              "race_score": final_r_score,
              "pitstop_points": 0,
              "total_points": total_cons_pts
          })
    
      if driver_inserts:
          conn.execute(text("""
              INSERT INTO driver_fantasy_results (race_id, driver_id, constructor_id, quali_points, sprint_points, race_points, overtake_points, total_points)
              VALUES (:race_id, :driver_id, :constructor_id, :quali_points, :sprint_points, :race_points, :overtake_points,:total_points)
          """), driver_inserts)

      if constructor_inserts:
          conn.execute(text("""
              INSERT INTO constructor_fantasy_results (race_id, constructor_id, quali_score, sprint_score, race_score, pitstop_points, total_points)
              VALUES (:race_id, :constructor_id, :quali_score, :sprint_score, :race_score, :pitstop_points, :total_points)
          """), constructor_inserts)

      trans.commit()
      print(f"   ✅ Processed {len(driver_inserts)} drivers and {len(constructor_inserts)} teams.")

    except Exception as e: 
        trans.rollback()
        print(f"Error calculating {race_event_name}: {e}")
        return

def main():
  #create the fantasy table (they will hold the results of our data)
  create_fantasy_tables()
  
  races = extract_races()
  print(f"📊 Found {len(races)} races to process...")
  
  
  for row in races:
    calculate_fantasy_results(row["id"], row["year"], row["event_name"])
  return


if __name__ == "__main__":
    main()