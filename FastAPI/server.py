
"""
server.py -- FastAPI server that runs the F1 data pipeline on a schedule.

Start it and go to sleep:
    python server.py

It will:
    1. Run the pipeline immediately on startup
    2. Re-run every 30 minutes
    3. Skip any data already in the DB (your check_data_exists handles this)
    4. Log progress to the console

Check status:
    GET http://localhost:8000/status

Trigger a run manually:
    POST http://localhost:8000/run

Stop it: Ctrl+C
"""

import asyncio
import threading
import time
from datetime import datetime

from fastapi import FastAPI
import uvicorn

from db.create_tables import main as create_tables
from db.ingestion.fastf1API import main as ingest_fastf1

app = FastAPI(title="F1 Pipeline Runner")

# Track pipeline state so you can check on it
pipeline_status = {
    "running": False,
    "last_run_start": None,
    "last_run_end": None,
    "last_error": None,
    "total_runs": 0,
}

INTERVAL_MINUTES = 30


def execute_pipeline():
    """Run the pipeline in a thread-safe way."""
    if pipeline_status["running"]:
        print("⏳ Pipeline already running, skipping this cycle.")
        return

    pipeline_status["running"] = True
    pipeline_status["last_run_start"] = datetime.now().isoformat()
    pipeline_status["last_error"] = None

    try:
        print(f"\n{'='*60}")
        print(f"🚀 Pipeline run #{pipeline_status['total_runs'] + 1} starting at {datetime.now()}")
        print(f"{'='*60}")
        create_tables()   # ensure tables exist on the new DB
        ingest_fastf1()   # the heavy FastF1 API work
        pipeline_status["total_runs"] += 1
        print(f"\n✅ Pipeline finished at {datetime.now()}")
    except Exception as e:
        pipeline_status["last_error"] = str(e)
        print(f"\n❌ Pipeline error: {e}")
    finally:
        pipeline_status["running"] = False
        pipeline_status["last_run_end"] = datetime.now().isoformat()


def scheduler_loop():
    """Background thread that runs the pipeline every INTERVAL_MINUTES."""
    while True:
        execute_pipeline()
        print(f"\n💤 Sleeping {INTERVAL_MINUTES} minutes until next run...")
        time.sleep(INTERVAL_MINUTES * 60)


@app.on_event("startup")
async def start_scheduler():
    """Launch the scheduler thread when FastAPI starts."""
    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()
    print(f"📡 Scheduler started: pipeline will run every {INTERVAL_MINUTES} minutes")


@app.get("/status")
def get_status():
    """Check pipeline status from your browser or curl."""
    return pipeline_status


@app.post("/run")
def trigger_run():
    """Manually trigger a pipeline run (non-blocking)."""
    if pipeline_status["running"]:
        return {"message": "Pipeline is already running"}
    thread = threading.Thread(target=execute_pipeline, daemon=True)
    thread.start()
    return {"message": "Pipeline started"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)