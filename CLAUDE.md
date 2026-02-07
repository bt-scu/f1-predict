# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 Fantasy Points Prediction System - predicts qualifying, sprint, and race outcomes for Formula 1 using XGBoost models. Combines FastF1 telemetry data with historical CSV/JSON fantasy data to forecast driver positions and fantasy point totals.

## Commands

### Database Setup
```bash
# PostgreSQL runs on localhost:6000 (credentials: postgres/f1-pass)
# Create all tables
python db/create_tables.py

# Run full ETL pipeline (FastF1 ingestion + fantasy scoring)
python db/pipeline.py
```

### ML Model Training & Prediction
```bash
python ml/pred_quali.py    # Qualifying position predictor
python ml/pred_sprint.py   # Sprint position predictor
python ml/pred_race.py     # Race position predictor
python ml/ml.py            # Season simulation benchmark
```

### Data Inspection
```bash
python inspect_ff1.py      # Verify FastF1 schedule mappings
```

## Architecture

### Data Flow
```
FastF1 API + CSV/JSON files
        ↓
db/pipeline.py (ETL orchestrator)
        ↓
PostgreSQL tables (races, results, practice_laps, driver_fantasy_results, etc.)
        ↓
ml/pred_*.py (XGBoost models read from DB, write predictions back)
        ↓
driver_predictions table
```

### Key Directories
- `db/` - Database schema (`create_tables.py`) and ETL pipeline
- `db/ingestion/` - Data ingestion modules (fastf1API.py, positions.py, overtakes.py, etc.)
- `ml/` - XGBoost prediction models for quali/sprint/race
- `2023-data/`, `2024-data/` - Historical CSV data by driver/constructor
- `2025-data/` - JSON fantasy data per driver/constructor
- `ff1_cache/` - FastF1 API cache directory

### Database Schema (PostgreSQL)
Core tables: `drivers`, `races`, `results`, `sprint_results`, `qualifying`, `practice_laps`, `laps`, `pit_stops`, `weather`

Fantasy tables: `driver_fantasy_results`, `constructor_fantasy_results`

Prediction output: `driver_predictions` (pred_quali_pos, pred_race_pos, prob_dnf, ev_points, etc.)

### ML Models
All models use XGBoost regression. Key features across models:
- Practice lap times (adjusted to session minimum)
- Weather conditions (track/air temp)
- Team momentum (constructor average position lag 1-2)
- Teammate gap (relative performance)
- Lap time consistency (std dev lag 1-3)
- Historical overtakes
- Grid position

## Dependencies

Python 3.12 with: fastf1, sqlalchemy, pandas, numpy, xgboost, scikit-learn, psycopg2

Virtual environment: `f1venv/`

## Database Connection

All modules use: `postgresql://postgres:f1-pass@localhost:6000/postgres`
