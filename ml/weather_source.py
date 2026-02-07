"""
Weather source module for F1 predictions.

Provides a unified interface for fetching weather data:
- Historical data from FastF1 (stored in database)
- Future races: placeholder for OpenMeteo API integration

This decoupling allows training on historical data while supporting
future predictions with forecast weather.
"""

from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Optional

DB_URL = "postgresql://postgres:f1-pass@localhost:6000/postgres"
engine = create_engine(DB_URL)


def get_race_weather(race_id: int, race_date: Optional[datetime] = None) -> dict:
    """
    Unified weather interface for both historical and future races.

    Args:
        race_id: The race ID from the database
        race_date: Optional date for the race (used for forecast lookups)

    Returns:
        dict: {
            "track_temp": float,
            "air_temp": float,
            "is_forecast": bool  # True if using forecast data
        }
    """
    # Convert numpy types to native Python int (psycopg2 compatibility)
    race_id = int(race_id)

    # Try to get historical data first
    historical = _fetch_historical_weather(race_id)
    if historical:
        return {**historical, "is_forecast": False}

    # If no historical data, this is a future race
    # TODO: Replace with OpenMeteo API call for actual forecasts
    forecast = _get_mock_forecast(race_id, race_date)
    return {**forecast, "is_forecast": True}


def _fetch_historical_weather(race_id: int) -> Optional[dict]:
    """
    Fetch actual weather from FastF1 data stored in the database.

    Args:
        race_id: The race ID to fetch weather for

    Returns:
        dict with track_temp and air_temp, or None if not found
    """
    query = text("""
        SELECT AVG(track_temp) as track_temp, AVG(air_temp) as air_temp
        FROM weather
        WHERE race_id = :rid AND session_type = 'Race'
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"rid": race_id}).fetchone()
        if result and result.track_temp:
            return {"track_temp": float(result.track_temp), "air_temp": float(result.air_temp)}
    return None


def _get_mock_forecast(race_id: int, race_date: Optional[datetime]) -> dict:
    """
    PLACEHOLDER: Returns average conditions for the circuit.

    TODO: Replace with actual OpenMeteo API integration.

    Future implementation should:
    1. Get circuit lat/long from races table
    2. Call OpenMeteo: https://api.open-meteo.com/v1/forecast?latitude=X&longitude=Y
    3. Extract temperature for race_date + 14:00 local time

    Args:
        race_id: The race ID
        race_date: The date of the race

    Returns:
        dict with track_temp and air_temp estimates
    """
    # Fallback: Use circuit's historical average
    query = text("""
        SELECT AVG(w.track_temp) as avg_track, AVG(w.air_temp) as avg_air
        FROM weather w
        JOIN races r ON w.race_id = r.race_id
        WHERE r.circuit_name = (SELECT circuit_name FROM races WHERE race_id = :rid)
        AND w.session_type = 'Race'
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"rid": race_id}).fetchone()
        if result and result.avg_track:
            return {"track_temp": float(result.avg_track), "air_temp": float(result.avg_air)}

    # Ultimate fallback: Global average (typical mid-season conditions)
    return {"track_temp": 35.0, "air_temp": 25.0}


def get_weather_for_dataframe(race_ids: list) -> dict:
    """
    Batch fetch weather for multiple races.

    Args:
        race_ids: List of race IDs to fetch weather for

    Returns:
        dict mapping race_id -> weather dict
    """
    weather_data = {}
    for rid in race_ids:
        weather_data[rid] = get_race_weather(rid)
    return weather_data


if __name__ == "__main__":
    # Test the module
    print("Testing weather_source module...")

    # Test with a known race ID (adjust as needed)
    test_race_id = 1
    weather = get_race_weather(test_race_id)
    print(f"Race {test_race_id}: {weather}")
