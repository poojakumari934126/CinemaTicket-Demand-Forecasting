"""Command-line prediction example."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models" / "cinema_demand_model.joblib")

sample = pd.DataFrame([{
    "genre": "Action",
    "showtime": "Evening",
    "day_of_week": 5,
    "is_weekend": 1,
    "is_holiday": 0,
    "screen_capacity": 150,
    "movie_popularity": 8.2,
    "ticket_price": 200,
    "marketing_spend": 25000,
    "competitor_event": 0
}])
prediction = model.predict(sample)[0]
print(f"Predicted ticket demand: {max(0, round(prediction))} tickets")
