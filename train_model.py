"""Train a cinema ticket demand forecasting model on the included synthetic dataset."""
from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cinema_ticket_demand.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    target = "tickets_sold"
    features = [
        "genre", "showtime", "day_of_week", "is_weekend", "is_holiday",
        "screen_capacity", "movie_popularity", "ticket_price",
        "marketing_spend", "competitor_event"
    ]
    X, y = df[features], df[target]
    categorical = ["genre", "showtime"]
    numeric = [c for c in features if c not in categorical]
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric)
    ])
    model = Pipeline([
        ("preprocessor", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=250, max_depth=14, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "model": "RandomForestRegressor",
        "test_rows": int(len(y_test)),
        "MAE": round(float(mean_absolute_error(y_test, predictions)), 3),
        "RMSE": round(float(mean_squared_error(y_test, predictions) ** 0.5), 3),
        "R2": round(float(r2_score(y_test, predictions)), 4)
    }
    joblib.dump(model, MODEL_DIR / "cinema_demand_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
