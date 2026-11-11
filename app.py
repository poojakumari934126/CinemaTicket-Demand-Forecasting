from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cinema_demand_model.joblib"

st.set_page_config(page_title="Cinema Demand Forecast", page_icon="🎬", layout="centered")
st.title("🎬 Cinema Ticket Demand Forecasting")
st.write("Estimate expected ticket sales for a movie show using a machine-learning model.")
st.caption("Educational demo. The included training data is synthetic, not actual cinema data.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python src/train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    genre = st.selectbox("Movie genre", ["Action", "Comedy", "Drama", "Thriller", "Animation", "Romance"])
    showtime = st.selectbox("Show time", ["Morning", "Afternoon", "Evening", "Night"])
    day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    weekend = int(day in ["Saturday", "Sunday"])
    holiday = st.selectbox("Is it a public holiday?", ["No", "Yes"])
    capacity = st.selectbox("Screen capacity", [80, 100, 120, 150, 180, 200])
    popularity = st.slider("Movie popularity (1–10)", 1.0, 10.0, 7.5, 0.1)
    price = st.slider("Ticket price (₹)", 100, 350, 200, 10)
    marketing = st.number_input("Marketing spend (₹)", min_value=0, max_value=200000, value=25000, step=1000)
    competitor = st.selectbox("Major competing event nearby?", ["No", "Yes"])
    submitted = st.form_submit_button("Predict ticket demand")

if submitted:
    day_number = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day)
    sample = pd.DataFrame([{
        "genre": genre, "showtime": showtime, "day_of_week": day_number,
        "is_weekend": weekend, "is_holiday": int(holiday == "Yes"),
        "screen_capacity": capacity, "movie_popularity": popularity,
        "ticket_price": price, "marketing_spend": marketing,
        "competitor_event": int(competitor == "Yes")
    }])
    estimate = max(0, min(capacity, round(float(model.predict(sample)[0]))))
    st.metric("Estimated tickets sold", f"{estimate} / {capacity}")
    st.progress(estimate / capacity)
    st.info(f"Estimated occupancy: {estimate / capacity:.1%}")
    st.caption("This is a model estimate, not a guaranteed booking count.")
