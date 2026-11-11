# Cinema Ticket Demand Forecasting

## 1. Project overview
This project predicts the expected number of tickets sold for a cinema show. It uses show information, movie popularity, ticket price, marketing spend, weekends, holidays, screen capacity, and nearby competing events.

**Important:** The included dataset is synthetically generated for learning and demonstration. It contains no real customer, patient, or student information. Model results on synthetic data do not prove real-world business performance.

## 2. Problem statement
Cinema operators need to estimate ticket demand to plan screen allocation, staffing, promotions, and show schedules. This project demonstrates a supervised regression approach to estimate ticket sales.

## 3. Technology stack
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regression
- Streamlit
- Joblib

## 4. Project structure
```text
Cinema_Ticket_Demand_Forecasting/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── cinema_ticket_demand.csv
├── models/
│   ├── cinema_demand_model.joblib
│   └── metrics.json
├── evidence/
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
└── src/
    ├── train_model.py
    └── predict.py
```

## 5. Setup and run (Windows CMD)
```bash
cd Cinema_Ticket_Demand_Forecasting
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/train_model.py
python src/predict.py
streamlit run app.py
```
If `py` is unavailable, use `python` to create the virtual environment.

## 6. Features used
- Movie genre and showtime
- Day of week and weekend indicator
- Public holiday indicator
- Screen capacity
- Movie popularity score
- Ticket price
- Marketing spend
- Nearby competing event indicator

## 7. Methodology
1. Load and inspect the CSV dataset.
2. Separate input features and target (`tickets_sold`).
3. One-hot encode categorical columns.
4. Split data into training and testing sets (80:20).
5. Train a Random Forest Regressor.
6. Evaluate using MAE, RMSE, and R².
7. Save the trained pipeline and metrics.
8. Use Streamlit to accept show details and display a demand estimate.

## 8. Evaluation
Run `python src/train_model.py`. The script prints test-set MAE, RMSE, and R² and saves them in `models/metrics.json`. These are results from a synthetic demonstration dataset only.

## 9. Limitations and future scope
- Synthetic data cannot capture all real cinema booking patterns.
- The current model does not explicitly model booking lead time or seat-level inventory.
- Real deployment would require permissioned historical booking data, data-quality checks, periodic retraining, and monitoring for drift.
- Future work: time-series forecasting, weather and local-event features, explainability with SHAP, and integration with a cinema booking platform.

## 10. AI tool disclosure
AI assistance was used to help draft and organize parts of the project code and documentation. The implementation should be reviewed, run, and understood by the submitting student before submission. Disclose AI assistance according to the institution's rules.

## 11. Suggested viva explanation
"I built a cinema ticket demand forecasting system using a Random Forest regression model. The model takes movie genre, showtime, day type, holiday status, screen capacity, popularity, ticket price, marketing spend, and competitor-event information as inputs. It predicts the expected number of tickets sold. I used a synthetic dataset so that no real customer data is exposed. I evaluated the model using MAE, RMSE, and R-squared, and deployed a simple Streamlit interface for interactive predictions."
