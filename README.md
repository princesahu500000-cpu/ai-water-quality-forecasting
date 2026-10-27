# AI Water Quality Forecasting System (CSE276 CA1, Project P18)

Team ZJ18: Prince Sahu (12508888), Ridham Dadwal (12514189)

Water Quality Index (WQI) forecasting and decision-support app.
- Part A (real data): next-year WQI for Indian river stations, using the Kaggle dataset
  "Indian water quality data" (Anbarivan.N.L):
  https://www.kaggle.com/datasets/anbarivan/indian-water-quality-data
  Download water_dataX.csv and place it in data/.
- Part B (synthetic data): 7-day ahead WQI forecast on a generated daily dataset.

## Setup
    py -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Run order
Synthetic part:

    python src\generate_data.py
    python src\preprocess.py
    python src\wqi.py
    python src\forecast.py

Real part (needs data\water_dataX.csv):

    python src\real_data_forecast.py

Demo app:

    streamlit run src\app.py

Note: the real WQI is on a 0-100 scale (higher = better); the synthetic WQI is a weighted
index (higher = worse).