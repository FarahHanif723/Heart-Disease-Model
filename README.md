# Heart Disease Prediction App

A machine learning web app that predicts the risk of heart disease based on user health inputs, built using **Logistic Regression** and deployed with **Streamlit**.

> This project was built as part of my internship at **Alpha**.

##  About the Project

This app takes health-related inputs from the user (such as age, cholesterol level, chest pain type, etc.) and predicts whether the person is at risk of heart disease, along with the probability score.

##  Tech Stack

- Python
- Streamlit
- Scikit-learn (Logistic Regression)
- Pandas
- Joblib

## Live App

[Click here to try the app] (https://heart-disease-model-5msgcv9jkvcqjgk6cdgatk.streamlit.app/)

## How It Works

1. User enters health parameters via the Streamlit interface
2. Inputs are scaled using a pre-trained `StandardScaler`
3. The trained Logistic Regression model predicts the risk
4. Result is displayed with risk level and probability

##  Files

- `app.py` — Main Streamlit application
- `Logistic_heart_final_1.pkl` — Trained Logistic Regression model
- `scaler.pkl` — Saved StandardScaler for input scaling
- `columns.pkl` — Column order used during training
- `requirements.txt` — Project dependencies

##  Author

**Farah Bibi**
Built during internship at **Alpha**

## Note

This app is for educational/demonstration purposes and should not be used as a substitute for professional medical advice.
##  How to Run

1. Install dependencies
   pip install -r requirements.txt

2. Run the app
   streamlit run app.py
