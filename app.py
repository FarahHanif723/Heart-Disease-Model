import streamlit as st
import pandas as pd
import joblib


model = joblib.load("Logistic_heart_final_1.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("Heart Disease Prediction App")
st.write("Enter your health details below to check the risk of heart disease.")


user_data = {}

for col in columns:
    # Select type of input based on column name
    if col in ['Sex', 'ExAng', 'ChestPain_nonanginal', 'ChestPain_nontypical', 
               'Thal_normal', 'Thal_reversable']:
        user_data[col] = st.selectbox(f"{col} (0 = No, 1 = Yes)", [0, 1])
    else:
        user_data[col] = st.number_input(f"Enter {col}", min_value=0.0, step=0.1)

# Convert user inputs to DataFrame
input_df = pd.DataFrame([user_data])

# Ensure same column order as during training
input_df = input_df[columns]

# SCALE AND PREDICT

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]


# DISPLAY RESULT

st.markdown("---")
st.subheader(" Prediction Result")

if prediction == 1:
    st.error(f" **High Risk of Heart Disease Detected!**\n\nProbability: **{probability:.2f}**")
else:
    st.success(f" **No Heart Disease Detected.**\n\nProbability: **{probability:.2f}**")

st.markdown("---")
st.caption("This prediction is based on your entered data and a trained Logistic Regression model.")


