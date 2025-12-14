import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

st.title("📊 Expresso Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict Churn"):
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")
