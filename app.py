import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_model.pkl')

st.title("📊 Telco Customer Churn Prediction")
st.caption("Model terbaik: Logistic Regression + Preprocessing")

# (isi code streamlit kamu di sini, contoh yang tadi)
