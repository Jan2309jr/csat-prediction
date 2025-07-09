import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Load model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# ✅ Corrected GitHub raw CSV URL
df = pd.read_csv("https://raw.githubusercontent.com/Jan2309jr/csat-prediction/main/Customer_support_data.csv")
df.columns = df.columns.str.strip()  # Strip whitespace from column names

st.title("CSAT Score Prediction")

# Input fields (clean + simple version)
channel_name = st.selectbox("Channel Name", df["Channel Name"].unique())
category = st.selectbox("Category", df["Category"].unique())
supervisor = st.selectbox("Supervisor", df["Supervisor"].unique())
tenure_bucket = st.selectbox("Tenure Bucket", df["Tenure Bucket"].unique())
agent_shift = st.selectbox("Agent Shift", df["Agent Shift"].unique())
week_type = st.selectbox("Week Type", df["Week Type"].unique())
response_time = st.slider("Response Time (mins)", min_value=0, max_value=200, value=10)

# Input DataFrame
input_df = pd.DataFrame({
    "Channel Name": [channel_name],
    "Category": [category],
    "Supervisor": [supervisor],
    "Tenure Bucket": [tenure_bucket],
    "Agent Shift": [agent_shift],
    "Week Type": [week_type],
    "Response Time": [response_time],
})

# Predict
prediction = model.predict(input_df)[0]
st.success(f"✅ Predicted CSAT Score: **{int(prediction)}**")
