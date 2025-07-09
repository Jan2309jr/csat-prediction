import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Load model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# Load original dataset (for dropdown options)
df = pd.read_csv("https://raw.githubusercontent.com/Jan2309jr/csat-prediction/refs/heads/main/Customer_support_data.csv")  # Replace with your actual CSV file name

st.title("CSAT Score Prediction")

# Input fields
channel_name = st.selectbox("Channel Name", df["Channel Name"].unique())
category = st.selectbox("Category", df["Category"].unique())
supervisor = st.selectbox("Supervisor", df["Supervisor"].unique())
tenure_bucket = st.selectbox("Tenure Bucket", df["Tenure Bucket"].unique())
agent_shift = st.selectbox("Agent Shift", df["Agent Shift"].unique())
week_type = st.selectbox("Week Type", df["Week Type"].unique())
response_time = st.slider("Response Time (mins)", min_value=0, max_value=200, value=10)

# Prepare input dataframe
input_dict = {
    "Channel Name": [channel_name],
    "Category": [category],
    "Supervisor": [supervisor],
    "Tenure Bucket": [tenure_bucket],
    "Agent Shift": [agent_shift],
    "Week Type": [week_type],
    "Response Time": [response_time],
}

input_df = pd.DataFrame(input_dict)

# DEBUG: Show input to user
st.write("🔍 Model Input DataFrame", input_df)

# Make prediction
prediction = model.predict(input_df)[0]
st.success(f"✅ Predicted CSAT Score: **{int(prediction)}**")
