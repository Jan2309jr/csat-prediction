import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

st.set_page_config(page_title="CSAT Score Prediction", layout="centered")

# Load CatBoost model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")  # Make sure this file exists in the same directory

# Load dataset for dropdown options
df = pd.read_csv("https://raw.githubusercontent.com/Jan2309jr/csat-prediction/refs/heads/main/Customer_support_data.csv")

# Clean column names
df.columns = df.columns.str.strip()

st.title("📊 CSAT Score Prediction App")

st.markdown("Please fill the inputs below to predict the **Customer Satisfaction (CSAT)** score.")

# Input fields
channel_name = st.selectbox("📢 Channel Name", df["Channel Name"].unique())
category = st.selectbox("📂 Category", df["Category"].unique())
supervisor = st.selectbox("🧑‍💼 Supervisor", df["Supervisor"].unique())
tenure_bucket = st.selectbox("📈 Tenure Bucket", df["Tenure Bucket"].unique())
agent_shift = st.selectbox("⏰ Agent Shift", df["Agent Shift"].unique())
week_type = st.selectbox("📅 Week Type", df["Week Type"].unique())
response_time = st.slider("⏳ Response Time (mins)", min_value=0, max_value=200, value=10)

# Create input dictionary
input_dict = {
    "Channel Name": [channel_name],
    "Category": [category],
    "Supervisor": [supervisor],
    "Tenure Bucket": [tenure_bucket],
    "Agent Shift": [agent_shift],
    "Week Type": [week_type],
    "Response Time": [response_time]
}

# Create DataFrame from inputs
input_df = pd.DataFrame(input_dict)

# Optional: Display user input for transparency
with st.expander("🔍 See Input Data"):
    st.dataframe(input_df)

# Set categorical feature indices (must match training)
cat_features = [0, 1, 2, 3, 4, 5]  # Columns: Channel Name to Week Type

# Predict
try:
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted CSAT Score: **{int(prediction)}**")
except Exception as e:
    st.error(f"❌ Prediction failed: {str(e)}")
