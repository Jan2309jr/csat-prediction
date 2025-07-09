import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Configure Streamlit
st.set_page_config(page_title="CSAT Predictor", layout="centered")

# Load trained model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# Load dataset for dropdown options
df = pd.read_csv("https://raw.githubusercontent.com/Jan2309jr/csat-prediction/main/Customer_support_data.csv")
df.columns = df.columns.str.strip()

st.title("📊 CSAT Score Prediction App")
st.markdown("Fill in the form to predict Customer Satisfaction (CSAT) score.")

# Inputs
channel_name = st.selectbox("📢 Channel Name", df["Channel Name"].unique())
category = st.selectbox("📂 Category", df["Category"].unique())
supervisor = st.selectbox("🧑‍💼 Supervisor", df["Supervisor"].unique())
agent_shift = st.selectbox("⏰ Agent Shift", df["Agent Shift"].unique())
response_time = st.slider("⏳ Response Time (mins)", 0, 200, 10)

# Create input DataFrame
input_df = pd.DataFrame({
    "Channel Name": [channel_name],
    "Category": [category],
    "Supervisor": [supervisor],
    "Agent Shift": [agent_shift],
    "Response Time": [response_time]
})

# Show input
with st.expander("🔍 Show Model Input"):
    st.dataframe(input_df)

# Predict
try:
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted CSAT Score: **{int(prediction)}**")
except Exception as e:
    st.error(f"❌ Prediction failed: {str(e)}")
