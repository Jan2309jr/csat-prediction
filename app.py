import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# Load trained model
model = joblib.load("catboost_best_model.pkl")

# Define UI
st.title("CSAT Score Predictor")

# Define input fields
st.header("Enter Customer Chat Details")

channel = st.selectbox("Channel Name", ['Inbound', 'Outcall'])  # update based on your data
category = st.selectbox("Category", ['Returns', 'Cancellation', 'Order Related'])  # update list
sub_category = st.selectbox("Sub-Category", ['Reverse Pickup Enquiry', 'Not Needed'])  # update
agent_shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Night'])
tenure = st.selectbox("Tenure Bucket", ['0-30', '31-60', '61-90', '>90', 'On Job Training'])
response_time = st.number_input("Response Time (minutes)", min_value=0.0, format="%.2f")

# Button to predict
if st.button("Predict CSAT Score"):
    input_df = pd.DataFrame({
        "Channel Name": [channel],
        "Category": [category],
        "Sub Category": [sub_category],
        "Agent Shift": [agent_shift],
        "Tenure Bucket": [tenure],
        "Response Time": [response_time]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted CSAT Score: **{int(prediction)}**")
