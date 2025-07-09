import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# Load saved components
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")
model_columns = joblib.load("model_columns.pkl")
cat_features = joblib.load("cat_features.pkl")

st.title("🎯 CSAT Score Prediction")

# User inputs
input_data = {
    'Channel Name': st.selectbox("Channel Name", ["Phone", "Email", "Chat"]),
    'Category': st.selectbox("Category", ["Billing", "Technical", "General"]),
    'Sub Category': st.text_input("Sub Category"),
    'Agent Name': st.text_input("Agent Name"),
    'Supervisor': st.text_input("Supervisor"),
    'Manager': st.text_input("Manager"),
    'Tenure Bucket': st.selectbox("Tenure Bucket", ["<1 yr", "1-2 yrs", "2-5 yrs", "5+ yrs"]),
    'Agent Shift': st.selectbox("Agent Shift", ["Morning", "Evening", "Night"]),
    'Day': st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    'Week Type': st.selectbox("Week Type", ["Weekday", "Weekend"])
}

# Predict button
if st.button("Predict CSAT Score"):
    # Create input DataFrame
    input_df = pd.DataFrame([input_data])

    # Add missing numeric columns with default 0
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[model_columns]

    # Predict with cat_features using column names
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted CSAT Score: **{int(prediction)}**")
