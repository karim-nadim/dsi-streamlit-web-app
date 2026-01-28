import streamlit as st
import pandas as pd
import joblib

# Load the pipeline (from data preprocessing until model precition)
model = joblib.load("pipeline.joblib")

# add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood of purchase")

# Age input 
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value = 35)

# Gender input
gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ["Male", "Female"])

# Credit Score input 
credit_score = st.number_input(
    label = "03. Enter the customer's age",
    min_value = 0,
    max_value = 1000,
    value = 500)

# Submit inputs to model
if st.button("Submit for Prediction"):
    
    new_data = pd.DataFrame({"age": [age],
                             "gender": [gender],
                             "credit_score": [credit_score]})
    
    pred_proba = model.predict_proba(new_data)[0][1] # get the prediction probability
    
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")
    












