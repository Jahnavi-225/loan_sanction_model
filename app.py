

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_model.pkl")

st.set_page_config(page_title="Loan Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Fill details to check loan eligibility")

# Sidebar input
st.sidebar.header("Applicant Details")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Married = st.sidebar.selectbox("Married", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.sidebar.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.sidebar.selectbox("Loan Term", [360, 120, 180, 240, 60])
Credit_History = st.sidebar.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert to dataframe
input_data = pd.DataFrame({
    'Gender': [Gender],
    'Married': [Married],
    'Dependents': [Dependents],
    'Education': [Education],
    'Self_Employed': [Self_Employed],
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Property_Area': [Property_Area]
})

# Show input
st.write("### 📋 Input Data")
st.dataframe(input_data)

# Predict
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved (Confidence: {probability[0][1]*100:.2f}%)")
    else:
        st.error(f"❌ Loan Not Approved (Confidence: {probability[0][0]*100:.2f}%)")