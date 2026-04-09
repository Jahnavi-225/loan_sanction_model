# Loan Approval Prediction Web App

This project is a machine learning–based web application that predicts whether a loan application will be approved based on applicant details. It combines a trained model with a simple web interface built using Streamlit.

---

## Project Overview

The goal of this project is to automate the loan approval process. Instead of manually evaluating each application, the model analyzes key features such as income, credit history, and personal details to predict eligibility.

---

## Features

* Predicts loan approval in real time
* Uses a Random Forest machine learning model
* Interactive web interface built with Streamlit
* Handles missing values and data preprocessing
* Displays prediction confidence

---

## Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## Project Structure

```
project/
│── train_model.py        # Machine learning training script
│── app.py                # Streamlit application
│── loan_model.pkl        # Saved trained model
│── dataset.xlsx          # Dataset file
│── README.md             # Project documentation
```

---

## Installation and Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/loan-prediction-app.git
cd loan-prediction-app
```

### 2. Install dependencies

```
pip install pandas numpy scikit-learn streamlit joblib openpyxl
```

---

## Running the Project

### Step 1: Train the model

```
python train_model.py
```

This will train the model and generate the file `loan_model.pkl`.

---

### Step 2: Run the web application

```
python -m streamlit run app.py
```

Open the browser and go to:

```
http://localhost:8501
```

---

## Input Features

The model takes the following inputs:

* Gender
* Marital Status
* Dependents
* Education
* Self Employment status
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Term
* Credit History
* Property Area

---

## Output

The application provides:

* Loan approval status (Approved / Not Approved)
* Confidence score of the prediction

---

## Challenges Addressed

* Handled missing data using appropriate imputation techniques
* Resolved mixed data type issues in categorical columns
* Built a preprocessing pipeline to ensure consistency and avoid data leakage

---

## Future Improvements

* Deploy the application online
* Add data visualization and dashboards
* Experiment with additional models and tuning
* Improve user interface design

---

## Author

Jahnavi
---

## Notes

This project is intended for educational purposes and demonstrates how machine learning can be applied to real-world decision-making problems.
