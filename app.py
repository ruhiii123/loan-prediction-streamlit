import streamlit as st
st.set_page_config(page_title="💸 Loan Eligibility Predictor", page_icon="💰", layout="wide")

import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/loan_pipeline.joblib"

# --- Inline training if model missing ---
def train_inline_model():
    st.info("⚙️ Model not found. Training new model automatically...")
    try:
        df = pd.read_csv("data/loan_data.csv")
        X = df.drop("Loan_Status", axis=1)
        y = df["Loan_Status"]

        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if X[c].dtype != "object"]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        st.success("✅ Model trained and saved!")
    except Exception as e:
        st.error(f"❌ Error training model: {e}")

# --- Train if model missing ---
if not os.path.exists(MODEL_PATH):
    train_inline_model()

# --- Always reload model after possible training ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

model = load_model()

# --- App Interface ---
st.title("💸 Loan Eligibility Predictor")
st.markdown("Enter applicant details (left) or upload a CSV for batch predictions. Click **Predict** or **Run batch prediction**.")

if model is None:
    st.error("⚠️ Model not found. Please ensure `loan_data.csv` is uploaded to the `data` folder.")
else:
    option = st.sidebar.radio("Select Input Mode", ["Manual form (single)", "Upload CSV (batch)"])

    if option == "Manual form (single)":
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", 0)
        coapplicant_income = st.number_input("Coapplicant Income", 0)
        loan_amount = st.number_input("Loan Amount", 0)
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 60])
        credit_history = st.selectbox("Credit History (1=Yes, 0=No)", [1, 0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        input_df = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }])

        st.write("### Applicant Data")
        st.dataframe(input_df)

        if st.button("Predict"):
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            st.success(f"Prediction: {'Approved ✅' if pred == 'Y' else 'Rejected ❌'}")
            st.info(f"Approval Probability: {prob:.2f}")

    else:
        file = st.file_uploader("Upload CSV file", type=["csv"])
        if file is not None:
            df = pd.read_csv(file)
            preds = model.predict(df)
            df["Prediction"] = np.where(preds == "Y", "Approved", "Rejected")
            st.dataframe(df)
            st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")

        
  
