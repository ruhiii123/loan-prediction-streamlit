# --- auto_train.py ---
# This file automatically trains and saves the model if models/loan_pipeline.joblib doesn't exist

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Paths
DATA_PATH = "data/loan_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_pipeline.joblib")

# Function to train model
def train_model():
    print("🔄 Training new model...")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Define categorical and numerical columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    # Preprocessor
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    # Pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model
    pipe.fit(X_train, y_train)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    print("✅ Model trained and saved to", MODEL_PATH)

# Run automatically
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    else:
        print("✅ Model already exists. Skipping training.")

