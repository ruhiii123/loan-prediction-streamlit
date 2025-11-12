import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --- Paths ---
DATA_PATH = "data/loan_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_pipeline.joblib")

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# --- Feature categories ---
categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
numerical_cols = [c for c in X.columns if X[c].dtype != "object"]

# --- Preprocessing pipeline ---
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

# --- Full model pipeline ---
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Grid search for better params ---
param_grid = {"model__n_estimators": [50, 100], "model__max_depth": [5, 10, None]}
grid = GridSearchCV(pipeline, param_grid, cv=3)
grid.fit(X_train, y_train)

print("✅ Best Parameters:", grid.best_params_)
print("✅ Train Accuracy:", grid.score(X_train, y_train))
print("✅ Test Accuracy:", grid.score(X_test, y_test))

# --- Save model ---
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(grid.best_estimator_, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
