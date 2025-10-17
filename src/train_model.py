import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
# --- 1. Configuration & Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define file paths
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'loan_data.csv')
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.joblib')
MODEL_COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.joblib')
# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# --- 2. Load Data ---
print(f"Loading data from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

# --- 3. Preprocessing & Feature Engineering ---
print("Preprocessing data...")
# Define target and features
target = 'loan_status'
features = ['loan_amount', 'grade', 'annual_income', 'home_ownership']

# Ensure target variable exists and drop rows with missing target
if target not in df.columns:
    print(f"Error: Target column '{target}' not in the dataset.")
    exit()
df.dropna(subset=[target], inplace=True)

# Separate features (X) and target (y)
X = df[features]
y = df[target]

# One-Hot Encode categorical features
# This is more robust than LabelEncoding for nominal categories
X = pd.get_dummies(X, columns=['grade', 'home_ownership'], drop_first=True)

# Store the column layout after encoding
model_columns = X.columns
joblib.dump(model_columns, MODEL_COLUMNS_PATH)
print(f"Model columns saved to {MODEL_COLUMNS_PATH}")

# Handle missing numerical values
X.fillna(X.median(), inplace=True)


# --- 4. Split Data ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- 5. Train XGBoost Model ---
print("Training XGBoost model...")
# Initialize the XGBoost classifier with common parameters
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train the model
xgb_classifier.fit(X_train, y_train)


# --- 6. Evaluate Model ---
print("\n--- Model Evaluation ---")
# Make predictions on the test set
y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]
y_pred_class = xgb_classifier.predict(X_test)

# Calculate AUC score
auc = roc_auc_score(y_test, y_pred_proba)
print(f"✅ AUC Score: {auc:.4f}")

# Display classification report
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred_class))


# --- 7. Save the Model ---
print("\nSaving the trained model...")
joblib.dump(xgb_classifier, MODEL_PATH)
print(f"✅ Model saved successfully to {MODEL_PATH}")