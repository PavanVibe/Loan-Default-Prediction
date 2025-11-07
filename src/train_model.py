# --- 1. Configuration & Setup ---
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import re # <-- IMPORT THE REGEX LIBRARY FOR CLEANING

# Get the absolute path of the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the correct, absolute paths to the data and model folders
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'loan_data.csv')
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.joblib')
MODEL_COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.joblib')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# --- 2. Load Data ---
print(f"Loading data from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

print("Data loaded successfully.")

# --- 3. Preprocessing & Feature Engineering ---
print("Preprocessing data...")

# Target: 'Status' (the column we want to predict)
target = 'Status'

# Features: The columns we'll use as clues
features = [
    'loan_amount',
    'income',
    'Credit_Score',
    'dtir1',
    'age',
    'Gender',
    'Region'
]

# Categorical features that need to be one-hot encoded
categorical_cols = ['age', 'Gender', 'Region']

# Ensure all specified columns exist
all_cols = features + [target]
for col in all_cols:
    if col not in df.columns:
        print(f"Error: Column '{col}' not found in the dataset. Please check spelling.")
        exit()

# Drop rows where the target column is missing
df.dropna(subset=[target], inplace=True)

# Separate features (X) and target (y)
X = df[features]
y = df[target]

# One-Hot Encode categorical features
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# === NEW CLEANING STEP TO FIX THE ERROR ===
# Clean column names: replace any bad characters with '_'
# This fixes the "ValueError: feature_names..."
print("Cleaning column names for XGBoost...")
X.columns = [re.sub(r'[\[\]<]', '_', col) for col in X.columns]
# =================================================

# Store the column layout after encoding (VERY important for prediction)
model_columns = X.columns
joblib.dump(model_columns, MODEL_COLUMNS_PATH)
print(f"Model columns saved to {MODEL_COLUMNS_PATH}")

# Fill missing numerical values with the median
X.fillna(X.median(), inplace=True)


# --- 4. Split Data ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- 5. Train XGBoost Model ---
print("Training XGBoost model... (This may take a few moments on a large dataset)")
xgb_classifier = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
xgb_classifier.fit(X_train, y_train)


# --- 6. Evaluate Model ---
print("\n--- Model Evaluation ---")
y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]
y_pred_class = xgb_classifier.predict(X_test)

# Calculate AUC score
auc = roc_auc_score(y_test, y_pred_proba)
print(f"✅ REAL AUC Score: {auc:.4f}")

# Display classification report
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred_class))


# --- 7. Save the Model ---
print("\nSaving the trained model...")
joblib.dump(xgb_classifier, MODEL_PATH)
print(f"✅ Model saved successfully to {MODEL_PATH}")