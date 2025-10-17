import pandas as pd
import joblib
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Configuration & Setup ---
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.joblib')
MODEL_COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.joblib')


# --- 2. Load Model and Preprocessing Info ---
print("Loading model and preprocessing columns...")
try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(MODEL_COLUMNS_PATH)
except FileNotFoundError:
    print("Error: Model or model columns file not found.")
    print("Please run train_model.py first to train and save the model.")
    exit()


# --- 3. Create Sample New Data ---
# This dictionary represents new loan applications to be scored.
# In a real application, this data would come from an API, a form, or a database.
new_data = {
    'loan_amount': [15000, 22000, 7000],
    'grade': ['C', 'A', 'D'],
    'annual_income': [62000, 150000, 38000],
    'home_ownership': ['RENT', 'MORTGAGE', 'RENT']
}
new_df = pd.DataFrame(new_data)
print("\n--- New Loan Applications to Score ---")
print(new_df)


# --- 4. Preprocess New Data ---
print("\nPreprocessing new data...")
# Apply the same one-hot encoding as in training
new_df_encoded = pd.get_dummies(new_df, columns=['grade', 'home_ownership'], drop_first=True)

# Align columns with the training data
# This step is crucial to ensure the new data has the exact same structure
# as the data the model was trained on.
final_df = new_df_encoded.reindex(columns=model_columns, fill_value=0)


# --- 5. Make Predictions ---
print("\nMaking predictions...")
# Predict probabilities (the second column is the probability of default)
predictions_proba = model.predict_proba(final_df)[:, 1]

# Assign predictions back to the original new data
new_df['default_probability'] = predictions_proba
new_df['predicted_loan_status'] = (new_df['default_probability'] > 0.5).astype(int) # Threshold at 50%

print("\n--- Prediction Results ---")
print(new_df)