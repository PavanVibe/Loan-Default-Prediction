
import os
import pandas as pd
import joblib
import re 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.joblib')
MODEL_COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.joblib')


print("Loading model and preprocessing columns...")
try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(MODEL_COLUMNS_PATH)
except FileNotFoundError:
    print("Error: Model or model columns file not found.")
    print("Please run train_model.py first to train and save the model.")
    exit()


new_data = {
    'loan_amount': [120000, 50000],
    'income': [35000, 75000],
    'Credit_Score': [620, 780],
    'dtir1': [45.0, 22.0],
    'age': ['45-54', '25-34'],
    'Gender': ['Male', 'Female'],
    'Region': ['south', 'north']
}
new_df = pd.DataFrame(new_data)
print("\n--- New Loan Applications to Score ---")
print(new_df)


print("\nPreprocessing new data...")
new_df_encoded = pd.get_dummies(new_df, columns=['age', 'Gender', 'Region'], drop_first=True)


print("Cleaning column names for XGBoost...")
new_df_encoded.columns = [re.sub(r'[\[\]<]', '_', col) for col in new_df_encoded.columns]

final_df = new_df_encoded.reindex(columns=model_columns, fill_value=0)


print("\nMaking predictions...")
predictions_proba = model.predict_proba(final_df)[:, 1]

new_df['prediction_probability'] = predictions_proba
new_df['predicted_status'] = (new_df['prediction_probability'] > 0.5).astype(int)

print("\n--- Prediction Results ---")
print(new_df)
