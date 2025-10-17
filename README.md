# Loan-Default-Prediction
machine learning project to predict loan default risk using Python, scikit-learn, and XGBoost.
# Loan Default Prediction & Lending Workflow Optimization

## 🎯 Project Overview
This project focuses on predicting loan default risk using machine learning. By analyzing historical loan data, we built an XGBoost model that achieves **89% AUC** on the test set. The goal is to provide a data-driven tool to help lenders make more accurate loan approval decisions and minimize financial losses.

## 🛠️ Tools & Technologies
- **Data Analysis**: Python, Pandas, SQL
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn

## 📈 Key Results
- **Model Performance**: The final XGBoost model achieved an **Area Under the Curve (AUC) of 0.89**.
- **Key Risk Drivers**: Analysis revealed that `loan grade`, `debt-to-income ratio`, and `annual income` are the most significant predictors of default.
- **Business Impact**: Proposed recommendations are projected to **improve loan approval accuracy by 15%**.

## 🚀 How to Run this Project
1.  Clone the repository:
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/Loan-Default-Prediction.git
    cd Loan-Default-Prediction
    ```
2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  Place your dataset `loan_data.csv` in the `data/` folder.

4.  Run the training script to train and save the model:
    ```bash
    python src/train_model.py
    ```