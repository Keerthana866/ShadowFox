# ShadowFox  

This repository contains two machine learning projects developed as part of my internship tasks.  

## Level 1: Boston House Price Prediction  
- Predicts house prices in Boston using Linear Regression.  
- Dataset: `HousingData.csv`  
- Steps involved:  
  1. Data preprocessing (handled missing values)  
  2. Train-test split  
  3. Model training with Linear Regression  
  4. Model evaluation (MSE, R² Score)  
  5. Visualization of actual vs predicted values  

## Level 2: Loan Approval Prediction  
- Predicts loan approval status using Logistic Regression and Random Forest.  
- Dataset: Loan dataset with applicant details (financial history, income, etc.)  
- Steps involved:  
  1. Data preprocessing and encoding categorical features  
  2. Feature scaling  
  3. Training Logistic Regression and Random Forest models  
  4. Model evaluation (Accuracy, Precision, Recall, F1-score)  
  5. Feature importance analysis (Random Forest)  

## Technologies Used  
- Python  
- pandas  
- numpy  
- matplotlib  
- scikit-learn  

## How to Run  
```bash
pip install pandas numpy matplotlib scikit-learn
python task.py   # For Boston House Price Prediction
python loan_prediction.py   # For Loan Approval Prediction

