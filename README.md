# ShadowFox  

This repository contains internship tasks completed as part of the **ShadowFox Internship Program**. It includes two machine learning projects:  

1. **Boston House Price Prediction (Level 1 - Beginner)**  
2. **Loan Approval Prediction (Level 2 - Intermediate)**  

---

## 1. Boston House Price Prediction (Level 1 - Beginner)  

### Project Overview  
- Predicts house prices in Boston using machine learning.  
- Dataset: `HousingData.csv` (includes features like number of rooms, crime rate, etc.)  
- Model: Linear Regression  

### Steps Involved  
1. Data Preprocessing (handled missing values)  
2. Splitting dataset into training and testing sets  
3. Training Linear Regression model  
4. Evaluating performance (MSE, R² Score)  
5. Visualizing actual vs predicted prices  

### Results  
- The model predicts Boston house prices fairly accurately.  
- Evaluation metrics are printed in the console.  
- A scatter plot compares actual vs predicted values.  

### How to Run  
```bash
pip install pandas numpy matplotlib scikit-learn
python task.py
