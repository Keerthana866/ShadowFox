# Boston House Price Prediction - Improved Version

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("HousingData.csv")

# 2. Features (X) and Target (y)
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# 3. Handle missing values (replace NaN with column mean)
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# 4. Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict on Test Data
y_pred = model.predict(X_test)

# 7. Evaluate Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Trained Successfully!")
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R^2 Score:", round(r2, 3))

# 8. Plot Actual vs Predicted
plt.figure(figsize=(6, 6))

# Blue dots → model predictions
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, edgecolors="k", label="Model Predictions")

# Red line → perfect prediction line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", label="Perfect Prediction")

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()  # <-- this shows labels in the graph
plt.show()

# 9. Feature Importance (coefficients)
feature_names = df.drop("MEDV", axis=1).columns
coefficients = model.coef_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance (by coefficients):")
print(importance_df)