# =====================================
# Turbofan Engine RUL Prediction (No TF)
# =====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ------------------------------
# 1. LOAD DATA
# ------------------------------
print("Loading dataset...")

file_path = "cleaned_rul_dataset.csv"   # use your cleaned dataset

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# ------------------------------
# 2. FEATURE SELECTION
# ------------------------------
print("Preparing features...")

X = df.drop(["RUL"], axis=1)
y = df["RUL"]

# ------------------------------
# 3. NORMALIZATION
# ------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# 4. TRAIN-TEST SPLIT
# ------------------------------
split = int(0.8 * len(X_scaled))

X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# ------------------------------
# 5. MODEL (Random Forest)
# ------------------------------
print("Training model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------
# 6. PREDICTION
# ------------------------------
print("Predicting...")

y_pred = model.predict(X_test)

# ------------------------------
# 7. EVALUATION
# ------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# ------------------------------
# 8. VISUALIZATION
# ------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual RUL")
plt.plot(y_pred[:100], label="Predicted RUL")
plt.legend()
plt.title("RUL Prediction (Actual vs Predicted)")
plt.xlabel("Samples")
plt.ylabel("RUL")
plt.show()