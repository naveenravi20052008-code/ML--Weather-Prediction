# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
[weather-station-eee-block_2024_07_13.csv](https://github.com/user-attachments/files/26130922/weather-station-eee-block_2024_07_13.csv)



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and preprocess the dataset by handling missing values and creating relevant features.

2.Split the data into training and testing sets.

3.Train the Random Forest Regressor using training data.

4.Predict outputs and evaluate performance using metrics and graphs.
 

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: 
RegisterNumber:  
*/# Random Forest with Graph and Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("/content/weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# Fill missing values
df.bfill(inplace=True)

# Feature engineering
df["hour"] = df["time"].dt.hour
df["temp_lag"] = df["tem"].shift(1)

# Drop null values
df = df.dropna()

# Input and output
X = df[["hum", "pressure", "hour", "temp_lag"]]
y = df["tem"]

# Train-test split
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = RandomForestRegressor(n_estimators=80, random_state=1)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation values
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Evaluation ---")
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)
print("MAE :", mae)

# Graph: Actual vs Predicted
plt.figure()
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred[:100], linestyle="--", label="Predicted")
plt.title("Temperature Prediction (Random Forest)")
plt.xlabel("Samples")
plt.ylabel("Temperature")
plt.legend()
plt.grid()

plt.show()

# Future prediction using last row
last_input = X.iloc[-1:]
future_pred = model.predict(last_input)

print("\nNext Predicted Temperature:", future_pred[0])
```

## Output:
<img width="757" height="679" alt="Screenshot 2026-03-20 070305" src="https://github.com/user-attachments/assets/23075a7f-27ef-4822-9586-e66a21add4bb" />


## Result:
The Random Forest model successfully predicts environmental values with good accuracy as indicated by R² score and error metrics.
