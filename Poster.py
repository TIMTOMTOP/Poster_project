import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

df=pd.read_csv('price_consumption_file.csv')

times = pd.to_datetime(df['Datetime'])
prices=df["SE3 (EUR)"]
consumption=df["Consumption(MW)"]
predicted_consumption=df["Prognosis (MW)"]


# Normalize prices and consumption if needed
prices_normalized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
consumption_normalized = (consumption - np.min(consumption)) / (np.max(consumption) - np.min(consumption))
prediction_normalized = (predicted_consumption - np.min(predicted_consumption)) / (np.max(predicted_consumption) - np.min(predicted_consumption))


# Define the logistic scaling function l
def l(x, k=1):
    return 1 / (1 + np.exp(-k * x))

# Define the state of charge response function f
# Assuming a simple linear relationship for illustration
def f(X, alpha1=1, alpha2=0):
    return alpha1 * X + alpha2

# Define the price response function g using a hypothetical I-spline
# For actual implementation, you'd need to fit this to your data
def g(u, beta=np.array([0.5, -0.5, 0.5]), knots=np.array([0.25, 0.5, 0.75])):
    # Example of implementing a simple piecewise linear function
    # that approximates the behavior of I-splines
    n_knots = len(knots)
    g_value = 0
    for i in range(n_knots):
        if u < knots[i]:
            g_value += beta[i] * u/knots[i]
        else:
            g_value += beta[i]
    return g_value


# Parameters
C = 10  # Capacitance of the system (controls speed of charge/discharge)
sigma_X = 0.05  # Noise intensity in the state of charge dynamics
Delta = 0.2  # Flexibility ratio

# Initial conditions with first data point initialization
X = np.zeros(len(times))
X[0] = prediction_normalized.iloc[0]  # Initialize with the first value of prediction_normalized
D = np.zeros_like(X)
D[0] = prediction_normalized.iloc[0]  # Similarly for D

# Simulation
for i in range(1, len(times)):
    dW = np.sqrt(1) * np.random.normal()
    delta = l(f(X[i - 1]) + g(prices_normalized.iloc[i - 1]))
    X[i] = X[i - 1] + (1 / C) * (delta * prediction_normalized.iloc[i - 1] - prediction_normalized.iloc[i - 1]) + sigma_X * X[i - 1] * (1 - X[i - 1]) * dW
    D[i] = prediction_normalized.iloc[i] + Delta * (delta - 1) * prediction_normalized.iloc[i]

# Add observation noise
sigma_Y = 0.05
Y = D + np.random.normal(0, sigma_Y, size=D.shape)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)


# Plotting
fig, ax1 = plt.subplots(figsize=(15, 8))
ax1.plot(times, Y, label='Predicted Demand', color='blue')
ax1.plot(times, consumption_normalized, label='Actual Demand', color='red')
ax1.plot(times, prediction_normalized, label='Baseline', linestyle='--', color='gray')
ax1.set_ylabel('Normalized Demand')
ax1.set_title('Simulated and Actual Demand Over Time')
ax1.legend(loc='upper left')

# Create bar chart for prices on the same axis
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
ax2.bar(times, prices_normalized, width=0.01, alpha=0.3, color='red', label='Normalized Price')
ax2.set_ylabel('Normalized Price')
ax2.legend(loc='upper right')

plt.show()






df['prices_normalized'] = (df["SE3 (EUR)"] - df["SE3 (EUR)"].min()) / (df["SE3 (EUR)"].max() - df["SE3 (EUR)"].min())
df['consumption_normalized'] = (df["Consumption(MW)"] - df["Consumption(MW)"].min()) / (df["Consumption(MW)"].max() - df["Consumption(MW)"].min())
df['prediction_normalized'] = (df["Prognosis (MW)"] - df["Prognosis (MW)"].min()) / (df["Prognosis (MW)"].max() - df["Prognosis (MW)"].min())

# Assuming Y contains your predicted normalized values already and has the same index as the dataframe
df['Predicted_Normalized'] = Y  # Your predicted demand normalized data

# Calculate errors
df['Error_Predicted'] = np.abs(df['Predicted_Normalized'] - df['consumption_normalized'])
df['Error_Prognosis'] = np.abs(df['prediction_normalized'] - df['consumption_normalized'])

# Mean Absolute Error (MAE)
mae_predicted = df['Error_Predicted'].mean()
mae_prognosis = df['Error_Prognosis'].mean()

# Root Mean Squared Error (RMSE)
rmse_predicted = np.sqrt(np.mean(df['Error_Predicted']**2))
rmse_prognosis = np.sqrt(np.mean(df['Error_Prognosis']**2))

# Display the results
print(f"MAE Predicted: {mae_predicted}")
print(f"MAE Prognosis: {mae_prognosis}")
print(f"RMSE Predicted: {rmse_predicted}")
print(f"RMSE Prognosis: {rmse_prognosis}")

# Compare improvements
improvement_mae = 100 * (mae_prognosis - mae_predicted) / mae_prognosis
improvement_rmse = 100 * (rmse_prognosis - rmse_predicted) / rmse_prognosis

print(f"Improvement in MAE: {improvement_mae}%")
print(f"Improvement in RMSE: {improvement_rmse}%")