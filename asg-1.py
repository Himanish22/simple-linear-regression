import numpy as np
import matplotlib.pyplot as plt
x_vals = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_vals = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
N_vals = len(x_vals)
x_vals_mean = np.mean(x_vals)
y_vals_mean = np.mean(y_vals)
numerator_vals = np.sum((x_vals - x_vals_mean) * (y_vals - y_vals_mean))
denominator_vals = np.sum((x_vals - x_vals_mean) ** 2)
slope_analytic = numerator_vals / denominator_vals
intercept_analytic = y_vals_mean - slope_analytic * x_vals_mean
y_pred_vals = intercept_analytic + slope_analytic * x_vals
SSE_analytic = np.sum((y_vals - y_pred_vals) ** 2)
SST_vals = np.sum((y_vals - y_vals_mean) ** 2)
R2_analytic = 1 - SSE_analytic / SST_vals
print("Analytic Solution:")
print(f"Intercept (Beta 0): {intercept_analytic}")
print(f"Slope (Beta 1): {slope_analytic}")
print(f"SSE: {SSE_analytic}")
print(f"R^2: {R2_analytic}")
intercept_gd = 0
slope_gd = 0
learning_rate = 0.01
iterations = 1000
for epoch in range(iterations):
    y_pred_gd_vals = intercept_gd + slope_gd * x_vals
    error_vals = y_pred_gd_vals - y_vals
    intercept_gd -= learning_rate * (1/N_vals) * np.sum(error_vals)
    slope_gd -= learning_rate * (1/N_vals) * np.sum(error_vals * x_vals)
SSE_gd_vals = np.sum((y_vals - y_pred_gd_vals) ** 2)
R2_gd_vals = 1 - SSE_gd_vals / SST_vals
print("\nGradient Descent Solution:")
print(f"Intercept (Beta 0): {intercept_gd}")
print(f"Slope (Beta 1): {slope_gd}")
print(f"SSE: {SSE_gd_vals}")
print(f"R^2: {R2_gd_vals}")
plt.scatter(x_vals, y_vals, color='blue', label='Data points')
plt.plot(x_vals, y_pred_vals, color='red', label='Analytic Solution')
plt.plot(x_vals, y_pred_gd_vals, color='green', linestyle='--', label='Gradient Descent Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
