import numpy as np # importing numpy for numerical operations
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Defining the given set data from Problem Statement A
x = np.array([2.5, 3.5, 5, 6, 7.5, 10, 12.5, 15, 17.5, 20]) # independent variables
y = np.array([5, 3.4, 2, 1.6, 1.2, 0.8, 0.6, 0.4, 0.3, 0.3]) # dependent variables

# Model Definitions
def straight_line(x, a, b):
    return a + b * x # y = a + bx 

def parabolic(x, a, b, c):
    return a + b * x + c * x**2 # y = a + bx + cx^2 

def power_model(x, a, b):
    return a * np.power(x, b) # y = ax^b 

def exponential_model(x, a, b):
    return a * np.exp(b * x) # y = ae^{bx} 

def logarithmic_model(x, a, b):
    return a + b * np.log(x) # y = a + b ln x 

models = {
    "Straight line": straight_line,
    "Parabolic": parabolic,
    "Power": power_model,
    "Exponential": exponential_model,
    "Logarithmic": logarithmic_model
}

#Preparation for Results Table
y_mean = np.mean(y)
sst = np.sum((y - y_mean)**2) # Total sum of squares for R^2 calculation 

print(f"{'Model':<15} | {'SSE (S_e)':<10} | {'R^2':<10}")
print("-" * 45)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Experimental Data', zorder=5) # Original data

#Performing Curve Fitting for each model
for name, func in models.items():
    # Clearly label each model section
    popt, _ = curve_fit(func, x, y, maxfev=10000)
    
    y_pred = func(x, *popt)
    residuals = y - y_pred # e_i = y_i - y_hat_i 
    sse = np.sum(residuals**2) # SSE = sum(y_i - y_hat_i)^2 
    r2 = 1 - (sse / sst) # Coefficient of determination formula 
    
    print(f"{name:<15} | {sse:<10.4f} | {r2:<10.4f}")
    
    # Plotting Instructions
    x_range = np.linspace(min(x), max(x), 100)
    plt.plot(x_range, func(x_range, *popt), label=f"{name} (R²={r2:.3f})")

# Formatting the plot
plt.xlabel("X (Independent Variable)")
plt.ylabel("y (Dependent Variable)")
plt.title("Curve Fitting using Method of Least Squares")
plt.legend()
plt.grid(True)
plt.show()