import numpy as np # importing numpy for numerical operations
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline

# Define the given set data
x = np.array([1990, 1995, 2000, 2005, 2010, 2015]) # independent variables (Years)
y = np.array([2450800, 2710500, 2890200, 3150700, 3420300, 3810600]) # dependent variables (Population)

# Newton Interpolation
def newton_interpolation(x_pts, y_pts, target_x):
    n = len(x_pts)
    coef = np.zeros([n, n]) # Creates table for divided differences
    coef[:,0] = y_pts
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x_pts[i+j] - x_pts[i])
    
    # Evaluating the polynomial at target_x
    result = coef[0,0]
    for i in range(1, n):
        term = coef[0,i]
        for j in range(i):
            term *= (target_x - x_pts[j])
        result += term
    return result

# Lagrange Interpolation
poly_lagrange = lagrange(x, y) # Computes Lagrange polynomial

# Cubic Spline Interpolation
cs_spline = CubicSpline(x, y) # Computes Cubic Spline function

# Verification and Predictions
years = [2018, 2025]
for yr in years:
    pop_newton = newton_interpolation(x, y, yr)
    print(f"Population Prediction for {yr}: {pop_newton:,.0f}")

# Graphical Representation
x_fine = np.linspace(1990, 2015, 100)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Census Data') # Plots data points
plt.plot(x_fine, [newton_interpolation(x, y, t) for t in x_fine], '--', label='Newton/Lagrange') # Plots Newton curve
plt.plot(x_fine, cs_spline(x_fine), label='Cubic Spline') # Plots Spline curve

plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Population Interpolation Analysis")
plt.legend()
plt.grid(True)
plt.show()