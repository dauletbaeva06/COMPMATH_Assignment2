import numpy as np
import pandas as pd

x_un = np.array([0.6, 1.5, 1.6, 2.5, 3.5])
y_un = np.array([0.3734, 0.9036, 0.3261, 0.08422, 0.01596])

def exact_deriv(x):
    # Product rule on 5*x*exp(-2x)
    return 5 * np.exp(-2*x) * (1 - 2*x)

def lagrange_derivative(x_val, x_pts, y_pts):
    n = len(x_pts)
    deriv = 0
    for i in range(n):
        inner_sum = 0
        for j in range(n):
            if i == j: continue
            prod = 1
            for k in range(n):
                if k == i or k == j: continue
                prod *= (x_val - x_pts[k]) / (x_pts[i] - x_pts[k])
            inner_sum += prod / (x_pts[i] - x_pts[j])
        deriv += y_pts[i] * inner_sum
    return deriv

results_b = []
for val in x_un:
    num_d = lagrange_derivative(val, x_un, y_un)
    ext_d = exact_deriv(val)
    results_b.append([val, ext_d, num_d, abs(ext_d - num_d)])

df_b = pd.DataFrame(results_b, columns=["x", "Exact f'(x)", "Numerical f'(x)", "Abs Error"])
print(df_b.to_string(index=False))