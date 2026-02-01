import numpy as np
import pandas as pd

t_p = np.arange(0, 26, 2) # 0 to 24 hours with step of 2
P_p = np.array([500, 480, 450, 600, 800, 950, 1000, 980, 920, 850, 700, 550, 500])
h = 2 # Constant time interval

# 1. Trapezoidal Rule
energy_trap = np.trapezoid(P_p, t_p)

# 2. Simpson's 1/3 Rule 
def simpson13(y, h):
    n = len(y) - 1
    return (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

# 3. Simpson's 3/8 Rule
def simpson38(y, h):
    n = len(y) - 1
    if n % 3 != 0: return None
    res = y[0] + y[-1]
    for i in range(1, n):
        if i % 3 == 0:
            res += 2 * y[i]
        else:
            res += 3 * y[i]
    return (3 * h / 8) * res

# 4. Boole's Rule
def booles_rule(y, h):
    n = len(y) - 1
    if n % 4 != 0: return None
    res = 0
    for i in range(0, n, 4):
        res += (2*h/45) * (7*y[i] + 32*y[i+1] + 12*y[i+2] + 32*y[i+3] + 7*y[i+4])
    return res

# Calculate results
results = {
    "Method": ["Trapezoidal", "Simpson's 1/3", "Simpson's 3/8", "Boole's Rule"],
    "Energy (MWh)": [
        energy_trap, 
        simpson13(P_p, h), 
        simpson38(P_p, h), 
        booles_rule(P_p, h)
    ]
}

# Presenting results in a table
df_energy = pd.DataFrame(results)
print("--- TOTAL ENERGY CONSUMPTION (24 HOURS) ---")
print(df_energy.to_string(index=False))