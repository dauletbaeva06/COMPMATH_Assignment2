import numpy as np
import pandas as pd

t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
x = np.array([0, 0.7, 1.8, 3.4, 5.1, 6.3, 7.3, 8.0, 8.4], dtype=float)
h = t[1] - t[0]  # Step size is 2

def get_diff_table(y):
    n = len(y)
    table = np.zeros((n, n))
    table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i+1, j-1] - table[i, j-1]
    return table

diff_table = get_diff_table(x)

# Newton's Forward at t=2 
# Formula: v = (1/h) * (Δy - 1/2 Δ²y + 1/3 Δ³y)
v_2 = (1/h) * (diff_table[1, 1] - 0.5*diff_table[1, 2] + (1/3)*diff_table[1, 3])
a_2 = (1/h**2) * (diff_table[1, 2] - diff_table[1, 3])

# Newton's Backward at t=14 
# Using backward differences from the table
v_14 = (1/h) * (diff_table[6, 1] + 0.5*diff_table[5, 2])
a_14 = (1/h**2) * (diff_table[5, 2])

print(f"Velocity at t=2: {v_2:.4f} m/s, Acceleration: {a_2:.4f} m/s²")
print(f"Velocity at t=14: {v_14:.4f} m/s, Acceleration: {a_14:.4f} m/s²")