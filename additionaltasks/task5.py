import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return y - t**2 + 1

def exact_sol(t):
    return (t + 1)**2 - 0.5 * np.exp(t)

def basic_euler(h, t_end, y0=0.5):
    t = np.arange(0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i+1] = y[i] + h * f(t[i], y[i])
    return t, y

def modified_euler(h, t_end, y0=0.5):
    t = np.arange(0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        y_predict = y[i] + h * f(t[i], y[i])
        y[i+1] = y[i] + (h/2) * (f(t[i], y[i]) + f(t[i+1], y_predict))
    return t, y

#Execution and Comparison
t_range = 2.0
h_vals = [0.1, 0.05]

plt.figure(figsize=(14, 6))

for i, h in enumerate(h_vals):
    t, y_mod = modified_euler(h, t_range)
    _, y_basic = basic_euler(h, t_range)
    y_ex = exact_sol(t)
    
    plt.subplot(1, 2, i+1)
    plt.plot(t, y_ex, 'k', label='Exact', linewidth=2)
    plt.plot(t, y_mod, '--o', label=f'Modified Euler (h={h})')
    plt.plot(t, y_basic, ':x', label=f'Basic Euler (h={h})')
    plt.title(f'Comparison at h = {h}')
    plt.legend()
    plt.grid(True)

plt.show()