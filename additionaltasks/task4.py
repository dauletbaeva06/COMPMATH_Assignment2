import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return -2*y + t

def exact_sol(t):
    return 0.25 * (2*t - 1 + 5*np.exp(-2*t))

def euler_method(h, t_end):
    t_values = np.arange(0, t_end + h, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = 1
    
    for i in range(len(t_values) - 1):
        y_values[i+1] = y_values[i] + h * f(t_values[i], y_values[i])
        
    return t_values, y_values

#Part (a) & (b): Standard Run with h=0.1
h = 0.1
t_end = 1.0
t_num, y_num = euler_method(h, t_end)
y_exact = exact_sol(t_num)
error = np.abs(y_exact - y_num)

plt.figure(figsize=(12, 5))

# Plot 1: Solutions
plt.subplot(1, 2, 1)
plt.plot(t_num, y_num, 'bo-', label=f'Euler (h={h})')
plt.plot(t_num, y_exact, 'r-', label='Exact')
plt.title('Euler vs Exact Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Plot 2: Absolute Error
plt.subplot(1, 2, 2)
plt.plot(t_num, error, 'g^-', label='Absolute Error')
plt.title('Error over Time')
plt.xlabel('t')
plt.ylabel('|Exact - Numerical|')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Part (c): Step Size Investigation
step_sizes = [0.2, 0.1, 0.05, 0.01]
max_errors = []

for s in step_sizes:
    t_s, y_s = euler_method(s, t_end)
    max_err = np.max(np.abs(exact_sol(t_s) - y_s))
    max_errors.append(max_err)

# Convergence Plot
plt.figure(figsize=(6, 4))
plt.loglog(step_sizes, max_errors, 'ko-', label='Max Error')
plt.title('Error vs Step Size (Log-Log)')
plt.xlabel('Step Size h')
plt.ylabel('Max Absolute Error')
plt.grid(True, which="both", ls="-")
plt.show()

# Order of Convergence Calculation
p = (np.log(max_errors[-1]) - np.log(max_errors[0])) / (np.log(step_sizes[-1]) - np.log(step_sizes[0]))
print(f"Estimated Order of Convergence (p): {p:.4f}")