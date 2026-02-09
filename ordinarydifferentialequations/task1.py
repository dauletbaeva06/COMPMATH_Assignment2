import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.integrate import solve_ivp

# Model Definitions
def f(x, y):
    """The ODE: dy/dx = e^x - y^2"""
    return np.exp(x) - y**2

def get_exact_sol(x_span):
    """Using a high-precision solver as the 'Exact' reference."""
    sol = solve_ivp(f, [x_span[0], x_span[-1]], [1], t_eval=x_span, rtol=1e-12)
    return sol.y[0]

# a) Picard's Method (3rd approximation)
# y1 = 1 + int(e^t - 1^2)dt = e^x - x
# y2 = 1 + int(e^t - (e^t - t)^2)dt
def picard_approx(x):
    return 1 + (x**2)/2 + (x**3)/6

# b) Taylor Series Method (up to 2nd derivative)
# y' = e^x - y^2
# y'' = e^x - 2y(y')
def taylor_step(x, y, h):
    yp = f(x, y)
    ypp = np.exp(x) - 2 * y * yp
    return y + h * yp + (h**2 / 2) * ypp

# Solver Engine
def solve_all_methods(h, x_end=2.0):
    x_vals = np.arange(0, x_end + h, h)
    n = len(x_vals)
    
    # Initialize arrays
    y_euler = np.zeros(n)
    y_mod_euler = np.zeros(n)
    y_rk3 = np.zeros(n)
    y_rk4 = np.zeros(n)
    y_taylor = np.zeros(n)
    y_picard = np.array([picard_approx(xi) for xi in x_vals])
    
    # Initial Condition y(0) = 1
    y_euler[0] = y_mod_euler[0] = y_rk3[0] = y_rk4[0] = y_taylor[0] = 1
    
    for i in range(n - 1):
        xi, yi_e, yi_me, yi_r3, yi_r4, yi_t = x_vals[i], y_euler[i], y_mod_euler[i], y_rk3[i], y_rk4[i], y_taylor[i]
        
        # c) Euler Method
        y_euler[i+1] = yi_e + h * f(xi, yi_e)
        
        # d) Modified Euler Method
        k1_me = f(xi, yi_me)
        y_predict = yi_me + h * k1_me
        y_mod_euler[i+1] = yi_me + (h/2) * (k1_me + f(xi + h, y_predict))
        
        # e) RK 3rd Order
        k1 = h * f(xi, yi_r3)
        k2 = h * f(xi + h/2, yi_r3 + k1/2)
        k3 = h * f(xi + h, yi_r3 - k1 + 2*k2)
        y_rk3[i+1] = yi_r3 + (1/6) * (k1 + 4*k2 + k3)
        
        # f) RK 4th Order
        m1 = h * f(xi, yi_r4)
        m2 = h * f(xi + h/2, yi_r4 + m1/2)
        m3 = h * f(xi + h/2, yi_r4 + m2/2)
        m4 = h * f(xi + h, yi_r4 + m3)
        y_rk4[i+1] = yi_r4 + (m1 + 2*m2 + 2*m3 + m4) / 6
        
        # b) Taylor Method
        y_taylor[i+1] = taylor_step(xi, yi_t, h)
        
    return x_vals, y_picard, y_taylor, y_euler, y_mod_euler, y_rk3, y_rk4

# Execution and Output
for h_val in [0.1, 0.2]:
    x, y_p, y_t, y_e, y_me, y_r3, y_r4 = solve_all_methods(h_val)
    y_exact = get_exact_sol(x)
    
    # 1.2 Tabulation
    table_data = []
    for i in range(len(x)):
        table_data.append([i, f"{x[i]:.1f}", f"{y_e[i]:.4f}", f"{y_r4[i]:.4f}", f"{y_exact[i]:.4f}", f"{abs(y_r4[i]-y_exact[i]):.2e}"])
    
    print(f"\n--- Results for h = {h_val} ---")
    print(tabulate(table_data, headers=["Step n", "xi", "Euler", "RK4", "Exact", "Abs Error (RK4)"]))

    # Graphical Comparison
    plt.figure(figsize=(12, 5))
    
    # Solutions Plot
    plt.subplot(1, 2, 1)
    plt.plot(x, y_exact, 'k-', label='Exact', linewidth=2)
    plt.plot(x, y_e, 'r--', label='Euler')
    plt.plot(x, y_me, 'g-.', label='Mod Euler')
    plt.plot(x, y_r4, 'b:', label='RK4')
    plt.title(f"Numerical Solutions (h={h_val})")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()

    # Error Plot
    plt.subplot(1, 2, 2)
    plt.plot(x, abs(y_e - y_exact), label='Euler Error')
    plt.plot(x, abs(y_me - y_exact), label='Mod Euler Error')
    plt.plot(x, abs(y_r4 - y_exact), label='RK4 Error')
    plt.yscale('log')
    plt.title(f"Error vs x (h={h_val})")
    plt.xlabel("x"); plt.ylabel("Absolute Error"); plt.legend()
    plt.tight_layout()
    plt.show()