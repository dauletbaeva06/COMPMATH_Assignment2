import numpy as np

def trapezoidal(f, a, b, n):
    x_space = np.linspace(a, b, n+1)
    y_space = f(x_space)
    return ( (b-a)/(2*n) ) * (y_space[0] + 2*np.sum(y_space[1:-1]) + y_space[-1])

def simpson13(f, a, b, n):
    if n % 2 != 0: n += 1 # n must be even for Simpson's 1/3
    x_space = np.linspace(a, b, n+1)
    y_space = f(x_space)
    h = (b-a)/n
    return (h/3) * (y_space[0] + 4*np.sum(y_space[1:-1:2]) + 2*np.sum(y_space[2:-2:2]) + y_space[-1])

# Test [cite: 40]
print(f"Trapezoidal (n=100): {trapezoidal(np.sin, 0, np.pi, 100):.4f}")
print(f"Simpson's 1/3 (n=100): {simpson13(np.sin, 0, np.pi, 100):.4f}")