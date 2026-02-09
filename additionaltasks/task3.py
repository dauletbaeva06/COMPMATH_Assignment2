import sympy as sp

def task_3_comparison():
    x = sp.Symbol('x')
    x_val = 0.2
    
    y_p = 1
    for _ in range(3):
        y_p = 1 + sp.integrate(y_p - x**2, (x, 0, x))
    val_picard = float(y_p.subs(x, x_val))
    
    val_taylor = (1 + (x_val * 1) + (x_val**2 / 2 * 1) + 
                  (x_val**3 / 6 * -1) + (x_val**4 / 24 * -1))
    
    print(f"Picard (3 iterations) at x=0.2: {val_picard:.6f}")
    print(f"Taylor (4th order) at x=0.2:    {val_taylor:.6f}")

if __name__ == "__main__":
    task_3_comparison()