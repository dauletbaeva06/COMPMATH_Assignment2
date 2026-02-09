import sympy as sp

def task_1_picard():
    x = sp.Symbol('x')
    y = sp.Function('y')
    y_n = 1
    
    for i in range(3):
        f_xy = y_n**2 + 3*x
        y_n = 1 + sp.integrate(f_xy, (x, 0, x))
        print(f"Iteration {i+1}: y = {y_n}")

    result = y_n.subs(x, 0.1)
    print(f"\nApproximate value at x=0.1: {float(result):.5f}")

if __name__ == "__main__":
    task_1_picard()