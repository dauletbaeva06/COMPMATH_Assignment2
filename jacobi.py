import numpy as np  

A = np.array([
    [4, -1, -1, 0],
    [-1, 4, 0, -1],
    [-1, 0, 4, -1],
    [0, -1, -1, 4]
], dtype=float)
b = np.array([-1, 3, 7, 11], dtype=float)
x0 = np.array([0, 0, 0, 0], dtype=float)
tol = 1e-3
max_iter = 100

def jacobi_method(A, b, x_init, tol, max_iter):
    n = len(b)
    x = x_init.copy()
    print(f"\n{'Iter':<5} | {'Approximation Vector':<30} | {'Rel Error'}")
    print("-" * 60)
    
    for k in range(1, max_iter + 1):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i, i]
        
        error = np.linalg.norm(x_new - x, ord=np.inf) / np.linalg.norm(x_new, ord=np.inf)
        x = x_new
        print(f"{k:<5} | {str(np.round(x, 4)):<30} | {error:.6f}")
        
        if error < tol:
            print(f"Result: Success. Tolerance reached at iteration {k}.")
            return x
            
    print("Result: Maximum iterations reached.")
    return x

res4 = jacobi_method(A, b, x0, tol, max_iter)