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
max_iter = 50

def relaxation_method(A, b, x_init, omega=1.1, tol=1e-3, max_iter=100):
    n = len(b)
    x = x_init.copy()
    print(f"\n{'Iter':<5} | {'Approximation Vector':<30} | {'Rel Error'}")
    print("-" * 60)
    
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i+1, n))
            # Standard iteration value
            x_gs = (b[i] - s1 - s2) / A[i, i]
            # Applying relaxation [cite: 11]
            x[i] = (1 - omega) * x_old[i] + omega * x_gs
            
        error = np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf)
        print(f"{k:<5} | {str(np.round(x, 4)):<30} | {error:.6f}")
        
        if error < tol:
            print(f"Result: Success. Tolerance reached at iteration {k}.")
            return x
            
    return x

res6 = relaxation_method(A, b, x0, omega=1.1, tol=tol, max_iter=max_iter)