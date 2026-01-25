import numpy as np

A = np.array([
    [4, -1, -1, 0],
    [-1, 4, 0, -1],
    [-1, 0, 4, -1],
    [0, -1, -1, 4]
], dtype=float)
b = np.array([-1, 3, 7, 11], dtype=float)
x0 = np.array([0, 0, 0, 0], dtype=float) 

def cramer_method(A, b):
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-12:
        return "Failure: Determinant is zero. No unique solution."
    
    n = len(b)
    roots = np.zeros(n)
    for i in range(n):
        A_temp = A.copy()
        A_temp[:, i] = b
        roots[i] = np.linalg.det(A_temp) / det_A
        
    print("Cramer's Method Root Estimate:", roots)
    return roots

res1 = cramer_method(A, b)