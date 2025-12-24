import numpy as np

A = np.array([
    [4, -1, -1, 0],
    [-1, 4, 0, -1],
    [-1, 0, 4, -1],
    [0, -1, -1, 4]
], dtype=float)
b = np.array([-1, 3, 7, 11], dtype=float)
x0 = np.array([0, 0, 0, 0], dtype=float)

def gauss_jordan_method(A, b):
    n = len(b)
    Ab = np.column_stack((A, b))
    for i in range(n):
        Ab[i] = Ab[i] / Ab[i,i]
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[j,i] * Ab[i]
    
    roots = Ab[:, -1]
    print("Gauss-Jordan Method Root Estimate:", roots)
    return roots

res3 = gauss_jordan_method(A, b)