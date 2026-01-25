import numpy as np

A = np.array([
    [5, 15, 55],
    [15, 55, 225],
    [55, 225, 979]
], dtype=float)
b = np.array([55, 225, 979], dtype=float)
x0 = np.array([0, 0, 0], dtype=float)

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