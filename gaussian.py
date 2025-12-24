import numpy as np  

A = np.array([
    [4, -1, -1, 0],
    [-1, 4, 0, -1],
    [-1, 0, 4, -1],
    [0, -1, -1, 4]
], dtype=float)
b = np.array([-1, 3, 7, 11], dtype=float)
x0 = np.array([0, 0, 0, 0], dtype=float)

def gaussian_method(A, b):
    n = len(b)
    Ab = np.column_stack((A, b))

    for i in range(n):
        for j in range(i+1, n):
            ratio = Ab[j,i] / Ab[i,i]
            Ab[j, i:] -= ratio * Ab[i, i:]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i,-1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i,i]
    
    print("Gaussian Method Root Estimate:", x)
    return x

res2 = gaussian_method(A, b)