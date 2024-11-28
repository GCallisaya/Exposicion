import numpy as np

def gauss_seidel(A, b, x0, tol=1e-10, max_iter=1000):
    n = len(b)
    x = x0.copy()
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]
        
        error = np.linalg.norm(x - x_old, ord=np.inf)
        if error < tol:
            break

        print(f"Iteración {k+1}: x = {x}")

    return x

# Sistema de ecuaciones para el problema
A = np.array([
    [2, 1, -1],
    [1, 3, 1],
    [-1, 1, 4]
])

b = np.array([20, 30, 10])
x0 = np.zeros(len(b))

# Resolver el sistema de ecuaciones
sol = gauss_seidel(A, b, x0)

print(f"Solución encontrada: {sol}")
