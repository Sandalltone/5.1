import numpy as np

def jacobi(A, f, x0, eps=1e-6, max_iter=1000):
    n = len(f)
    x = x0.copy()
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (f[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x) < eps:
            return x_new
        x = x_new
    return x


n = 10
a = 0.5
A = np.zeros((n, n))
for i in range(n):
    A[i, i] = 2
    if i > 0:
        A[i, i-1] = 1 + a
    if i < n-1:
        A[i, i+1] = -1 - a
f = np.zeros(n)
f[0] = 1 - a
f[n-1] = 1 + a
x0 = np.zeros(n)
x_exact = np.ones(n)
x = jacobi(A, f, x0)

print("Приближённое решение: ", x)
print("Точное решение:", x_exact)
print("Погрешность:", np.linalg.norm(x - x_exact))