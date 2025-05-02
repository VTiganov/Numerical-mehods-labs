import numpy as np

def power_method_svd(A, max_iter=1000, tol=1e-6):
    """
    Возвращает (sigma, u, v) — наибольшее сингулярное число и соответствующие векторы для матрицы A.
    """
    _, n = A.shape
    # Инициализируем случайный вектор v
    v = np.random.randn(n)
    v /= np.linalg.norm(v)

    for _ in range(max_iter):
        w = A.T @ (A @ v)
        v_new = w / np.linalg.norm(w)
        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            break
        v = v_new

    u = A @ v
    sigma = np.linalg.norm(u)
    if sigma > 0:
        u /= sigma
    return sigma, u, v