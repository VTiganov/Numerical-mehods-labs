import numpy as np

def jacobi_eigenvals(A, tol=1e-10, max_iter=100):
    n = A.shape[0]
    V = np.eye(n)

    for _ in range(max_iter):
        max_off = 0
        p, q = 0, 0

        # Поиск максимального внедиагонального элемента
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i,j]) > max_off:
                    max_off = abs(A[i,j])
                    p, q = i, j

        if max_off < tol: break

        # Вычисление угла вращения
        if A[p,p] == A[q,q]:
            theta = np.pi/4
        else:
            theta = 0.5 * np.arctan(2*A[p,q]/(A[p,p]-A[q,q]))

        c, s = np.cos(theta), np.sin(theta)

        # Обновление матрицы
        row_p = A[p,:].copy()
        row_q = A[q,:].copy()

        A[p,:] = c*row_p - s*row_q
        A[q,:] = s*row_p + c*row_q

        col_p = A[:,p].copy()
        col_q = A[:,q].copy()

        A[:,p] = c*col_p - s*col_q
        A[:,q] = s*col_p + c*col_q

        # Обновление матрицы вращений
        V[:, [p,q]] = np.dot(V[:, [p,q]], np.array([[c, s], [-s, c]]))

    return np.diag(A), V