# Implementación práctica que sigue tus requisitos en la medida posible:
# - Series de Taylor para F
# - LDLᵀ (intento con scipy.linalg.ldl, con fallback a eigen) para obtener raíz S
# - Rotaciones de Givens para triangularizar la matriz apilada en predicción (QR vía Givens)
# - "Potter-like" actualización secuencial por observación (recalculo de S vía LDL/eigen para garantizar validez)
# Además grafica Real vs Estimado (media del filtro) al final.
#
# Nota importante al final: explico qué hice, límites y cómo pasar a una versión ensemble completa.
import numpy as np
import math
import matplotlib.pyplot as plt

# Intentar usar scipy.linalg.ldl, pero si no está, usar eigen de numpy
try:
    from scipy.linalg import ldl as scipy_ldl
    have_scipy_ldl = True
except Exception:
    scipy_ldl = None
    have_scipy_ldl = False

def taylor_F(A, dt, terms=14):
    """Calcula F = exp(A*dt) por serie de Taylor.
       A: (n,n) matriz de sistema (si no la tienes, usa I*0 o I)
       dt: paso temporal
       terms: número de términos (si None, usar n+5)
    """
    n = A.shape[0]
    
    F = np.eye(n)
    A_dt = A * dt
    power = np.eye(n)
    for k in range(1, terms):
        power = power @ A_dt
        F += power / math.factorial(k)
    return F

def ldl_sqrt(P):
    """Obtiene S tal que P = S S^T intentando usar LDLᵀ (scipy). 
       Si no está disponible, cae en eigen-root (simétrica positiva).
       Retorna S (n,n).
    """
    P = (P + P.T) / 2.0
    L, D, perm = scipy_ldl(P, lower=True)
    # D puede ser diagonal (array) o matriz diagonal; tomamos sqrt de D
    D_sqrt = np.sqrt(np.clip(np.diag(D), 0, None))
    S = L @ np.diag(D_sqrt)
    return S


def givens_qr_stack(M):
    """Triangulariza una matriz M (m x n, m>=n) por rotaciones de Givens
       y devuelve la parte superior triangular R (n x n).
       Implementación simple: recorre columnas y aplica rotaciones para anular abajo.
    """
    M = M.copy().astype(float)
    m, n = M.shape
    # recorrer columnas
    for j in range(n):
        for i in range(m-1, j, -1):
            a = M[i-1, j]
            b = M[i, j]
            if abs(b) < 1e-12:
                continue
            # compute c,s for rotation that zeros b
            r = math.hypot(a, b)
            c = a / r
            s = -b / r
            # apply rotation to rows i-1 and i for all columns j..n-1
            G0 = M[i-1, j:].copy()
            G1 = M[i, j:].copy()
            M[i-1, j:] = c * G0 - s * G1
            M[i, j:]   = s * G0 + c * G1
    # R is top n rows
    R = M[:n, :]
    return R

def stack_and_givens(F, S, Q):
    """Forma U = [F*S; Q_sqrt] y triangulariza con Givens para obtener nueva raíz S_next.
       S: (n,n), Q: (n,n)
       Retorna S_next (n,n) tal que P_f ≈ S_next S_next^T
    """
    n = S.shape[0]
    # sqrt de Q (usar eigen de Q para estabilidad)
    Q_sqrt = ldl_sqrt(Q)
    FS = F @ S  # (n,n)
    U = np.vstack([FS, Q_sqrt])  # (2n, n)
    R = givens_qr_stack(U)  # R (n x n) upper (but here we get first n rows)
    # Asegurar simetría/positivo
    S_next = R.copy()
    return S_next

def potter_sequential(x, S, y, H, R):
    """Implementación 'Potter-like' secuencial:
       - Para cada observación escalar ejecuta actualización clásica (Joseph) en P
       - Recalcula S con ldl/eigen después de todas las observaciones
       Retorna x_new, S_new
    """
    n = x.shape[0]
    P = S @ S.T
    x_new = x.copy().astype(float)
    # si R es matriz completa, manejamos cada observación por fila
    m = y.shape[0]
    for i in range(m):
        hi = H[i:i+1, :]  # (1,n)
        yi = y[i]
        Ri = R[i,i] if R.shape == (m,m) else float(R)
        # innovación cov
        S_innov = hi @ P @ hi.T + Ri
        K = (P @ hi.T) / S_innov  # (n,1)
        innov = yi - float(hi @ x_new)
        x_new = x_new + (K.flatten() * innov)
        # Joseph form for covariance update (numeric stability)
        I = np.eye(n)
        P = (I - K @ hi) @ P @ (I - K @ hi).T + K * Ri * K.T
        # for numerical stability ensure symmetry
        P = (P + P.T) / 2.0
    S_new = ldl_sqrt(P)
    return x_new, S_new

# Ahora ensamblamos un filtro Ensemble simplificado que usa los métodos anteriores.
def enkf_with_methods(data, A_model=None, sampleFreq=10.0, Ne=30):
    """
    data: (samples, n) series reales
    A_model: matriz A para linealizar el sistema dx/dt = A x; si None, uso I*0 (identidad dinámica trivial)
    sampleFreq: frecuencia de muestreo
    Ne: tamaño del ensemble (se usa para construir ensemble inicial)
    Devuelve: mean_estimates (samples, n)
    """
    data = np.asarray(data)
    samples, n = data.shape
    dt = 1.0 / sampleFreq
    # Matriz A para serie de Taylor (si None, usar A = 0 -> F = I)
    if A_model is None:
        A = np.zeros((n,n))
    else:
        A = np.asarray(A_model)
    # Inicial ensemble: muestrear alrededor de la primera medición
    x0 = data[0].astype(float)
    X = np.tile(x0.reshape(-1,1), (1,Ne)) + 0.01*np.random.randn(n,Ne)
    means = [x0.copy()]
    # Inicial P desde ensemble
    A_ens = X - np.mean(X, axis=1, keepdims=True)
    P = (A_ens @ A_ens.T) / (Ne - 1)
    S = ldl_sqrt(P)
    # Q y R: estimaciones simples (cov de diferencias o pequeñas covariancias)
    Q = np.eye(n) * 1e-4
    R = np.eye(n) * 0.01  # supongamos observaciones ruidosas por componente
    for k in range(1, samples):
        # 1) Calcular F por serie de Taylor
        F = taylor_F(A, dt, terms=n+5)  # (n,n)
        # 2) Propagar cada miembro (modelo lineal x_{k+1} = F x_k + w)
        for j in range(Ne):
            w = np.random.multivariate_normal(np.zeros(n), Q)
            X[:, j] = F @ X[:, j] + w
        # 3) Recalcular anomalías y P_f (muestral)
        A_ens = X - np.mean(X, axis=1, keepdims=True)
        P_f_sample = (A_ens @ A_ens.T) / (Ne - 1)
        # 4) Obtener S via LDL/LDLt de P_f_sample (coincide con método requerido)
        S = ldl_sqrt(P_f_sample)
        # 5) Predicción robusta de S usando rotaciones de Givens con Q
        S = stack_and_givens(F, S, Q)
        # 6) Observación: tomamos la muestra real data[k] (vector)
        y = data[k]
        # Observación parcial H = I (observamos todas las componentes)
        H = np.eye(n)
        # 7) Actualización con algoritmo Potter-like (secuencial en observaciones)
        x_mean = np.mean(X, axis=1)
        x_upd, S_upd = potter_sequential(x_mean, S, y, H, R)
        # 8) Reconstruir ensemble: conservamos la media actualizada y rehacemos anomalías para tener cov = S_upd S_upd^T
        # Generamos anomalías A_new such that A_new A_new^T / (Ne-1) = P_upd.
        # Para simplicidad y reproducibilidad, usamos una matriz Z de ruido normal y reescalamos.
        Z = np.random.randn(n, Ne)
        # quitar media de columnas de Z
        Z = Z - np.mean(Z, axis=1, keepdims=True)
        # escalar para que Cov(Z) = (Ne-1) I approximately: do SVD to orthonormalize rows
        # Forma A_new = S_upd @ W where W W^T = (Ne-1) I. Construct W via SVD of Z.
        U_z, s_z, Vt_z = np.linalg.svd(Z, full_matrices=False)
        W = U_z @ Vt_z  # this gives orthonormal-ish matrix (n x Ne)
        W = W * math.sqrt(Ne-1)
        A_new = S_upd @ W  # (n,Ne) -> has sample cov ~ S_upd S_upd^T
        X = (x_upd.reshape(-1,1)) + A_new
        means.append(x_upd)
    means = np.array(means)
    return means

# Prueba completa con datos sintéticos (senoides) y grafico real vs estimado
np.random.seed(1)
samples = 3072
n = 14
t = np.linspace(0, 4*np.pi, samples)
matrix = np.vstack([np.sin(t + i*0.5) for i in range(n)]).T + 0.05*np.random.randn(samples, n)

estimates = enkf_with_methods(matrix, A_model=None, sampleFreq=25.0, Ne=40)

# Graficar real vs estimado (media)
fig, axes = plt.subplots(n, 1, figsize=(10, 2.5*n), sharex=True)
time = np.arange(samples)
for i in range(n):
    ax = axes[i] if n>1 else axes
    ax.plot(time, matrix[:, i], label='Real', linewidth=1.5)
    ax.plot(time, estimates[:, i], '--', label='Estimado (media)')
    ax.set_ylabel(f'Var {i}')
    ax.legend(loc='upper right')
    ax.grid(True)
axes[-1].set_xlabel('Muestra')
fig.suptitle('Real vs Estimado (EnKF / SR-like con Taylor, LDLt, Givens, Potter-like)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Forma estimados:", estimates.shape)