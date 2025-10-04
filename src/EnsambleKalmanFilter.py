import numpy as np
import scipy


'''
1. Estado inicial se hace el anterior
2. Matriz de covarianza (Funcion) - 
3. Convertir matriz de covarianza en matriz raiz cuadrada (LDLt) -
4. Predecir siguiente estado X -
    - Calcular error Gaussiano W -
    - Calcular matriz de transición "Series Taylor" -
5. Predecir nueva matriz de covarianza "Rotación de Givens" (Funcion)
    - Calcular Q, matriz de covarianza del ruido Gaussiano W
    - Calcular matriz de transición "Series Taylor"
6. Calcular el dato de entrada con la muestra real
    - Calcular H Matriz de transformación
    - Calcular Z error gaussiano 
7. Actualización con ganancia de Kalman para la matriz y el estado "Potter"
8. Nuevo estado se convierte en el anterior

'''

# Crear Matriz de Covarianza inicial
def initCovMatrix(m):
    covMatrix = np.eye(m) * 0.01
    return covMatrix

# Aplicar LDLt para convertir la matriz de covarianza en matriz raiz cuadrada
def transformCovMatrix2sqrtMatrix(matrix):
    l, d, _ = scipy.ldl(matrix, True) # Use the upper part
    S = l.dot(np.sqrt(d))
    return S

# Calcular error gaussiano 
def calcGaussError(m):
    return np.random.normal(size=m)

def calcTransitionMatrix(sample, dt):
    m = len(sample)
    F = np.eye(m)
    A = np.eye(m)
    factorial = 1.0

    for i in range(m):
        A = A @ (A * dt) # Calcular (A*dt)^i
        factorial *= i # Calcular i!
        F += A / factorial

    return F

# Calcular Q matriz de covarianza del ruida Gaussiano W
def calcGaussCovMatrix(noise_matrix):
    m = len(noise_matrix)
    Q = np.zeros((m,m))

    for i in range(m):
        for j in range(m):
            if i == j:
                # Calcula varianza 
                pass
    
    return Q
                
def calcNextS(Q, S, F):
    m = len(S)

    U = np.vstack([
        S.T @ F.T,     
        np.linalg.cholesky(Q) #  DUDA: Como calcular Q T/2
    ])

    # Iterar sobre columnas
    for j in range(m):
        # Iterar sobre filas desde abajo hacia arriba
        for i in range(2*m-1, j, -1):
            a = U[i-1, j]
            b = U[i, j]

            # Cálculo de rotación de Givens
            if b == 0:
                c = 1.0
                s = 0.0
            else:
                if abs(b) > abs(a):
                    r = a / b
                    s = 1.0 / np.sqrt(1 + r**2)
                    c = s * r
                else:
                    r = b / a
                    c = 1.0 / np.sqrt(1 + r**2)
                    s = c * r

            # Construir la rotación 2x2
            G = np.eye(2)
            G[0, 0] = c
            G[0, 1] = -s
            G[1, 0] = s
            G[1, 1] = c

            # Aplicar la rotación a las filas i-1 e i de U
            temp = U[i-1:i+1, :].copy()
            U[i-1:i+1, :] = G @ temp

    # Extraer la parte superior m x m como nueva raíz cuadrada
    nextS = U[:m, :]
    return nextS
    

def EnKL(matrix, sampleFreq):
    samples = len(matrix)
    m = len(matrix[0])
    P0 = initCovMatrix(m)
    S0 = transformCovMatrix2sqrtMatrix(P0)
    dt = 1/sampleFreq
    xP = [matrix[0]]

    # Iterar muestras
    for i in range(1, samples):
        # Predecir siguiente estado
        W = calcGaussError(m)
        F = calcTransitionMatrix(matrix[i], dt)
        new_xp = np.dot(F, xP[i-1]) + W
        xP.append(new_xp)

        # Predecir siguiente matriz raiz cuadrada de procesos 