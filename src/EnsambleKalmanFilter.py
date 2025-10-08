import numpy as np
from scipy.linalg import ldl


'''
1. Estado inicial se hace el anterior
2. Matriz de covarianza (Funcion) - 
3. Convertir matriz de covarianza en matriz raiz cuadrada (LDLt) -
4. Predecir siguiente estado X -
    - Calcular error Gaussiano W -
    - Calcular matriz de transición "Series Taylor" -
5. Predecir nueva matriz de covarianza "Rotación de Givens" (Funcion)
    - Calcular Q, matriz de covarianza del ruido Gaussiano W
    - Calcular matriz de transición "Series Taylor" F
6. Calcular el dato de entrada con la muestra real
    - Calcular H Matriz de transformación
    - Calcular Z error gaussiano 
7. Actualización con ganancia de Kalman para la matriz y el estado "Potter"
8. Nuevo estado se convierte en el anterior

'''

# Crear Matriz de Covarianza inicial
def initCovMatrix(m):
    covMatrix = np.cov(m)
    return covMatrix

# Aplicar LDLt para convertir la matriz de covarianza en matriz raiz cuadrada
def transformCovMatrix2sqrtMatrix(matrix):
    l, d, _ = ldl(matrix, True) # Use the upper part
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

    for i in range(1,m-1):
        A = A @ (A * dt) # Calcular (A*dt)^i
        factorial *= i # Calcular i!
        F += A / factorial

    return F
                
def calcNextS(Q, S, F):
    m = len(S)
    #print(Q)
    Qsqrt = np.sqrt(Q)
    QT2 = Qsqrt.T
    
    #print(S.shape, F.shape, QT2.shape)

    U = np.vstack([
        S.T @ F.T,     
        QT2 
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
    
def Potter(x, S, y, H, R):
    x0 = x.copy()
    S0 = S.copy()

    for i in range(len(y)):
        Hi = H[i, :].reshape(1,len(x))
        yi = y[i]
        Ri = R[i]

        phi = S0.T @ Hi.T
        ai = 1/((phi.T @ phi) + Ri )
        gammai = ai/(1 + np.sqrt(ai * Ri))
        S0 = S0 @ (np.eye(len(S)) - ai*gammai * (phi @ phi.T))
        Ki = S0 @ phi
        x0 = x0 + (Ki[:,0]) * (yi - (Hi @ x0)) 
    return x0, S0 

def calcY(H, x, z):
    return np.dot(H, x) + z

def calcH(x, list_idx = [0,1]):
    H = []
    for idx in list_idx:
        row = np.zeros((len(x)))
        row[idx] = 1
        H.append(row)
    return np.array(H)

def EnKL(matrix, sampleFreq):
    samples = len(matrix)
    P0 = initCovMatrix(matrix.T)
    S = transformCovMatrix2sqrtMatrix(P0)   # 14x14
    dt = 1/sampleFreq
    xP = [matrix[0]] # 14,1

    # Iterar muestras
    for i in range(1, samples):
        # Predecir siguiente estado
        W = calcGaussError((14,14)) # 14x14
        F = calcTransitionMatrix(matrix[i], dt) # 14x14
        new_xp = np.dot(F, xP[i-1]) + W # 14x14
        
        # Predecir siguiente matriz raiz cuadrada de procesos 
        Q = initCovMatrix(W)  # 14x14
        S = calcNextS(Q, S, F)  # 14x14

        # Calcular dato de entrada Y
        H = calcH(matrix[i], list_idx=range(len(matrix[i])))  # 14x14
        z = calcGaussError((14,14))  # 14x14
        y = calcY(H, xP[i-1], z)  # 14x14

        # Actualizar x y S 
        R = initCovMatrix(z) # Matriz de covarianza de errores   14x14
        new_x, S = Potter(new_xp, S, y, H, R)  # 14x14  # 14x14

        xP.append(new_x)
    
    shapes = set([np.array(x).shape for x in xP])
    print("Formas detectadas:", shapes)

    return np.array(xP, dtype=np.float32)