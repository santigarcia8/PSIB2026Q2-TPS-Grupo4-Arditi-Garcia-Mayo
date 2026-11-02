import matplotlib.pyplot as plt
import numpy as np
from Módulos.pantompkins import detectar_qrs


def referencia_materna(señal, fs, q=1):
    
    #PASO 1 - DETECTAR QRS MATERNOS
    r_maternos = detectar_qrs(señal, fs, refractario=0.3, ventana=0.15, devolver_etapas=False)

    #PASO 2 - ARMO MATRIZ DE SVD
    w1 = 40
    w2 = 60
    r_maternos = r_maternos[(r_maternos - w1 >= 0) & (r_maternos + w2 + 1 <= len(señal))] #guarda los valores que entren. Puede que un QRS quede muy cerca del borde y al restarle w1 no haya muestras
    matriz =[]
    for i in range(len(r_maternos)):
        matriz.append(señal[r_maternos[i]-w1:r_maternos[i]+w2+1])
    
    matriz = np.array(matriz) #armo la matriz como establece el paper, con los latidos en fila
    
    
    #PASO 3 y 4 - HAGO ESTIMACIÓN DE LA MATRIZ Y VOY ACTUALIZANDOLA
    N=20 #seteo 20 latidos como en el paper
    matriz_estimada = np.zeros_like(matriz)
    for j in range(len(matriz)):
            ini = max(0, j - N + 1) #donde voy a iniciar a contar las 20 muestras
            Yj = matriz[ini:j+1] #cual es la fila a sumar
            U, s, Vt = np.linalg.svd(Yj, full_matrices=False) #hago la descomposición en SVD donde me dan los vectores singulares y los valores singulares en la diagonal de s
            Yj_est = (U[:, :q] * s[:q]) @ Vt[:q, :] #hago la estimación de esa fila (latido) con las q dominantes (la primera en este caso)
            matriz_estimada[j] = Yj_est[-1] #le pongo la estimación en la matriz de estimaciones
    
    #PASO 5 - RECONSTRUYO LO QUE VA A SER MI SEÑAL ESTIMADA DE ECG
    x_estimada = np.zeros(len(señal))
    es_qrs = np.zeros(len(señal), dtype=bool)
    for i, r in enumerate(r_maternos):
        x_estimada[r - w1 : r + w2 + 1] = matriz_estimada[i] #guardo las señales que son los QRS maternos estimados
        es_qrs[r - w1 : r + w2 + 1] = True 
    
    #PASO 6 - SUAVIZADO DEL TRAMO INTER-QRS
    L = 30 #el paper toma ventanas de 30 para suavizar entre los picos
    M = np.full(len(señal), L) 

    for r in r_maternos: #para cada pico R se va a correr un rango
            iA = np.arange(r - w1 - L, r - w1) #en el paper lo llama A
            iA = iA[iA >= 0]
            M[iA] = r - w1 - iA #arma un vector donde pone la distancia de esa muestra al incio del QRS
            iD = np.arange(r + w2 + 1, r + w2 + L + 1) #se fija las posiciones después del QRS, lo que llama B
            iD = iD[iD < len(señal)]
            M[iD] = iD - r - w2

    x_ref = x_estimada.copy() #la copio porque la aprte del QRS la dejo y el resto es el suaviazdo
    for i in np.where(~es_qrs)[0]: #donde NO sea QRS, es decir, entre los tramos
        m = M[i] if M[i] % 2 == 1 else M[i] - 1
        h = (m - 1) // 2 #de toda esa ventana que NO es QRS me armo la mitad
        x_ref[i] = señal[max(i - h, 0): i + h + 1].mean() #pone la media en ese tramo

    return x_ref, es_qrs

def filtro_rls(señal, x_ref, lam=1.0):
    H = 20 #cantidad de coeficientes del papaer
    delta = 1e-4 #elijo un valor chico para la matriz porque no dice dónde arranca
    w = np.zeros(H)         #vector de pesos
    P = np.eye(H) / delta          #donde voy a guardar la inversa
    f_est = np.zeros(len(señal))         # donde guardo el FECG

    for n in range(H, len(señal)):
        xv = x_ref[n-H:n][::-1]  #toma las muestras anteriores y las ordena de más reciente a más lejana
        Px = P @ xv #multiplica los pesos con las muestras ordenadas
        P = (P - np.outer(Px, Px) / (lam + xv @ Px)) / lam    #P es la inversa de la matriz de autocorrelación entre las muestras que se actualiza 
        e = señal[n] - w @ xv                        #evalúa el error que sería el FECG
        w = w + (P @ xv) * e                         #actualiza los pesos
        f_est[n] = e                                 #el FECG estimado es el error
    
    return f_est