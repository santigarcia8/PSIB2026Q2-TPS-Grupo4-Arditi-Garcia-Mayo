import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def filtro_pasabanda(señal, fs):
    #Pan-Tompkins se queda con 5-15 Hz porque ahí está la mayor parte de la energía del QRS
    #y atenúa la onda P, la T, la deriva y el ruido muscular
    b, a = butter(2, [5, 15], btype="bandpass", fs=fs)
    señal_pb = filtfilt(b, a, señal) #filtfilt no desfasa, así los R no se corren
    return señal_pb

def derivada(señal_pb, fs):
    #derivada de 5 puntos de Pan-Tompkins: y(n) = fs/8 * (x(n+2) + 2x(n+1) - 2x(n-1) - x(n-2))
    #la uso centrada para que no meta retardo
    h = np.array([1, 2, 0, -2, -1]) * fs / 8
    señal_ext = np.pad(señal_pb, 2, mode="edge") #repito los extremos para que no aparezca un salto falso en los bordes
    señal_d = np.convolve(señal_ext, h, mode="valid") #"valid" me devuelve el mismo largo que la original
    return señal_d

def cuadrado(señal_d):
    #elevo al cuadrado: todo queda positivo y se resaltan las pendientes grandes (QRS materno)
    #frente a las chicas (fetal, ruido)
    señal_c = señal_d**2
    return señal_c

def integracion(señal_c, fs, ventana=0.15):
    #promedio móvil de 150 ms (duración de un QRS adulto): junta los picos de cada QRS en un solo bulto
    #ventana queda como parámetro para poder usar una más corta con el fetal
    N = int(ventana*fs)
    señal_i = np.convolve(señal_c, np.ones(N)/N, mode="same") #"same" la deja centrada, sin retardo
    return señal_i

def picos_candidatos(señal_i, fs, refractario=0.3):
    #busco los máximos locales del integrado separados al menos por el período refractario (300 ms materno)
    #acá todavía se mezclan QRS con picos de ruido o fetales: los separo después con los umbrales
    candidatos, _ = find_peaks(señal_i, distance=int(refractario*fs))
    return candidatos

def umbrales(señal_i, señal_d, candidatos, fs):
    #inicializo con los primeros 2 s (fase de aprendizaje de Pan-Tompkins)
    inicio = señal_i[:int(2*fs)]
    SPKI = np.max(inicio)/3 #nivel estimado de los picos de QRS
    NPKI = np.mean(inicio)/2 #nivel estimado de los picos de ruido
    qrs = [] #acá guardo los candidatos que son QRS
    umbral = [] #guardo el umbral en cada candidato para poder graficarlo
    medio = int(0.075*fs) #media ventana de integración: el QRS está a +-75 ms del pico del integrado
    pendiente_ant = 0 #pendiente máxima del último QRS
    descartados = [] #candidatos que tomé como ruido desde el último QRS (para el searchback)
    for p in candidatos:
        thr = NPKI + 0.25*(SPKI - NPKI)

        #searchback: si pasó demasiado tiempo sin QRS, seguro se me escapó uno.
        #Vuelvo a mirar los descartados con un umbral más bajo (la mitad) y me quedo con el más alto
        if len(qrs) >= 2 and len(descartados) > 0:
            rr_medio = np.mean(np.diff(qrs[-9:])) #promedio de los últimos 8 RR
            if (p - qrs[-1]) > 1.66*rr_medio:
                mejor = max(descartados, key=lambda c: señal_i[c])
                if señal_i[mejor] > 0.5*thr:
                    qrs.append(mejor)
                    SPKI = 0.25*señal_i[mejor] + 0.75*SPKI #en el searchback Pan-Tompkins actualiza con más peso
                    pendiente_ant = np.max(np.abs(señal_d[max(mejor-medio, 0):mejor+medio]))
                    thr = NPKI + 0.25*(SPKI - NPKI)
                descartados = []

        umbral.append(thr)
        pendiente = np.max(np.abs(señal_d[max(p-medio, 0):p+medio])) #pendiente máxima de este candidato
        es_qrs = señal_i[p] > thr

        #regla de la onda T de Pan-Tompkins: si llega a menos de 360 ms del QRS anterior
        #y tiene menos de la mitad de su pendiente, es una onda T (o ruido), no un QRS
        if es_qrs and len(qrs) > 0:
            if (p - qrs[-1]) < 0.36*fs and pendiente < 0.5*pendiente_ant:
                es_qrs = False

        if es_qrs: #es QRS: actualizo el nivel de señal
            qrs.append(p)
            SPKI = 0.125*señal_i[p] + 0.875*SPKI
            pendiente_ant = pendiente
            descartados = []
        else: #es ruido: actualizo el nivel de ruido
            NPKI = 0.125*señal_i[p] + 0.875*NPKI
            descartados.append(p)
    return np.array(qrs), np.array(umbral)

def ajuste_r(señal_pb, qrs, fs, ventana=0.075):
    #el pico del integrado cae en el medio del QRS, no exactamente en el R
    #busco el R en la pasabanda, en una ventana de +-75 ms alrededor de cada detección
    w = int(ventana*fs)
    segmentos = [señal_pb[max(q-w, 0):q+w] for q in qrs]

    #decido UNA vez si en este canal el R apunta para arriba (+1) o para abajo (-1),
    #así no salta entre R y S de un latido a otro
    extremos = [s[np.argmax(np.abs(s))] for s in segmentos] #el valor más grande (en módulo) de cada latido
    signo = np.sign(np.median(extremos))

    picos_r = []
    for q, s in zip(qrs, segmentos):
        picos_r.append(max(q-w, 0) + np.argmax(signo*s)) #posición del R dentro de la señal completa
    return np.array(picos_r)

def detectar_qrs(señal, fs, refractario=0.3, ventana=0.15, devolver_etapas=False):
    #Pan-Tompkins completo: recibe UN canal (1D) ya preprocesado y devuelve las muestras de los R
    señal_pb = filtro_pasabanda(señal, fs)
    señal_d = derivada(señal_pb, fs)
    señal_c = cuadrado(señal_d)
    señal_i = integracion(señal_c, fs, ventana)
    candidatos = picos_candidatos(señal_i, fs, refractario)
    qrs, umbral = umbrales(señal_i, señal_d, candidatos, fs)
    picos_r = ajuste_r(señal_pb, qrs, fs)

    if devolver_etapas: #para graficar en el informe o revisar si algo falla
        etapas = {"pasabanda": señal_pb, "derivada": señal_d, "cuadrado": señal_c,
                  "integrada": señal_i, "candidatos": candidatos, "umbral": umbral, "qrs_integrada": qrs}
        return picos_r, etapas
    return picos_r
