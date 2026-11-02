import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from numpy.lib.stride_tricks import sliding_window_view


def qrs_detect(ecg, ref_period=0.250, thres=0.6, fs=1000):
    """Traducción a Python de qrs_detect.m (J. Behar)."""
    fs=int(round(fs)) #lo pone en int porque el archivo dice 1000.0
    ecg = np.asarray(ecg, dtype=float)#lo nrmaliza a valores entre 0 y 1
    n_samp = len(ecg) 

    med_n = int(fs / 100) #largo del filtro de mediana
    int_n = int(round(7 * fs / 256)) #largo de la ventana de integración
    low_cut, high_cut = 5, 45 #cortes del "pasabanda"

    b_lp, a_lp = butter(5, high_cut, btype="high", fs=fs)
    b_bas, a_bas = butter(2, low_cut, btype="high", fs=fs)
    ecg = ecg - np.mean(ecg) #le resta la media para que quede centrado
    bpf = ecg - filtfilt(b_lp, a_lp, ecg)
    bpf = filtfilt(b_bas, a_bas, bpf)

    dff = np.diff(bpf) 
    sqr = dff * dff
    integ = lfilter(np.ones(int_n), 1, sqr) 
    xp = np.pad(integ, (med_n // 2, med_n // 2 - 1)) #rellena con 0 para no tener discontinuidades
    mdf = np.median(sliding_window_view(xp, med_n), axis=1) #filtro media móvil
    delay = int(np.ceil(int_n / 2)) 
    mdf = np.roll(mdf, -delay)# compenso el retraso del filtrado

    if n_samp / fs > 90:
        xs = np.sort(mdf[fs - 1: fs * 90])
    else:
        xs = np.sort(mdf[fs - 1:])
    pct = 98 if n_samp / fs > 10 else 99 # en caso de que sea de más de 10 segundos, voy a separar con percentil 98. sino 99
    ind = int(np.ceil(pct / 100 * len(xs))) #me fijo pnto donde está ese percentil en la lista ordenada
    en_thres = xs[ind - 1] #cuanto vale la señal en ese percentil

    poss = mdf > thres * en_thres #me fijo las posiciones donde se supere el umbral

    idx = np.where(poss)[0] #posiciones que superaron el umbral
    if len(idx) > 1: 
        rr = np.diff(idx) / fs
        med_rr = np.median(rr[rr > 0.01]) #hace la mediana de la separación entre segundos RR
        for k in np.where(rr > 1.5 * med_rr)[0]: #se fija que si hay más de 1,5 veces la mediana entre picos, seguro se salteó uno
            ini, fin = idx[k], idx[k + 1]
            poss[ini:fin + 1] = mdf[ini:fin + 1] > 0.5 * thres * en_thres #reduce el umbral para ver si hay algún pico más pequeño

    p = poss.astype(int) #rellena con 0 para que en la diferenciación no se pierdan datos en los bordes
    left = np.where(np.diff(np.concatenate(([0], p))) == 1)[0]
    right = np.where(np.diff(np.concatenate((p, [0]))) == -1)[0]

    loc = np.array([l + np.argmax(np.abs(bpf[l:r + 1])) for l, r in zip(left, right)])
    signo = np.mean(ecg[loc])

    picos, vals = [], [] #donde guarda dónde está el pico y su altura
    for l, r in zip(left, right):
        seg = ecg[l:r + 1] #se fija cómo es la señal en ese tramo donde hay un pico
        k = np.argmax(seg) if signo > 0 else np.argmin(seg) #guarda según si es positio o negativo
        pos, val = l + k, seg[k] #guarda el valor 
        if picos:
            dist = pos - picos[-1]
            if dist < fs * ref_period and abs(val) < abs(vals[-1]): #si el pico está a menos de X muestras y es menor, se descarta
                continue
            elif dist < fs * ref_period and abs(val) > abs(vals[-1]): #si está a menos de X muestras y es mayor, se deja
                picos[-1], vals[-1] = pos, val
                continue
        picos.append(pos)
        vals.append(val)
    return np.array(picos)