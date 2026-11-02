from pathlib import Path
from Módulos.preprocesamiento import eleccion_archivos, apertura_archivos, preprocesamiento
from Módulos.pantompkins import detectar_qrs
from Módulos.filtro_adaptativo import referencia_materna, filtro_rls
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import wfdb
from Módulos.qrs_detect import qrs_detect

def evaluar(det, ref, tol):
    det = np.sort(det); ref = np.sort(ref) #ordena la lista detectada y la de referencia
    usados = np.zeros(len(det), dtype=bool) #para evitar dobles detecciones
    TP = 0 #verdaderos positivos
    for r in ref: 
        d = np.abs(det - r).astype(float) #distancia entre lo detectado y el latido
        d[usados] = tol + 1    #si ya se detectó esa posición, le pone un valor más alto que el threshold para que no lo cuente              
        k = np.argmin(d) 
        if d[k] <= tol:
            TP += 1 #si entre la resta ve que es menor que el máxmo, entonces está bien detectdo
            usados[k] = True #lo marca pra no repetir
    FP = len(det) - TP #el largo de lo detectao y lo correcto
    FN = len(ref) - TP #el largo de lo que tendría que haber sido detectado y lo correcto
    #fórmulas del paper
    Se  = TP / (TP + FN)
    PPV = TP / (TP + FP)
    ACC = TP / (TP + FP + FN)
    F1  = 2*TP / (2*TP + FN + FP)
    return TP, FP, FN, Se, PPV, ACC, F1

def main():
    path, raiz = eleccion_archivos()
    if path is None:
        print("Saliendo del programa")
        return

    señal, canales, fs = apertura_archivos(path)
    nombre = Path(path).stem
    señal_filtrada = preprocesamiento(señal, canales, fs, nombre, raiz)
    
    
    x = señal_filtrada[:, 4] #abdomen 1
    picos_r = detectar_qrs(x, fs)
    print(f"QRS: {len(picos_r)}  FC materna media: {60*len(picos_r)/(len(x)/fs):.0f} lpm")
    n = int(10*fs)
    r = picos_r[picos_r < n]
    plt.figure(figsize=(12,4))
    plt.plot(x[:n], linewidth=0.5, color="darkgreen", label="Abdomen 1 preprocesada")
    plt.plot(r, x[r], "o", color="black", label="R materno")
    plt.title("Pan-Tompkins: R maternos")
    plt.legend(); plt.tight_layout(); plt.show()
    
    
    b_h, a_h = butter(5, 8, btype="highpass", fs=fs)
    b_l, a_l = butter(5, 100, btype="lowpass", fs=fs)
    x = filtfilt(b_l, a_l, filtfilt(b_h, a_h, x))
    # Latidos fetales anotados del dataset
    ann = wfdb.rdann("D:/abdominal-and-direct-fetal-ecg-database-1.0.0/r01.edf", "qrs")
    r_fetales = ann.sample
    print("latidos fetales anotados:", len(r_fetales), " largo de x:", len(x))

    x_ref, es_qrs = referencia_materna(x, fs, q=1, incluir_actual=True)
    f_est = filtro_rls(x, x_ref)

    #oli
    r_det = detectar_qrs(f_est, fs, refractario=0.15, ventana=0.05)
    TP, FP, FN, Se, PPV, ACC, F1 = evaluar(r_det, r_fetales, int(0.05 * fs))
    print(f"P&T compañera: TP={TP} FP={FP} FN={FN} Se={Se:.3f} PPV={PPV:.3f} F1={F1:.3f}")

    #Behar
    r_det = qrs_detect(f_est, 0.15, 0.42, fs)
    TP, FP, FN, Se, PPV, ACC, F1 = evaluar(r_det, r_fetales, int(0.05 * fs))
    print(f"qrs_detect Behar: TP={TP} FP={FP} FN={FN} Se={Se:.3f} PPV={PPV:.3f} F1={F1:.3f}")


if __name__ == "__main__":
    main()