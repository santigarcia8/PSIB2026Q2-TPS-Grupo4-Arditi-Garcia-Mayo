from pathlib import Path
from Módulos.preprocesamiento import eleccion_archivos, apertura_archivos, preprocesamiento, guardar_figura
from Módulos.pantompkins import detectar_qrs
from Módulos.filtro_adaptativo import referencia_materna, filtro_rls
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import wfdb
from Módulos.qrs_detect import qrs_detect
from Módulos.hrv_fetal import analisis_hrv
from Módulos.validacion import validar


def main():
    path, raiz = eleccion_archivos()
    if path is None:
        print("Saliendo del programa")
        return

    señal, canales, fs = apertura_archivos(path)
    nombre = Path(path).stem
    carpeta_res = raiz / "Resultados" / nombre #acá se guardan todas las figuras y métricas de este registro
    señal_filtrada = preprocesamiento(señal, canales, fs, nombre, raiz)


    #PAN-TOMPKINS: se usa SOLO para detectar el QRS materno
    x = señal_filtrada[:, 4] #abdomen 1
    picos_r = detectar_qrs(x, fs)
    print(f"QRS: {len(picos_r)}  FC materna media: {60*len(picos_r)/(len(x)/fs):.0f} lpm")
    n = int(10*fs)
    r = picos_r[picos_r < n]
    plt.figure(figsize=(12,4))
    plt.plot(x[:n], linewidth=0.5, color="darkgreen", label="Abdomen 1 preprocesada")
    plt.plot(r, x[r], "o", color="black", label="R materno")
    plt.title("Pan-Tompkins: R maternos")
    plt.legend(); plt.tight_layout()
    guardar_figura(carpeta_res / "QRS maternos", "qrs_maternos_pan_tompkins.png")


    b_h, a_h = butter(5, 8, btype="highpass", fs=fs)
    b_l, a_l = butter(5, 100, btype="lowpass", fs=fs)
    x = filtfilt(b_l, a_l, filtfilt(b_h, a_h, x))

    #Latidos fetales anotados del dataset (electrodo en la cabeza del feto): son la REFERENCIA
    #las anotaciones (rXX.edf.qrs) tienen que estar en data/raw/, junto al .edf, así la ruta es relativa al repo
    ruta_ann = Path(path + ".qrs")
    if not ruta_ann.exists():
        raise FileNotFoundError(f"Falta {ruta_ann.name} en {ruta_ann.parent}. Copialo desde la carpeta del dataset a data/raw/ (ver README).")
    ann = wfdb.rdann(path, "qrs")
    r_fetales = ann.sample
    print("latidos fetales anotados:", len(r_fetales), " largo de x:", len(x))

    #CANCELACIÓN DEL ECG MATERNO
    x_ref, es_qrs = referencia_materna(x, fs, q=1)
    f_est = filtro_rls(x, x_ref)

    #DETECCIÓN DE LATIDOS FETALES: solo con qrs_detect (Behar)
    r_det = qrs_detect(f_est, 0.15, 0.42, fs)
    

    #SERIE RR Y HRV FETAL (solo con lo detectado)
    analisis_hrv(r_det, fs, nombre, titulo="qrs_detect",
                 guardar=carpeta_res / "HRV fetal" / f"hrv_{nombre}_qrs_detect.png")

    #VALIDACIÓN contra la referencia (FECG directo): detección, exactitud de RR y HRV
    val = validar(r_det, r_fetales, fs, nombre, titulo="qrs_detect", carpeta=carpeta_res / "Validación")

    #aviso de dónde quedó todo
    n_fig = len(list(carpeta_res.rglob("*.png")))
    n_csv = len(list(carpeta_res.rglob("*.csv")))
    print(f"\nSe guardaron {n_fig} figuras y {n_csv} archivos de métricas (.csv) en: {carpeta_res}")


if __name__ == "__main__":
    main()