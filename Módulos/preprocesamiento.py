import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch
import mne, wfdb
from pathlib import Path

def eleccion_archivos():
    path = None
    raiz = Path(__file__).resolve().parent.parent
    carpeta = raiz/ "Archivos"

    registros_disponibles = sorted(carpeta.glob("*.edf"))
    print("Registros disponibles:")
    for i, r in enumerate(registros_disponibles):
        print(f" [{i+1}] {r.name}")

    loop = True
    while loop:
        try:
            print("Si desea salir del programa, elija 0")
            opcion = int(input("Ingrese la opción a procesar: "))
            if opcion == 1:
                path = str(registros_disponibles[opcion-1])
                loop = False
            elif opcion == 2:
                path = str(registros_disponibles[opcion-1])
                loop = False
            elif opcion == 3:
                path = str(registros_disponibles[opcion-1])
                loop = False
            elif opcion == 4: 
                path = str(registros_disponibles[opcion-1])
                loop = False
            elif opcion == 0:
                loop = False
            else:
                print("Por favor ingrese un valor correcto")
        except ValueError:
            print("Por favor ingrese un valor correcto")
    
    return path, raiz

def apertura_archivos(path):
    registro = mne.io.read_raw_edf(path, preload=True)

    fs = registro.info["sfreq"]
    señal = registro.get_data().T*1000 #lo paso a mV para no tener tanos problemas de escala
    canales = registro.ch_names
    
    return señal, canales, fs

def preprocesamiento(señal, canales, fs, nombre, raiz):
    t=np.arange(0,len(señal[:,1]))/fs #defino e tiempo que duran los registros
    for i in range(señal.shape[1]):
        plt.figure()
        plt.plot(t, señal[:,i], color="darkgreen", linewidth=0.5)
        plt.suptitle(f"ECG del canal {canales[i]}")
        plt.grid(color="lightgray")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud [mV]")
        plt.tight_layout()
        # plt.savefig(raiz / "Resultados" / nombre / "ECG original" / f"ecg_original_{canales[i]}.png", dpi=300)


    #sospecho ruido de baja frecuencia, hago PSD para confirmar. No debería haber ruido de línea porque fue
    #eliminado en el sistema de detección según el dataset. Igual confirmo

    for i in range(señal.shape[1]):
        freq, p = welch(señal[:,i], fs=fs, nperseg=2048, noverlap=2048//2)
        plt.figure()
        plt.semilogy(freq,p,color="purple", linewidth=0.5)
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("PSD")
        plt.xlim(0,200)
        plt.title(f"PSD con Welch del canal {canales[i]}")
        # plt.savefig(raiz / "Resultados" / nombre / "Periodograma crudo" / f"periodograma_crudo_{canales[i]}.png", dpi=300)
        
    #El ruido de baja y alta frecuencia está presente en todas así que armo el butterworth y filtro para todos

    b, a = butter(5, [1,100], btype="bandpass", fs=fs) #usé orden 5 que es el de mayor orden antes de ser inestable, aplico 1 de corte porque en 0,5 mantenía todavía algo de deriva
    señal_filtrada = filtfilt(b, a, señal, axis=0)

    #El ruido de línea depende de qué persona y canal estoy así que hago un if ponerlo correctamente
    d, c = iirnotch(50, Q=30, fs=fs) #corto en 50, aplico Q=30 para hacer muy fina la banda de rechazo y lo aplico al canal abdomen 3
    if nombre == "r01" or nombre == "r10":
        señal_filtrada[:,3] = filtfilt(d, c, señal_filtrada[:,3])
    elif nombre == "r04" or nombre == "r07":
        señal_filtrada[:,1] = filtfilt(d, c, señal_filtrada[:,1])
        señal_filtrada[:,3] = filtfilt(d, c, señal_filtrada[:,3])
        

    for i in range(señal.shape[1]):
        plt.figure()
        plt.plot(t, señal_filtrada[:,i], color="darkgreen", linewidth=0.5)
        plt.suptitle(f"ECG post filtrado del canal {canales[i]}")
        plt.grid(color="lightgray")
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Amplitud [mV]")
        # plt.savefig(raiz / "Resultados" / nombre / "ECG filtrado" / f"ecg_filtrado_{canales[i]}.png", dpi=300)

    for i in range(señal.shape[1]):
        freq, p = welch(señal_filtrada[:,i], fs=fs, nperseg=2048, noverlap=2048//2)
        plt.figure()
        plt.semilogy(freq,p,color="purple", linewidth=0.5)
        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("PSD")
        plt.xlim(0,200)
        plt.title(f"PSD con Welch del canal {canales[i]} filtrado")
        # plt.savefig(raiz / "Resultados" / nombre / "Periodograma filtrado" / f"periodograma_filtrado_{canales[i]}.png", dpi=300)
    
    return señal_filtrada
