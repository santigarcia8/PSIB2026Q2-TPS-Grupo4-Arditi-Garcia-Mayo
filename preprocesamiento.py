import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch
import mne


path = r"D:\abdominal-and-direct-fetal-ecg-database-1.0.0\r04.edf"

registro = mne.io.read_raw_edf(path, preload=True)

fs = registro.info["sfreq"]
señal = registro.get_data().T*1000 #lo paso a mV para no tener tanos problemas de escala
canales = registro.ch_names


t=np.arange(0,len(señal[:,1]))/fs #defino e tiempo que duran los registros
for i in range(señal.shape[1]):
    plt.figure()
    plt.plot(t, señal[:,i], color="darkgreen", linewidth=0.5)
    plt.suptitle(f"ECG del canal {canales[i]}")
    plt.grid(color="lightgray")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [mV]")
    plt.tight_layout()
    plt.show()

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
    plt.show()

#Se observa un pico predominante en bajas frecuencias de deriva, no hay picos en 50 ni 60 Hz así que no hay ruido de línea.
#Se observa frecuencias por encima de los 150 Hz así que aplico pasabandas en ese rango
#armo filtros

b, a = butter(5, [0.5,100], btype="bandpass", fs=fs)
señal_filtrada = filtfilt(b, a, señal, axis=0)

#como el canal abomen 1 y abdomen 3 tienen picos en 50 Hz, asumo que no filtraron todos los canales
d, c = iirnotch(50, Q=30, fs=fs)
señal_filtrada[:,1]=filtfilt(d,c, señal_filtrada[:,1])
señal_filtrada[:,3]=filtfilt(d,c, señal_filtrada[:,3])


for i in range(señal.shape[1]):
    plt.figure()
    plt.plot(t, señal_filtrada[:,i], color="darkgreen", linewidth=0.5)
    plt.suptitle(f"ECG del canal {canales[i]}")
    plt.grid(color="lightgray")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [mV]")
    plt.show()

#ploteo comparativamente
for i in range(señal.shape[1]):
    plt.figure()
    plt.plot(t, señal_filtrada[:,i], color="darkgreen", linewidth=0.5, label="Filtrada")
    plt.plot(t, señal[:,i], color="orange", linewidth=0.5, label="Original", linestyle="--")
    plt.suptitle(f"ECG del canal {canales[i]}")
    plt.grid(color="lightgray")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [mV]")
    plt.legend()
    plt.show()

for i in range(señal.shape[1]):
    freq, p = welch(señal_filtrada[:,i], fs=fs, nperseg=2048, noverlap=2048//2)
    plt.figure()
    plt.semilogy(freq,p,color="purple", linewidth=0.5)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD")
    plt.xlim(0,200)
    plt.title(f"PSD con Welch del canal {canales[i]} filtrado")
    plt.show()