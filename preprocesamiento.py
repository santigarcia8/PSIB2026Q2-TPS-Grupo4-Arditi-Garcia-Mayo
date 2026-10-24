import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch
import mne, wfdb



path = r"D:\abdominal-and-direct-fetal-ecg-database-1.0.0\r08.edf"

registro = mne.io.read_raw_edf(path, preload=True)

fs = registro.info["sfreq"]
señal = registro.get_data().T*1000 #lo paso a mV para no tener tanos problemas de escala
canales = registro.ch_names
anotaciones = wfdb.rdann(path, "qrs")
ref = anotaciones.sample


t=np.arange(0,len(señal[:,1]))/fs #defino e tiempo que duran los registros
# for i in range(señal.shape[1]):
#     plt.figure()
#     plt.plot(t, señal[:,i], color="darkgreen", linewidth=0.5)
#     plt.suptitle(f"ECG del canal {canales[i]}")
#     plt.grid(color="lightgray")
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Amplitud [mV]")
#     plt.tight_layout()
#     plt.show()

#Primero que nada detecto un pico ENORME después de 165 s por lo que ahí hay un artefacto predominante.
#Decido sacarlo antes de filtrar para evitar que se disperse por el resto del ECG ya que es muy puntual
#Voy a buscar el punto máximo y tomar como referencia 20 muestras antes y despues para interpolar
señal_sin_artefacto=señal.copy()
for i in range(señal.shape[1]):
    punto_max = np.argmax(señal[:,i])
    inicio, fin = punto_max-20, punto_max+20
    señal_sin_artefacto[inicio:fin+1, i] = np.linspace(señal[inicio-1,i], señal[fin+1,i], fin-inicio+3)[1:-1]

# for i in range(señal_sin_artefacto.shape[1]):
#     plt.figure()
#     plt.plot(t, señal_sin_artefacto[:,i], color="magenta", linewidth=0.5)
#     plt.suptitle(f"ECG sin artefacto del canal {canales[i]}")
#     plt.grid(color="lightgray")
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Amplitud [mV]")
#     plt.tight_layout()
#     plt.show()

#sospecho ruido de baja frecuencia, hago PSD para confirmar. No debería haber ruido de línea porque fue
#eliminado en el sistema de detección según el dataset. Igual confirmo

# for i in range(señal.shape[1]):
#     freq, p = welch(señal_sin_artefacto[:,i], fs=fs, nperseg=2048, noverlap=2048//2)
#     plt.figure()
#     plt.semilogy(freq,p,color="purple", linewidth=0.5)
#     plt.xlabel("Frecuencia [Hz]")
#     plt.ylabel("PSD")
#     plt.xlim(0,200)
#     plt.title(f"PSD con Welch del canal {canales[i]}")
#     plt.show()

#Se observa un pico predominante en bajas frecuencias de deriva
#Se observa frecuencias por encima de los 100 Hz así que aplico pasabandas en ese rango
#Solamente el canal de abdomen 3 tiene pico en 50 Hz, debe ser de error de filtrado del dataset, aplico Notch
#armo filtros

b, a = butter(5, [1,100], btype="bandpass", fs=fs) #usé orden 5 que es el de mayor orden antes de ser inestable, aplico 1 de corte porque en 0,5 mantenía todavía algo de deriva
señal_filtrada = filtfilt(b, a, señal_sin_artefacto, axis=0)

d, c = iirnotch(50, Q=30, fs=fs) #corto en 50, aplico Q=30 para hacer muy fina la banda de rechazo y lo aplico al canal abdomen 3
señal_filtrada[:,3]=filtfilt(d,c, señal_filtrada[:,3])


# for i in range(señal.shape[1]):
#     plt.figure()
#     plt.plot(t, señal_filtrada[:,i], color="darkgreen", linewidth=0.5)
#     plt.suptitle(f"ECG del canal {canales[i]}")
#     plt.grid(color="lightgray")
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Amplitud [mV]")
#     plt.show()

#ploteo comparativamente
# for i in range(señal.shape[1]):
#     plt.figure()
#     plt.plot(t, señal_filtrada[:,i], color="darkgreen", linewidth=0.5, label="Filtrada")
#     plt.plot(t, señal[:,i], color="orange", linewidth=0.5, label="Original", linestyle="--")
#     plt.suptitle(f"ECG del canal {canales[i]}")
#     plt.grid(color="lightgray")
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Amplitud [mV]")
#     plt.legend()
#     plt.show()

# for i in range(señal.shape[1]):
#     freq, p = welch(señal_filtrada[:,i], fs=fs, nperseg=2048, noverlap=2048//2)
#     plt.figure()
#     plt.semilogy(freq,p,color="purple", linewidth=0.5)
#     plt.xlabel("Frecuencia [Hz]")
#     plt.ylabel("PSD")
#     plt.xlim(0,200)
#     plt.title(f"PSD con Welch del canal {canales[i]} filtrado")
#     plt.show()
    
#FILTRO ADAPTATIVO - ANC
#Primero defino qué canal del ECG voy a usar como referencia. Para eso tengo que usar el que menos información contenga del FECG
#En base a Direct_1 voy a buscar los QRS del bebé, luego ver qué ECG tiene más de esa info, y ahí ver cuál uso como referencia del ANC

#Pan Tompkins
f, e = butter(4, [12,42], btype="bandpass", fs = fs)
fetal_filtrado_banda = filtfilt(f, e, señal_filtrada[:,0])

#Derivativo
fetal_filtrado_derivativo = np.convolve(fetal_filtrado_banda, np.array([1, 2, 0, -2, -1]) / 8 * fs, mode="same") #uso el original para no desfasar

#Elevo al cuadrado
fetal_cuadrado = fetal_filtrado_derivativo**2

#Media móvil
M =  31  # 30 ms; impar para que la ventana quede centrada
fetal_integrado = np.convolve(fetal_cuadrado, np.ones(M) / M, mode="same")
altura_tipica = np.percentile(fetal_integrado, 99.5)
fraccion = 0.35          # parámetro a barrer
threshold = fraccion * altura_tipica
picos = []
i = 0
while i < len(fetal_integrado):
    if fetal_integrado[i] > threshold:
        inicio = i
        while i < len(fetal_integrado) and fetal_integrado[i] > threshold:
            i += 1                      # avanza hasta que la señal baja del umbral
        picos.append(inicio + np.argmax(fetal_integrado[inicio:i]))   # el máximo de ese grupo
    else:
        i += 1
#Una vez que detecté QRS, voy a guardar los R
w = int(0.03 * fs)    # 30 ms
R = []
for p in picos:
    a = max(p - w, 0)
    b = min(p + w + 1, len(fetal_filtrado_banda))
    R.append(a + np.argmax(np.abs(fetal_filtrado_banda[a:b])))
R = np.array(R)

ventana = int(0.15 * fs)                         # ±150 ms alrededor de cada R
R_limpios = R[(R > 5*fs) & (R < 40*fs)]          # tramo limpio, sin artefactos largos
t_v = np.arange(-ventana, ventana + 1) / fs * 1000   # eje en ms

# plt.figure(figsize=(12, 6))
# for c in range(1, 5):                            # canales 1..4 = abdomen
#     latidos = np.array([señal_filtrada[r - ventana : r + ventana + 1, c] for r in R_limpios])
#     promedio = latidos.mean(axis=0)
#     print(canales[c], "amplitud pico a pico del promedio:", np.ptp(promedio))
#     plt.plot(t_v, promedio, label=canales[c])
# plt.xlabel("Tiempo respecto del R [ms]")
# plt.legend()
# plt.show()

def nlms(d, ref, L, mu, eps=1e-6):
    w = np.zeros(L)
    salida = np.zeros(len(d))
    for n in range(L, len(d)):
        xv = ref[n-L+1:n+1][::-1]
        y = w @ xv
        salida[n] = d[n] - y
        w += mu * salida[n] * xv / (eps + xv @ xv)
    return salida

fetal_anc = nlms(señal_filtrada[:, 4], señal_filtrada[:, 1], L=256, mu=0.003)

t = np.arange(len(fetal_anc)) / fs
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1); plt.plot(t, señal_filtrada[:, 4]); plt.title("Abdomen_4 antes"); plt.xlim(10, 15)
plt.subplot(2, 1, 2); plt.plot(t, fetal_anc);            plt.title("Abdomen_4 después del filtro"); plt.xlim(10, 15)
plt.tight_layout(); plt.show()