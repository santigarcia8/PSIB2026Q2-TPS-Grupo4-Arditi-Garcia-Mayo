import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch, find_peaks
import mne, wfdb



path = r"D:\abdominal-and-direct-fetal-ecg-database-1.0.0\r08.edf"

registro = mne.io.read_raw_edf(path, preload=True)

fs = registro.info["sfreq"]
señal = registro.get_data().T*1000 #lo paso a mV para no tener tanos problemas de escala
canales = registro.ch_names
anotaciones = wfdb.rdann(path, "qrs")
ref = anotaciones.sample


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

#Primero que nada detecto un pico ENORME después de 165 s por lo que ahí hay un artefacto predominante.
#Decido sacarlo antes de filtrar para evitar que se disperse por el resto del ECG ya que es muy puntual
#Voy a buscar el punto máximo y tomar como referencia 20 muestras antes y despues para interpolar
señal_sin_artefacto=señal.copy()
for i in range(señal.shape[1]):
    punto_max = np.argmax(señal[:,i])
    inicio, fin = punto_max-20, punto_max+20
    señal_sin_artefacto[inicio:fin+1, i] = np.linspace(señal[inicio-1,i], señal[fin+1,i], fin-inicio+3)[1:-1]

for i in range(señal_sin_artefacto.shape[1]):
    plt.figure()
    plt.plot(t, señal_sin_artefacto[:,i], color="magenta", linewidth=0.5)
    plt.suptitle(f"ECG sin artefacto del canal {canales[i]}")
    plt.grid(color="lightgray")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [mV]")
    plt.tight_layout()
    plt.show()

#sospecho ruido de baja frecuencia, hago PSD para confirmar. No debería haber ruido de línea porque fue
#eliminado en el sistema de detección según el dataset. Igual confirmo

for i in range(señal.shape[1]):
    freq, p = welch(señal_sin_artefacto[:,i], fs=fs, nperseg=2048, noverlap=2048//2)
    plt.figure()
    plt.semilogy(freq,p,color="purple", linewidth=0.5)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("PSD")
    plt.xlim(0,200)
    plt.title(f"PSD con Welch del canal {canales[i]}")
    plt.show()

#Se observa un pico predominante en bajas frecuencias de deriva
#Se observa frecuencias por encima de los 100 Hz así que aplico pasabandas en ese rango
#Solamente el canal de abdomen 3 tiene pico en 50 Hz, debe ser de error de filtrado del dataset, aplico Notch
#armo filtros

b, a = butter(5, [1,100], btype="bandpass", fs=fs) #usé orden 5 que es el de mayor orden antes de ser inestable, aplico 1 de corte porque en 0,5 mantenía todavía algo de deriva
señal_filtrada = filtfilt(b, a, señal_sin_artefacto, axis=0)

d, c = iirnotch(50, Q=30, fs=fs) #corto en 50, aplico Q=30 para hacer muy fina la banda de rechazo y lo aplico al canal abdomen 3
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
    
#FILTRO ADAPTATIVO - ANC
#Propongemos un filtro adaptativo que tome como referencia la señal del ECG materno y la elimine de los canales 

def nlms_multicanal(d, refs, L, mu, eps=1e-6): #d es la señal a limpiar, refs las referencia, L muestras pasadas, mu el paso 
    N, R = refs.shape
    w = np.zeros(L * R) #vector donde se guardan los pesos
    salida = np.zeros(N) #vector donde guardo la señal limpia
    for n in range(L, N): #voy recorriendo toda la señal
        xv = refs[n-L+1:n+1][::-1].ravel()   # toma las muestras de todos los canales, las ordena de más reciente a más vieja, y las pone en un vector juntas
        y = w @ xv                           # estimación del ECG materno con los pesos
        salida[n] = d[n] - y                 # lo que queda
        w += mu * salida[n] * xv / (eps + xv @ xv) #actualización de pesos, normalizada para que no dispare
    return salida

#Como necesito una referencia solo del latido matenro y no tengo, me la construyo

def referencia_materna(x, picos, fs, pre=0.2, post=0.4): #defino de dónde tomar la referencia (x), puntos donde están los latidos, y cuánto antes voy a recortar y cuanto después
    a, b = int(pre*fs), int(post*fs) #defino la cantidad de muestras que responden a esos pre y post
    picos = picos[(picos > a) & (picos < len(x) - b)] #no tengo en cuenta picos iniciales
    latidos = np.array([x[p-a:p+b] for p in picos]) #toma los fragmentos un poco antes y un poco después de los picos en la señal
    plantilla = latidos.mean(axis=0) #promeido para realzar el de la madre y atenuar fetal y ruido
    ref = np.zeros(len(x)) #donde voy a guardar los valores de la referencia final
    for p, lat in zip(picos, latidos):
        esc = (lat @ plantilla) / (plantilla @ plantilla)   # amplitud de este latido
        ref[p-a:p+b] = esc * plantilla
    return ref #devuelve la referencia

#Primero busco los latidos de a madre en abomen 1 que es de las que mejor se ve

x1 = señal_filtrada[:, 1]
umbral = 0.5 * np.percentile(np.abs(x1), 99.5) #hago que la señal sea toda positiva y busco un valor the threshold que solo el 0,5% supere y lo pongo a la mitad
picos, _ = find_peaks(np.abs(x1), height=umbral, distance=int(0.5 * fs)) #uso la función que cuenta los picos respecto al umbral, intentando que salgan con una distancia de 0,5fs para evitar tomar dos latidos juntos

#Armo la referencia de la madre con los picos

refs = np.column_stack([referencia_materna(señal_filtrada[:, c], picos, fs) for c in (1, 2, 3, 4)]) #tengo las referenias de cada canal por columna puestas

#Uso el filtro adaptativio con las referencias encontradas 
#uso mu 0.01 porque da pasos pequeños para que no haya problemas de convergencia, mu = 0.02 ya daba una señal muy atenuada y mala
# filtro los 4 canales para luego promediarlos y bajar varianza pero realzar fetal

salidas = np.column_stack([
    nlms_multicanal(señal_filtrada[:, c], refs, L=64, mu=0.01) for c in (1, 2, 3, 4)
])

promedio = salidas.mean(axis=1) #hago el promedio por el eje entre canales distintos

plt.figure(figsize=(16,4))
plt.subplot(2,1,1)
plt.plot(t, señal_filtrada[:,0], color = "navy", linewidth = 0.5)
plt.grid(color="lightgray")
plt.subplot(2,1,2)
plt.plot(t, promedio, color = "magenta", linewidth=0.5)
plt.grid(color="lightgray")
plt.show()

# Guardamos la señal fECG limpia, la frecuencia, el vector de tiempo Y los picos maternos
np.savez("fecg_limpio.npz", senal_fetal=promedio, picos_maternos=picos, fs=fs, t=t)
print("Archivo 'fecg_limpio.npz' guardado correctamente con picos_maternos.")