import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 0. DETECCIÓN DE QRS FETAL (desde pantompkins.py)
# ==============================================================================
# Al importar pantompkins se ejecuta la detección completa sobre fecg_limpio.npz
# (hay que correr antes preprocesamiento.py). 
import pantompkins as pt

fs = pt.fs
picos_det = np.asarray(pt.picos_fetales_det)  # QRS fetales detectados en la señal abdominal
picos_ref = np.asarray(pt.picos_fetal_ref)    # QRS fetales de referencia (electrodo de cuero cabelludo)

# ==============================================================================
# 1. EXTRACCIÓN DE LAS SERIES RR
# ==============================================================================
# RR_k = t_k - t_(k-1): cada intervalo se asocia al tiempo del latido que lo cierra

def series_rr(picos, fs):
    t_latidos = picos / fs           # tiempos de ocurrencia de los latidos [s]
    rr = np.diff(t_latidos)          # intervalos RR [s]
    t_rr = t_latidos[1:]             # instante asociado a cada intervalo [s]
    return t_rr, rr

t_rr_det, rr_det = series_rr(picos_det, fs)
t_rr_ref, rr_ref = series_rr(picos_ref, fs)

def resumen(nombre, rr):
    fc = 60 / rr  # frecuencia cardíaca instantánea [lpm]
    print(f"--- {nombre} ---")
    print(f"Cantidad de intervalos RR: {len(rr)}")
    print(f"RR medio: {rr.mean()*1000:.1f} ms | desvío: {rr.std()*1000:.1f} ms")
    print(f"RR mínimo: {rr.min()*1000:.1f} ms | RR máximo: {rr.max()*1000:.1f} ms")
    print(f"FC media: {fc.mean():.1f} lpm (rango {fc.min():.1f} - {fc.max():.1f} lpm)")

resumen("Detectados (abdominal)", rr_det)
resumen("Referencia (directo)", rr_ref)

# ==============================================================================
# 2. VISUALIZACIÓN DE LAS SERIES RR (antes de buscar ectópicos)
# ==============================================================================
plt.figure(figsize=(14, 4))
plt.plot(t_rr_ref, rr_ref * 1000, "o-", color="darkgreen", markersize=3, linewidth=0.7, label="Referencia (directo)")
plt.plot(t_rr_det, rr_det * 1000, "o-", color="magenta", markersize=3, linewidth=0.7, label="Detectados (abdominal)")
plt.title("Series RR fetales en función del tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("RR [ms]")
plt.legend(loc="upper right")
plt.grid(color="lightgray")
plt.tight_layout()
plt.show()

# ==============================================================================
# 3. DETECCIÓN DE LATIDOS ECTÓPICOS
# ==============================================================================
# Se usan los dos criterios vistos en clase:
#  a) Morfología -> análisis de correlación con un template (ectópicos ventriculares)
#  b) Tiempo de aparición -> latido prematuro + pausa compensatoria (supraventriculares)

senal = pt.promedio            # fECG residual (salida del NLMS)
picos_m = pt.picos_m           # QRS maternos
w = int(0.05 * fs)             # ventana de ±50 ms alrededor de cada QRS fetal
max_corr = int(0.01 * fs)      # corrimiento máximo para alinear (±10 ms)

# Solo latidos con ventana completa
picos = picos_det[(picos_det > w + max_corr) & (picos_det < len(senal) - w - max_corr)]

# a.1) Template: promedio de 6 latidos sinusales elegidos a mano (verificar visualmente)
idx_template = range(10, 16)
template = np.mean([senal[picos[i]-w:picos[i]+w] for i in idx_template], axis=0)

# a.2) Alinear cada latido con el template (máximo de la correlación) y calcular rho
def correlacion(seg, tpl):
    a = seg - seg.mean()
    b = tpl - tpl.mean()
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

rho = np.zeros(len(picos))
for k, p in enumerate(picos):
    rho[k] = max(correlacion(senal[p-w+s:p+w+s], template) for s in range(-max_corr, max_corr + 1))

umbral_rho = 0.5
sospechosos = np.where(rho < umbral_rho)[0]

# b) Tiempo de aparición: RR anterior y posterior relativos a la mediana local (±5 RR)
rr = np.diff(picos) / fs
mediana_local = np.array([np.median(rr[max(0, i-5):i+6]) for i in range(len(rr))])

print(f"\nCorrelación con el template: mediana = {np.median(rho):.2f}")
print(f"Latidos con rho < {umbral_rho}: {len(sospechosos)}")
print("  t [s]   rho   RRprev/med  RRpost/med  dist. QRS materno [ms]")
for k in sospechosos:
    pre = rr[k-1] / mediana_local[k-1] if k > 0 else np.nan
    post = rr[k] / mediana_local[k] if k < len(rr) else np.nan
    d_mat = np.min(np.abs(picos_m - picos[k])) / fs * 1000
    print(f"{picos[k]/fs:7.1f}  {rho[k]:.2f}     {pre:.2f}        {post:.2f}        {d_mat:.0f}")

# Un ectópico sería prematuro (RRprev/med < 0.8) y seguido de pausa (RRpost/med > 1.2)
prematuros = [k for k in range(1, len(rr)) if rr[k-1]/mediana_local[k-1] < 0.8 and rr[k]/mediana_local[k] > 1.2]
print(f"Latidos prematuros con pausa compensatoria: {len(prematuros)}")

plt.figure(figsize=(14, 4))
plt.plot(picos / fs, rho, "o", color="purple", markersize=3)
plt.axhline(umbral_rho, color="red", linestyle="--", label=f"Umbral rho = {umbral_rho}")
plt.title("Coeficiente de correlación de cada QRS fetal con el template")
plt.xlabel("Tiempo [s]")
plt.ylabel("rho")
plt.legend(loc="lower right")
plt.grid(color="lightgray")
plt.tight_layout()
plt.show()

# ==============================================================================
# 4. VERIFICACIÓN DE LOS LATIDOS CON BAJA CORRELACIÓN
# ==============================================================================
# a) ¿La baja correlación depende de la superposición con el QRS materno?
#    Se compara rho de los latidos cercanos (< 100 ms) y lejanos (>= 100 ms) a un QRS materno
dist_materno = np.array([np.min(np.abs(picos_m - p)) / fs * 1000 for p in picos])  # [ms]
lejos = dist_materno >= 100

print(f"\nLatidos lejos de un QRS materno (>= 100 ms): {lejos.sum()}")
print(f"  rho mediana = {np.median(rho[lejos]):.2f} | latidos con rho < {umbral_rho}: {(rho[lejos] < umbral_rho).sum()}")
print(f"Latidos cerca de un QRS materno (< 100 ms): {(~lejos).sum()}")
print(f"  rho mediana = {np.median(rho[~lejos]):.2f} | latidos con rho < {umbral_rho}: {(rho[~lejos] < umbral_rho).sum()}")

# b) Validación con el ECG directo (electrodo de cuero cabelludo, canal Direct_1):
#    misma correlación con template, pero sobre la señal directa y sus picos de referencia
import mne
from scipy.signal import butter, filtfilt

registro = mne.io.read_raw_edf(pt.path_record_anotaciones, preload=True, verbose=False)
directo = registro.get_data()[0] * 1000                   # canal Direct_1 en mV
b, a = butter(5, [1, 100], btype="bandpass", fs=fs)       # mismo pasabanda que preprocesamiento.py
directo = filtfilt(b, a, directo)

picos_dir = picos_ref[(picos_ref > w + max_corr) & (picos_ref < len(directo) - w - max_corr)]
template_dir = np.mean([directo[picos_dir[i]-w:picos_dir[i]+w] for i in idx_template], axis=0)

rho_dir = np.zeros(len(picos_dir))
for k, p in enumerate(picos_dir):
    rho_dir[k] = max(correlacion(directo[p-w+s:p+w+s], template_dir) for s in range(-max_corr, max_corr + 1))

print(f"\nECG directo: rho mediana = {np.median(rho_dir):.2f}")
print("Latidos con rho bajo en la abdominal, vistos en el directo:")
print("  t [s]   rho abdominal   rho directo   corrimiento vs. referencia [ms]")
for k in sospechosos:
    j = np.argmin(np.abs(picos_dir - picos[k]))       # latido de referencia más cercano
    corrimiento = abs(picos_dir[j] - picos[k]) / fs * 1000
    print(f"{picos[k]/fs:7.1f}     {rho[k]:.2f}           {rho_dir[j]:.2f}          {corrimiento:.0f}")
    
# ==============================================================================
# 5. SERIE NN Y TACOGRAMA
# ==============================================================================
# No se identificaron latidos ectópicos (secciones 3 y 4), por lo que la serie NN
# es la serie RR detectada sin los intervalos con subdetección (latido perdido):
# RR ≈ 2 veces la mediana local (cociente entre 1.9 y 2.1)
mediana_rr = np.array([np.median(rr_det[max(0, i-5):i+6]) for i in range(len(rr_det))])
cociente = rr_det / mediana_rr
subdeteccion = (cociente > 1.9) & (cociente < 2.1)

nn = rr_det[~subdeteccion]       # serie NN [s]
t_nn = t_rr_det[~subdeteccion]   # instante de cada intervalo NN [s] (no equiespaciado)

print(f"\nIntervalos excluidos por subdetección: {subdeteccion.sum()}")
print(f"Serie NN: {len(nn)} intervalos | media = {nn.mean()*1000:.1f} ms | desvío = {nn.std()*1000:.1f} ms")

# Tacograma: intervalos NN consecutivos en función del número de latido
plt.figure(figsize=(14, 4))
plt.plot(np.arange(len(nn)), nn * 1000, "o-", color="darkgreen", markersize=3, linewidth=0.7)
plt.title("Tacograma fetal (serie NN)")
plt.xlabel("Número de latido")
plt.ylabel("NN [ms]")
plt.grid(color="lightgray")
plt.tight_layout()
plt.show()

# ==============================================================================
# 6. FUNCIÓN DE INTERVALO
# ==============================================================================
# Se le agrega temporalidad al tacograma: cada intervalo NN se ubica como un impulso
# en el instante t_k del latido que lo cierra -> dIF(t) = sum (t_k - t_(k-1)) * delta(t - t_k)
# El eje x pasa a estar en segundos (base temporal no equiespaciada), lo que permite
# encarar el análisis espectral.
plt.figure(figsize=(14, 4))
marcas, lineas, base = plt.stem(t_nn, nn * 1000, basefmt=" ")
plt.setp(marcas, color="darkgreen", markersize=2)
plt.setp(lineas, color="darkgreen", linewidth=0.5)
plt.ylim(nn.min() * 1000 - 50, nn.max() * 1000 + 50)
plt.title("Función de intervalo fetal (serie NN)")
plt.xlabel("Tiempo [s]")
plt.ylabel("NN [ms]")
plt.grid(color="lightgray")
plt.tight_layout()
plt.show()

# ==============================================================================
# 7. PERIODOGRAMA DE LOMB
# ==============================================================================
# Para cada frecuencia se ajusta por cuadrados mínimos s(t_k) = a1*cos(W t_k) + a2*sin(W t_k)
# a la serie d(t_k) y el espectro es la proyección de los datos sobre ese ajuste:
#   a = (H^T H)^-1 H^T d      S(W) = d^T H a / (M+1)
# Trabaja directamente con los tiempos de los latidos (no requiere muestras equiespaciadas).

def periodograma_lomb(t, d, frecuencias):
    d = d - d.mean()                     # se quita la media (componente de continua)
    S = np.zeros(len(frecuencias))
    for i, f in enumerate(frecuencias):
        W = 2 * np.pi * f
        H = np.column_stack([np.cos(W * t), np.sin(W * t)])
        a = np.linalg.solve(H.T @ H, H.T @ d)
        S[i] = (d @ H @ a) / len(d)
    return S

frecuencias = np.linspace(0.01, 1.0, 1000)   # [Hz]
S_det = periodograma_lomb(t_nn, nn * 1000, frecuencias)        # NN detectada [ms^2]
S_ref = periodograma_lomb(t_rr_ref, rr_ref * 1000, frecuencias) # referencia (sin ectópicos) [ms^2]

# Potencia por bandas fetales:
#   LF 0.03-0.15 Hz | MF 0.15-0.5 Hz | HF 0.5-1 Hz (asociada a movimientos respiratorios fetales)
bandas = {"LF": (0.03, 0.15), "MF": (0.15, 0.5), "HF": (0.5, 1.0)}

def potencia_banda(S, f, f_min, f_max):
    banda = (f >= f_min) & (f <= f_max)
    return np.sum(S[banda]) * (f[1] - f[0])   # integración del espectro en la banda

for nombre, S in [("Detectada (abdominal)", S_det), ("Referencia (directo)", S_ref)]:
    p = {b: potencia_banda(S, frecuencias, *lim) for b, lim in bandas.items()}
    print(f"--- {nombre} --- LF = {p['LF']:.2f} | MF = {p['MF']:.2f} | HF = {p['HF']:.2f} | LF/(MF+HF) = {p['LF']/(p['MF']+p['HF']):.2f}")

plt.figure(figsize=(14, 4))
plt.plot(frecuencias, S_ref, color="darkgreen", linewidth=0.8, label="Referencia (directo)")
plt.plot(frecuencias, S_det, color="magenta", linewidth=0.8, label="Detectada (abdominal)")
for (b, (f1, f2)), color in zip(bandas.items(), ["lightblue", "lightyellow", "mistyrose"]):
    plt.axvspan(f1, f2, color=color, alpha=0.5, label=b)
plt.title("Periodograma de Lomb de la serie NN fetal")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("S(Ω) [ms²]")
plt.legend(loc="upper right")
plt.grid(color="lightgray")
plt.tight_layout()
plt.show()