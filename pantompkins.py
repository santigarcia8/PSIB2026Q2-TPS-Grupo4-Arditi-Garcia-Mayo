import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import wfdb

# ==============================================================================
# 1. CARGA DE DATOS Y ANOTACIONES DE REFERENCIA
# ==============================================================================
# Cargar la señal fECG residual y picos maternos guardados por preprocesamiento.py
datos = np.load("fecg_limpio.npz")
promedio = datos["senal_fetal"]
picos_m = datos["picos_maternos"]
fs = float(datos["fs"])
t = datos["t"]

# Cargar las anotaciones de referencia de PhysioNet (.qrs)
path_record_anotaciones = r"C:\Users\agust\Downloads\r08.edf"
anotaciones_ref = wfdb.rdann(path_record_anotaciones, "qrs")
picos_fetal_ref = anotaciones_ref.sample  # Muestras reales del QRS fetal

# ==============================================================================
# 2. ALGORITMO PAN-TOMPKINS PARA QRS FETAL
# ==============================================================================
def pan_tompkins_fetal(signal, picos_maternos, fs):
    # a. Filtro Pasa-Banda (14 Hz - 40 Hz)
    b, a = butter(2, [14 / (fs / 2), 40 / (fs / 2)], btype='band')
    filtered = filtfilt(b, a, signal)
    
    # b. Derivada de 5 puntos
    derivative = np.pad(np.diff(filtered), (1, 0), 'edge')
    
    # c. Elevación al cuadrado
    squared = derivative ** 2
    
    # d. Integración por Ventana Móvil (~65 ms)
    win_len = int(0.065 * fs)
    kernel = np.ones(win_len) / win_len
    integrated = np.convolve(squared, kernel, mode='same')
    
    # e. Máscara para anular residuos del QRS materno en el integrador (±50 ms)
    integrated_masked = integrated.copy()
    win_m = int(0.05 * fs)
    for pm in picos_maternos:
        start_m = max(0, pm - win_m)
        end_m = min(len(integrated), pm + win_m)
        integrated_masked[start_m:end_m] = 0.0
        
    # f. Detección adaptativa por bloques de 6 segundos
    win_samples = int(6 * fs)
    min_dist = int(0.33 * fs)  # Refractariedad fetal (~180 BPM máx)
    t_skip = int(5 * fs)        # Ignorar transitorio de convergencia
    
    r_peaks = []
    
    for start in range(t_skip, len(integrated), win_samples):
        end = min(len(integrated), start + win_samples)
        segment = integrated_masked[start:end]
        
        segment_nz = segment[segment > 0]
        if len(segment_nz) == 0:
            continue
            
        th_local = 0.35 * np.percentile(segment_nz, 92)
        
        peaks_seg, _ = find_peaks(segment, height=th_local, distance=min_dist)
        peaks_global = start + peaks_seg
        
        # Búsqueda del máximo positivo local y compensación de desfase del integrador (~4.57 ms)
        search_win = int(0.035 * fs)
        offset_muestras = int(0.00457 * fs)
        
        for p in peaks_global:
            p_start = max(0, p - search_win)
            p_end = min(len(signal), p + search_win)
            idx_max = p_start + np.argmax(filtered[p_start:p_end])
            r_peaks.append(idx_max + offset_muestras)
            
    # Purga final por refractariedad estricta
    r_peaks = np.array(sorted(list(set(r_peaks))))
    if len(r_peaks) > 1:
        diffs = np.diff(r_peaks)
        valid = np.ones(len(r_peaks), dtype=bool)
        for idx in range(len(diffs)):
            if diffs[idx] < min_dist:
                valid[idx + 1] = False
        r_peaks = r_peaks[valid]
        
    return r_peaks, integrated_masked, filtered

# ==============================================================================
# 3. EJECUCIÓN Y VISUALIZACIÓN
# ==============================================================================
# Ejecutar la detección
picos_fetales_det, senal_integrada, senal_filtrada = pan_tompkins_fetal(promedio, picos_m, fs)

# Ploteo de control (Zoom en el tramo de 30s a 45s)
t_in, t_out = 30, 45
mask = (t >= t_in) & (t <= t_out)

mask_det = (picos_fetales_det / fs >= t_in) & (picos_fetales_det / fs <= t_out)
mask_ref = (picos_fetal_ref / fs >= t_in) & (picos_fetal_ref / fs <= t_out)

plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(t[mask], promedio[mask], color="magenta", linewidth=0.7, label="fECG Residual (NLMS)")
plt.plot(picos_fetales_det[mask_det] / fs, promedio[picos_fetales_det[mask_det]], "r*", markersize=8, label="Detección Pan-Tompkins")
plt.plot(picos_fetal_ref[mask_ref] / fs, promedio[picos_fetal_ref[mask_ref]], "go", markersize=6, fillstyle='none', label="Referencia Directa (.qrs)")
plt.title(f"Comparación Pan-Tompkins vs. Anotación de Referencia ({t_in}s - {t_out}s)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.legend(loc="upper right")
plt.grid(color="lightgray")

plt.subplot(2, 1, 2)
plt.plot(t[mask], senal_integrada[mask], color="purple", linewidth=0.7, label="Salida Integrador Enmascarado")
plt.title("Integrador por Ventana Móvil con Enmascaramiento Materno")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend(loc="upper right")
plt.grid(color="lightgray")

plt.tight_layout()
plt.show()
