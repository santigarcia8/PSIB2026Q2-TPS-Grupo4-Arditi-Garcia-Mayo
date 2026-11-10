# PSIB2026Q2-TPS-Grupo4-Arditi-Garcia-Mayo
# HRV fetal a partir de ECG abdominal

**PSIB2026Q2-TPS-Grupo4-Arditi-Garcia-Mayo**
Trabajo Práctico de Señales (TPS) — 16.63 Procesamiento de Señales e Imágenes Biomédicas, ITBA, 2026-Q2.

## Descripción

El proyecto estima la **variabilidad de la frecuencia cardíaca fetal (HRV fetal)** a partir de registros de ECG abdominal materno, y valida el resultado contra el **electrodo colocado en la cabeza del feto** (ECG fetal directo), que se toma como referencia.

Pipeline completo:

1. **Preprocesamiento** de los canales (filtro pasabanda y notch de 50 Hz según registro).
2. **Detección del QRS materno** con Pan-Tompkins (se usa *solo* para la madre).
3. **Cancelación del ECG materno** (referencia del ECG materno estimada por SVD + filtro adaptativo RLS, cuyo error es la estimación del ECG fetal).
4. **Detección de los latidos fetales** con `qrs_detect` (traducción a Python de `qrs_detect.m`, J. Behar).
5. **Serie RR y HRV fetal** (dominio temporal y periodograma de Lomb-Scargle).
6. **Validación** contra las anotaciones del electrodo directo: calidad de la detección, exactitud de los RR y diferencia entre la HRV estimada y la de referencia.

## Base de datos

[Abdominal and Direct Fetal ECG Database v1.0.0](https://physionet.org/content/adfecgdb/1.0.0/) (PhysioNet).
Registros usados: `r01`, `r04`, `r07` y `r10`. Cada uno dura 5 minutos y está muestreado a 1 kHz (`fs = 1000 Hz`).

Artículo de referencia del dataset: *Determination of fetal heart rate from abdominal signals: evaluation of beat-to-beat accuracy in relation to the direct fetal electrocardiogram* — DOI [10.1515/bmt-2011-0130](https://doi.org/10.1515/bmt-2011-0130).

De la descarga se necesitan, para cada registro, **dos archivos**:

| Archivo | Contenido |
|---|---|
| `rXX.edf` | Señales (canales abdominales y electrodo directo) |
| `rXX.edf.qrs` | Anotaciones de los latidos fetales (referencia) |

## Instalación

Requiere **Python 3.10 o superior**.

```bash
# 1) clonar el repositorio y entrar a la carpeta
git clone https://github.com/santigarcia8/PSIB2026Q2-TPS-Grupo4-Arditi-Garcia-Mayo.git
cd PSIB2026Q2-TPS-Grupo4-Arditi-Garcia-Mayo

# 2) crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3) instalar las dependencias
pip install -r requirements.txt
```

Dependencias (ver `requirements.txt`): `numpy`, `scipy`, `matplotlib`, `mne` (lectura de EDF) y `wfdb` (lectura de las anotaciones `.qrs`).

## Preparar los datos

El dataset **no está incluido en el repositorio**: cada uno lo descarga y lo copia a la carpeta `data/raw/`, que está en el `.gitignore` (así los datos nunca se suben).

1. Descargar el dataset desde https://physionet.org/content/adfecgdb/1.0.0/ (botón **Download the ZIP file**) y descomprimirlo.
2. En la raíz del repositorio (al lado de `main.py`) crear la carpeta `data/raw/`.
3. Copiar ahí, de los registros **r01, r04, r07 y r10**, el `.edf` **y** su `.edf.qrs`, sin cambiarles el nombre. El resto de los archivos del ZIP no se usa.

```
data/
└── raw/
    ├── r01.edf
    ├── r01.edf.qrs
    ├── r04.edf
    ├── r04.edf.qrs
    ├── r07.edf
    ├── r07.edf.qrs
    ├── r10.edf
    └── r10.edf.qrs
```

Si falta el archivo de anotaciones de un registro, el programa se detiene y avisa cuál es.

## Ejecución

```bash
python main.py
```

1. El programa lista los registros disponibles en `data/raw/`.
2. Se elige el registro ingresando su número (con `0` se sale).
3. Se ejecuta todo el pipeline y se imprimen en consola las métricas de detección, de exactitud de RR y la tabla de HRV (referencia vs. detectado).

**No se abre ninguna ventana**: todas las figuras se guardan en `Resultados/` y al final el programa informa cuántas figuras y archivos de métricas se guardaron y dónde.

## Estructura del repositorio

```
.
├── main.py
├── requirements.txt
├── README.md
├── data/raw/                 # dataset (.edf y .edf.qrs), no se sube
├── Módulos/
│   ├── preprocesamiento.py
│   ├── pantompkins.py
│   ├── filtro_adaptativo.py
│   ├── qrs_detect.py
│   ├── hrv_fetal.py
│   └── validacion.py
└── Resultados/               # se genera al ejecutar
```

| Archivo | Qué contiene |
|---|---|
| `main.py` | Programa principal: arma el flujo completo y guarda los resultados. |
| `Módulos/preprocesamiento.py` | Menú de elección de registro, lectura del EDF, filtrado (Butterworth pasabanda 1–100 Hz, notch de 50 Hz en los canales afectados de cada registro) y gráficos de señal y PSD (Welch) antes y después de filtrar. Incluye `guardar_figura`. |
| `Módulos/pantompkins.py` | Detector de QRS de Pan-Tompkins (pasabanda, derivada, cuadrado, integración, umbrales adaptativos con searchback y regla de onda T, ajuste del R). Se usa **solo** para el QRS materno. |
| `Módulos/filtro_adaptativo.py` | Cancelación del ECG materno: `referencia_materna` (matriz de latidos maternos, estimación por SVD y suavizado entre QRS) y `filtro_rls` (filtro adaptativo RLS cuya salida estima el ECG fetal). |
| `Módulos/qrs_detect.py` | Detector de QRS fetal (traducción a Python de `qrs_detect.m`, J. Behar). Se usa para detectar los latidos fetales. |
| `Módulos/hrv_fetal.py` | A partir de los latidos detectados: serie RR, limpieza de RR, métricas temporales, periodograma de Lomb-Scargle y gráficos de HRV. |
| `Módulos/validacion.py` | Comparación contra el electrodo directo: Se/PPV/F1, exactitud de los RR (error, RMSE, correlación, Bland-Altman) y HRV detectada vs. referencia. Guarda figura y `.csv`. |

## Detalle del procesamiento

**Preprocesamiento** — Butterworth pasabanda de orden 5 (1–100 Hz) con `filtfilt` (fase cero). Notch de 50 Hz (Q = 30) solo en los canales con interferencia de línea: r01 y r10 (canal 3); r04 y r07 (canales 1 y 3). Antes de la cancelación materna, `main.py` aplica además un pasaaltos de 8 Hz y un pasabajos de 100 Hz (orden 5).

**QRS materno (Pan-Tompkins)** — pasabanda 5–15 Hz, derivada de 5 puntos, cuadrado, integración de 150 ms, umbrales adaptativos con searchback, y ajuste del R sobre la señal pasabanda.

**Cancelación del ECG materno** — se arma una matriz con segmentos de 101 muestras alrededor de cada R materno (40 antes y 60 después), se estima cada latido con SVD sobre los últimos 20 latidos usando la componente dominante (`q = 1`) y se suaviza el tramo entre QRS (ventana de 30 muestras). Luego el filtro RLS (20 coeficientes, `λ = 1`, `δ = 1e-4`) usa esa referencia para obtener el ECG fetal como señal de error.

**QRS fetal** — `qrs_detect(f_est, ref_period=0.15, thres=0.42, fs)`.

**Serie RR y limpieza** — el RR de cada par de latidos consecutivos se marca como válido si está entre 0.30 y 0.65 s y no se aparta más de 10 % de la mediana de sus vecinos (ventana de 11 latidos). Los RR inválidos se descartan: no se interpolan.

**Métricas de HRV**

| Dominio | Métricas |
|---|---|
| Calidad | N de RR válidos, % descartados |
| Temporal | FCF media, RR medio, SDNN, RMSSD, STV (promedio de \|ΔRR\|), LTV (rango de FCF por minuto) |
| Frecuencial | Potencia en LF (0.04–0.15 Hz), MF (0.15–0.5 Hz) y HF (0.5–1 Hz), porcentajes y cociente LF/(MF+HF), con periodograma de Lomb-Scargle (0.02–1 Hz) |

**Validación** — un latido detectado es correcto si está a menos de 50 ms de uno de referencia. Los RR se comparan solo en los intervalos cuyos dos latidos fueron detectados.

## Resultados generados

```
Resultados/<registro>/
├── ECG original/             # una figura por canal
├── Periodograma crudo/
├── ECG filtrado/
├── Periodograma filtrado/
├── QRS maternos/             # R maternos detectados (primeros 10 s)
├── HRV fetal/                # tacograma y periodograma de lo detectado
└── Validación/
    ├── validacion_<registro>_qrs_detect.png   # tacograma, PSD y Bland-Altman vs. referencia
    └── metricas_<registro>_qrs_detect.csv     # todas las métricas (detección, RR y HRV)
```

El `.csv` tiene las columnas `seccion, metrica, referencia, detectado, error, error_%`.

## Notas y limitaciones

- Los registros duran 5 minutos y la serie de FCF no es estacionaria (por ejemplo, con aceleraciones), por lo que las potencias espectrales deben interpretarse con cuidado.
- Los valores absolutos de potencia espectral están escalados a la varianza de la serie; los porcentajes y el cociente son más comparables.
- STV y LTV son versiones simplificadas (por latido y por minuto), no las definiciones clínicas de cardiotocografía.
- La HRV de variabilidad rápida (RMSSD, STV, HF) es sensible al error de ubicación del R, por lo que en el ECG abdominal tiende a sobreestimarse respecto de la referencia directa.

## Referencias

- Dataset: *Abdominal and Direct Fetal ECG Database* v1.0.0, PhysioNet. https://physionet.org/content/adfecgdb/1.0.0/
- Artículo del dataset: DOI 10.1515/bmt-2011-0130.
- J. Pan y W. J. Tompkins, "A real-time QRS detection algorithm", *IEEE Trans. Biomed. Eng.*, 32(3):230–236, 1985.
- J. Behar, `qrs_detect.m` (detector de QRS traducido a Python en `Módulos/qrs_detect.py`).
- L. Castro et al., "Systematic Review of Intrapartum Fetal Heart Rate Spectral Analysis and an Application in the Detection of Fetal Acidemia", *Frontiers in Pediatrics*, 9:661400, 2021 (bandas LF/MF/HF de HRV fetal).
- N. Zhang et al., "A novel technique for fetal ECG extraction using single-channel abdominal recording", *Sensors*, 17(3):457, 2017. DOI [10.3390/s17030457](https://doi.org/10.3390/s17030457) (cancelación del ECG materno: SVD + suavizado + filtro adaptativo).
- P. P. Kanjilal, S. Palit y G. Saha, "Fetal ECG extraction from single-channel maternal ECG using singular value decomposition", *IEEE Trans. Biomed. Eng.*, 44(1):51–59, 1997. DOI [10.1109/10.553712](https://doi.org/10.1109/10.553712).