import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Módulos.hrv_fetal import hrv_completo, BANDAS, guardar_fig

#Este módulo compara lo detectado con qrs_detect (desde las señales abdominales) contra la referencia: las
#anotaciones del electrodo en la cabeza del feto (FECG directo). Calcula tres cosas:
# 1) calidad de la detección de latidos (Se, PPV, F1)
# 2) exactitud latido a latido de los intervalos RR
# 3) cuánto se parece la HRV obtenida a la HRV de la referencia
#y guarda las figuras y un CSV con todas las métricas.



#DETECCIÓN DE LATIDOS

def evaluar(det, ref, tol):
    #empareja cada latido de referencia con la deteccion mas cercana que no se haya usado y cuenta TP,FP y FN, con eso devuelve Se, PPV, ACC y F1
    det = np.sort(det); ref = np.sort(ref) #ordena la lista detectada y la de referencia
    usados = np.zeros(len(det), dtype=bool) #para evitar dobles detecciones
    TP = 0 #verdaderos positivos
    for r in ref:
        d = np.abs(det - r).astype(float) #distancia entre lo detectado y el latido
        d[usados] = tol + 1    #si ya se detectó esa posición, le pone un valor más alto que el threshold para que no lo cuente
        k = np.argmin(d) #un latido es acierto si hay una deteccion dentro de la ventana de tolerancia
        if d[k] <= tol:
            TP += 1 #si entre la resta ve que es menor que el máxmo, entonces está bien detectdo
            usados[k] = True #lo marca pra no repetir
    FP = len(det) - TP #detectados que no son latidos reales
    FN = len(ref) - TP #latidos reales que no se detectaron

    #metricas estandar de deteccion de QRS
    Se  = TP / (TP + FN)
    PPV = TP / (TP + FP)
    ACC = TP / (TP + FP + FN)
    F1  = 2*TP / (2*TP + FN + FP)
    return TP, FP, FN, Se, PPV, ACC, F1

def metricas_deteccion(det, ref, fs, tol_s=0.05):
    #Convierte la tolerancia de segundos a muestras
    #llama a evaluar y devuelve un diccionario
    TP, FP, FN, Se, PPV, ACC, F1 = evaluar(det, ref, int(tol_s * fs))
    return {"TP": TP, "FP": FP, "FN": FN, "Se": Se, "PPV": PPV, "ACC": ACC, "F1": F1}



#EXACTITUD LATIDO A LATIDO DE LOS RR: ¿son exactos los RR?
def comparar_con_referencia(det, ref, fs, tol=0.05):
    #repite el emparejamiento guardando que deteccion corresponde a cada latido y con eso compara solo los intervalos RR cuyos dos extremos fueron detectados
    #devuelve las series de RR, el error y sus metricas (sesgo, desvio, RMSE y correlacion)
    det = np.sort(np.asarray(det)); ref = np.sort(np.asarray(ref))
    tol = tol * fs
    usados = np.zeros(len(det), dtype=bool)
    par = -np.ones(len(ref), dtype=int) #para cada latido de referencia, índice del detectado (-1 si no hay)
    for i, r in enumerate(ref): #hace falta saber que deteccion corresponde a cada latido, no solo cuantas
        d = np.abs(det - r).astype(float)
        d[usados] = tol + 1
        k = np.argmin(d)
        if d[k] <= tol:
            par[i] = k
            usados[k] = True

    #solo comparo intervalos donde ambos latidos consecutivos fueron detectados
    ok = (par[:-1] >= 0) & (par[1:] >= 0)
    i = np.where(ok)[0]
    rr_ref = (ref[i + 1] - ref[i]) / fs * 1000
    rr_det = (det[par[i + 1]] - det[par[i]]) / fs * 1000
    t = ref[i + 1] / fs
    ok_fis = (rr_ref >= 300) & (rr_ref <= 650) #ignoro tramos donde la referencia misma tiene huecos
    rr_ref, rr_det, t = rr_ref[ok_fis], rr_det[ok_fis], t[ok_fis] #error medio
    err = rr_det - rr_ref #error con signo
    return {
        "t": t, "rr_ref": rr_ref, "rr_det": rr_det, "err": err,
        "Intervalos comparados": len(err),
        "Error medio [ms]": err.mean(),
        "Desvío del error [ms]": err.std(ddof=1),
        "RMSE [ms]": np.sqrt(np.mean(err**2)),
        "Correlación RR": np.corrcoef(rr_ref, rr_det)[0, 1],
    }



#HRV DETECTADA vs HRV DE LA REFERENCIA

def comparar_hrv(res_det, res_ref):
    #Para cada metrica de la referencia calcula error= detectado-referencia y el error porcentual
    filas = []
    for k in res_ref:
        r, d = res_ref[k], res_det[k]
        err = d - r #el error relativo permite comparar metricas de distinta escala
        rel = 100 * err / r if r != 0 else np.nan
        filas.append((k, r, d, err, rel))
    return filas

def imprimir_comparacion(filas, titulo):
    print(f"\n=== HRV: referencia vs detectado - {titulo} ===")
    print(f"{'':<24}{'Referencia':>12}{'Detectado':>12}{'Error':>12}{'Error [%]':>12}")
    for k, r, d, e, rel in filas:
        print(f"{k:<24}{r:>12.3f}{d:>12.3f}{e:>12.3f}{rel:>12.1f}")



#GRÁFICOS DE LA VALIDACIÓN
def graficar_validacion(extra_det, extra_ref, comp, nombre, titulo="", guardar=None):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.suptitle(f"Validación contra FECG directo - {nombre} {titulo}")

    #Tacograma superpuesto
    #superpone 60/rr (FCF) de la referencia (negro) y del detectado (verde), solo rr VALIDOS
    a = ax[0]
    for ex, col, lab in [(extra_ref, "black", "Referencia (FECG directo)"), (extra_det, "darkgreen", "Detectado (abdominal)")]:
        v = ex["valido"]
        a.plot(ex["t"][v], 60 / ex["rr"][v], ".", ms=3, color=col, label=lab)
    a.set_xlabel("Tiempo [s]"); a.set_ylabel("FCF [lpm]"); a.set_title("Tacograma fetal")
    a.grid(color="lightgray"); a.legend()

    #PSD con Lomb-Scargle superpuesto
    #superpone ambos espectros sobre las bandas
    a = ax[1]
    colores = {"LF": "tab:blue", "MF": "tab:orange", "HF": "tab:red"}
    for k, (lo, hi) in BANDAS.items():
        a.axvspan(lo, hi, color=colores[k], alpha=0.12, label=k)
    a.plot(extra_ref["f"], extra_ref["p"], color="black", label="Referencia")
    a.plot(extra_det["f"], extra_det["p"], color="darkgreen", label="Detectado")
    a.set_xlabel("Frecuencia [Hz]"); a.set_ylabel("PSD [ms²/Hz]"); a.set_title("Periodograma de Lomb-Scargle")
    a.grid(color="lightgray"); a.legend()

    #Bland-Altman de los RR
    #X: RR medio de ambos metodos
    #Y: rr_det - rr_ref
    a = ax[2]
    m = (comp["rr_ref"] + comp["rr_det"]) / 2
    e = comp["err"]
    a.plot(m, e, ".", ms=3, color="purple")
    mu, sd = e.mean(), e.std(ddof=1)
    for yv, ls in [(mu, "-"), (mu + 1.96 * sd, "--"), (mu - 1.96 * sd, "--")]:
        a.axhline(yv, color="gray", linestyle=ls, linewidth=0.8)
    a.set_xlabel("RR medio [ms]"); a.set_ylabel("RR det - RR ref [ms]")
    a.set_title(f"Bland-Altman (sesgo {mu:.2f} ms, ±1.96 SD {1.96*sd:.1f} ms)")
    a.grid(color="lightgray")

    plt.tight_layout()
    if guardar is not None:
        guardar_fig(fig, guardar)
    return fig



#VALIDACIÓN COMPLETA
def validar(r_det, r_ref, fs, nombre, titulo="", carpeta=None):
    #r_det: latidos fetales detectados desde abdominales (qrs_detect); r_ref: anotaciones del FECG directo
    #si se pasa carpeta, guarda ahí la figura y un CSV con todas las métricas
    print(f"\n######## Validación {nombre} - {titulo} ########")

    det = metricas_deteccion(r_det, r_ref, fs)
    print("\n--- Detección de latidos ---")
    print(f"TP={det['TP']} FP={det['FP']} FN={det['FN']}  Se={det['Se']:.3f}  PPV={det['PPV']:.3f}  F1={det['F1']:.3f}")

    comp = comparar_con_referencia(r_det, r_ref, fs)
    print("\n--- Exactitud latido a latido (RR detectado vs RR referencia) ---")
    escalares = {k: v for k, v in comp.items() if k not in ("t", "rr_ref", "rr_det", "err")}
    for k, v in escalares.items():
        print(f"{k:<24}{v:>12.3f}")

    res_det, ex_det = hrv_completo(r_det, fs)
    res_ref, ex_ref = hrv_completo(r_ref, fs)
    filas = comparar_hrv(res_det, res_ref)
    imprimir_comparacion(filas, titulo)

    if carpeta is not None:
        carpeta = Path(carpeta)
        slug = (titulo or "deteccion").lower().replace(" ", "_")
        graficar_validacion(ex_det, ex_ref, comp, nombre, titulo, carpeta / f"validacion_{nombre}_{slug}.png")
        ruta_csv = carpeta / f"metricas_{nombre}_{slug}.csv"
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seccion", "metrica", "referencia", "detectado", "error", "error_%"])
            for k, v in det.items():
                w.writerow(["deteccion", k, "", v, "", ""])
            for k, v in escalares.items():
                w.writerow(["exactitud_rr", k, "", v, "", ""])
            for k, r, d, e, rel in filas:
                w.writerow(["hrv", k, r, d, e, rel])
        print(f"\nMétricas y figura guardadas en: {carpeta}")

    return {"deteccion": det, "rr": escalares, "hrv_det": res_det, "hrv_ref": res_ref, "tabla": filas}