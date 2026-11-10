import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lombscargle

#Este módulo trabaja SOLO con una serie de latidos fetales (los detectados con qrs_detect): arma la serie RR y calcula la HRV

#integral numerica por trapecios que se usa para calcular areas bajo la PSD
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

#Bandas de HRV fetal (distintas a las del adulto porque la FC fetal es ~2x y la respiración también)
#LF: actividad simpática/barorreflejo; MF: movimientos fetales/ondas de Mayer fetales; HF: respiración fetal
BANDAS = {"LF": (0.04, 0.15), "MF": (0.15, 0.5), "HF": (0.5, 1.0)}


def guardar_fig(fig, ruta):
    #guarda la figura en ruta (creando la carpeta si no existe) y la cierra para que no se abra ninguna ventana
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=300)
    plt.close(fig)



#SERIE RR Y LIMPIEZA

def serie_rr(picos, fs):
    #ordena los picos, elimina duplicados y calcula diff(picos)/fs que da el RR en segundos
    #a cada RR le asigna el instante del latido  en el que termina
    picos = np.sort(np.unique(np.asarray(picos))) #unique elimina posiciones repetidas
    rr = np.diff(picos) / fs #da las muestras entre latidos consecutivos y dividir por fs las pasa a segundos
    t = picos[1:] / fs
    return t, rr

def limpiar_rr(rr, rr_min=0.30, rr_max=0.65, ventana=11, tol=0.10):
    #Devuelve una mascara booleana: True si el RR es utilizable
    #No modifica la serie, no interpola y no inventa latidos

    #valida falsas detecciones y se usa un rango un poco mas ancho para no recortar taquicardias o bradicardias reales
    valido = (rr >= rr_min) & (rr <= rr_max)

    #Se toma una ventana de 11 latidos para cada RR (centrado en el RR sobre el que estoy parado)
    #De esa ventana solo usa los vecino que pasaron por validacion y asi no se usan RR que contaminen la cuenta
    #Se calcula la mediana y compara: si se aparta de mas del 10% de tolerancia (arbitrario) se descarta, si no queda vecino valido usa la mediana global
    #Usamos la mediana y no un umbral global fijo porque permite que la FCF suba o baje lentamente sin que el filtro las descarte

    h = ventana // 2
    med = np.zeros(len(rr))
    for i in range(len(rr)):
        a, b = max(0, i - h), i + h + 1
        vecinos = rr[a:b][valido[a:b]]
        med[i] = np.median(vecinos) if len(vecinos) > 0 else np.median(rr)
    valido &= np.abs(rr - med) <= tol * med
    return valido

#DOMINIO TEMPORAL
def metricas_temporales(t, rr, valido):
    #primero pasa a milisegundos y se queda con los RR validos
    #fhr es la FCF instantanea de cada latido
    rr_ms = rr * 1000
    rr_v = rr_ms[valido]
    fhr = 60000 / rr_v

    #par marca las parejas de RR consecutivos que son ambos validos
    #np.diff calcula la diferencia entre RR adyacentes en la serie completa
    #Si hubiera un RR descartado en el medio, la diferencia enre el anterior y el siguiente mediria dos latidos, no uno e inflaria la variabilidad
    par = valido[1:] & valido[:-1]
    d = np.diff(rr_ms)[par]

    #LTV fetal
    #etiqueta cada RR con el minuto al que pertenece, seleccion los validos de ese minuto, descarta los minutos con menos de 10 RR
    #pasa a FCF y toma el rango maximo menos el minimo en lpm
    #despues promedia los rangos de todos los minutos
    minuto = np.floor(t / 60).astype(int)
    rangos = []
    for m in np.unique(minuto):
        sel = valido & (minuto == m)
        if sel.sum() >= 10: #descarto minutos casi vacíos
            f = 60 / rr[sel]
            rangos.append(f.max() - f.min())

    return { 
        #calidad de la deteccion
        "N RR válidos": int(valido.sum()), 
        "% RR descartados": 100 * (1 - valido.mean()),

        #nivel basal
        "FCF media [lpm]": fhr.mean(),
        "RR medio [ms]": rr_v.mean(),

        #variabilidad total (corta y larga)
        "SDNN [ms]": rr_v.std(ddof=1),

        #variabilidad latido a latido, eleva al cuadrado para darle mas peso a los saltos grandes
        "RMSSD [ms]": np.sqrt(np.mean(d**2)),

        #tambien es variabilidad latido a latido pero con valor absoluto
        "STV [ms]": np.mean(np.abs(d)),

        #variabilidad de minutos
        "LTV [lpm]": np.mean(rangos) if rangos else np.nan,
    }


#DOMINIO FRECUENCIAL (LOMB-SCARGLE)

def psd_lomb(t, rr, valido, fmin=0.02, fmax=1.0, n=500):
    #Los RR no estan equiespaciados 
    #La FFT estandar exige muestras uniformes y al hacer el tacograma se suele agregar datos inventados que distorsionan el espectro
    #Al usar Lomb-Scargle se estima el espectro ajustando senoides solo a las muestras reales en sus instantes reales

    tv = t[valido] #Selecciona solo los RR validos y sus tiempos
    x = rr[valido] * 1000 #Pasa a ms y despues resta la media para que el espectro no tenga una componente grande en frecuencia cero
    x = x - x.mean()
    f = np.linspace(fmin, fmax, n) #defiene 500 frecuencias equiespaciadas entre 0.02 y 1Hz
    p = lombscargle(tv, x, 2 * np.pi * f, precenter=False, normalize=False) #espera frecuencias angulares por eso el 2pif
    p = p / _trapz(p, f) * np.var(x) #reescala dividiendo por el area bajo la curva y multiplica por la varianza
    return f, p

def potencias_banda(f, p):
    #para cada banda selecciona las frecuencias que caen dentro e integra la PSD con trapecios
    #Da la potencia en ms^2 
    #El total es LF+MF+HF

    res = {}
    for nombre, (a, b) in BANDAS.items():
        sel = (f >= a) & (f <= b)
        res[nombre] = _trapz(p[sel], f[sel])
    total = sum(res.values())
    out = {f"P {k} [ms²]": v for k, v in res.items()}
    out.update({f"{k} [%]": 100 * v / total for k, v in res.items()})
    out["LF/(MF+HF)"] = res["LF"] / (res["MF"] + res["HF"])
    return out


#HRV COMPLETA DE UNA SERIE DE LATIDOS + GRÁFICOS
def hrv_completo(picos, fs):
    t, rr = serie_rr(picos, fs)
    valido = limpiar_rr(rr)
    res = metricas_temporales(t, rr, valido)
    f, p = psd_lomb(t, rr, valido)
    res.update(potencias_banda(f, p))
    extra = {"t": t, "rr": rr, "valido": valido, "f": f, "p": p}
    return res, extra

def imprimir_tabla(res, titulo="HRV fetal"):
    print(f"\n=== {titulo} ===")
    for k, v in res.items():
        print(f"{k:<24}{v:>12.3f}")

def graficar_hrv(extra, nombre, titulo="", guardar=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(f"HRV fetal - {nombre} {titulo}")
    v = extra["valido"]

    #Tacograma: FCF instantánea (los RR descartados se marcan aparte)
    a = ax[0]
    fhr = 60 / extra["rr"]
    a.plot(extra["t"][v], fhr[v], ".", ms=3, color="darkgreen", label="RR válidos")
    if (~v).any():
        a.plot(extra["t"][~v], fhr[~v], "x", ms=4, color="tab:red", label="RR descartados")
    a.set_ylim(fhr[v].min() - 10, fhr[v].max() + 10)
    a.set_xlabel("Tiempo [s]"); a.set_ylabel("FCF [lpm]"); a.set_title("Tacograma fetal")
    a.grid(color="lightgray"); a.legend()

    #PSD con Lomb-Scargle
    a = ax[1]
    colores = {"LF": "tab:blue", "MF": "tab:orange", "HF": "tab:red"}
    for k, (lo, hi) in BANDAS.items():
        a.axvspan(lo, hi, color=colores[k], alpha=0.12, label=k)
    a.plot(extra["f"], extra["p"], color="darkgreen")
    a.set_xlabel("Frecuencia [Hz]"); a.set_ylabel("PSD [ms²/Hz]"); a.set_title("Periodograma de Lomb-Scargle")
    a.grid(color="lightgray"); a.legend()

    plt.tight_layout()
    if guardar is not None:
        guardar_fig(fig, guardar)
    return fig

def analisis_hrv(picos, fs, nombre, titulo="", guardar=None):
    #picos: latidos fetales detectados (muestras). Devuelve las métricas y los datos para graficar
    res, extra = hrv_completo(picos, fs)
    imprimir_tabla(res, titulo=f"HRV fetal {nombre} - {titulo}")
    graficar_hrv(extra, nombre, titulo, guardar)
    return res, extra