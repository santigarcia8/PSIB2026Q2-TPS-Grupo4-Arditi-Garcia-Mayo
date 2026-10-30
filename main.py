from pathlib import Path
from Módulos.preprocesamiento import eleccion_archivos, apertura_archivos, preprocesamiento

def main():
    path, raiz = eleccion_archivos()
    if path is None:
        print("Saliendo del programa")
        return

    señal, canales, fs = apertura_archivos(path)
    nombre = Path(path).stem
    señal_filtrada = preprocesamiento(señal, canales, fs, nombre, raiz)
    
    from Módulos.pantompkins import detectar_qrs
    import matplotlib.pyplot as plt
    x = señal_filtrada[:, 1] #abdomen 1
    picos_r = detectar_qrs(x, fs)
    print(f"QRS: {len(picos_r)}  FC materna media: {60*len(picos_r)/(len(x)/fs):.0f} lpm")
    n = int(10*fs)
    r = picos_r[picos_r < n]
    plt.figure(figsize=(12,4))
    plt.plot(x[:n], linewidth=0.5, color="darkgreen", label="Abdomen 1 preprocesada")
    plt.plot(r, x[r], "o", color="black", label="R materno")
    plt.title("Pan-Tompkins: R maternos")
    plt.legend(); plt.tight_layout(); plt.show()
    
if __name__ == "__main__":
    main()