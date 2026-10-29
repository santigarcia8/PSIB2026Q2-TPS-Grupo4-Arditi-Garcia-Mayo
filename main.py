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

if __name__ == "__main__":
    main()