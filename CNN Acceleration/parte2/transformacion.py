import csv
import os
import shutil
import openpyxl
from openpyxl import load_workbook
import matplotlib.pyplot as plt

def listar_archivos(directorio):
    lista_archivos = []
    archivos = os.listdir(directorio)
    archivos_ordenados = sorted(archivos)
    for archivo in archivos_ordenados:
        lista_archivos.append(archivo)
    return lista_archivos

def procesar_txt(lista_txt):
    dir_principal = '/home/alumnos/mgonzalez/trt_txt/'
    informacion = []
    for archivo in lista_txt:
        dir_archivo = dir_principal + archivo
        try:
            with open(dir_archivo, 'r') as txt:
                contenido = txt.readlines()
                ultimas_lineas = [linea.rstrip('\n') for linea in contenido[-12:]]
                informacion.append(ultimas_lineas)
        except:
            print("el archivo no existe")
    return informacion

def reparticion(info_archivos):

    databases = {
    'casia': [],
    'iit': [],
    'polyu': [],
    'put': [],
    'syntheticspvd': [],
    'tongji': [],
    'vera': []
    }

    indice_inicio = 0
    tamano_grupo = 12

    for nombre_lista, lista_vacia in databases.items():
        lista_vacia.extend(info_archivos[indice_inicio:indice_inicio+tamano_grupo])
        indice_inicio += tamano_grupo

    #print(len(databases['vera']))
    return databases

def xlsx_funcion(databases):

    for nombre_lista, lista in databases.items():
        archivo_xlsx = openpyxl.Workbook()
        hoja = archivo_xlsx.active
        nombres_columnas = obtener_nombres_columnas()
        hoja.append(nombres_columnas)
        for elementos in lista:
            hoja.append(elementos)
        archivo_xlsx.save(f"{nombre_lista}.xlsx")
        

def obtener_nombres_columnas():
    
    nombres_columnas = ["---", "Throughput", "Latency", " Enqueue Time", " H2D Latency", "GPU Compute Time", "D2H Latency", "Total Host Walltime", "Total GPU Compute Time", "---", "---", "NOMBRE ARCHIVO"]

    return nombres_columnas

def datos_necesarios(databases):
    lista_datos = []
    info_final = []
    for nombre, info in databases.items():
        for j in range(12):
            cont=0
            aux = []
            for i in databases[nombre][j]:
                if cont == 5 or cont == 11:
                    aux.append(i)
                cont+=1

            lista_datos.append(aux)  
     
    for ja in lista_datos:
        lista2 = []
        aux = ja[1].rsplit("/", maxsplit=1)
        nombre = aux[1]
        aux2 = ja[0].split(",", maxsplit=5)
        lista2.append(nombre)
        lista2.append(aux2[2])
        info_final.append(lista2)

    # for i in info_final:
    #     print(i)
    return info_final

def graficar(datos):
    grupos = [datos[i:i+3] for i in range(0, len(datos), 3)]

    nombres = [elemento[0] for grupo in grupos for elemento in grupo]
    valores = [float(elemento[1].split('=')[1].strip().split(' ')[0]) for grupo in grupos for elemento in grupo]
    plt.bar(nombres, valores)
    plt.xlabel('Elemento')
    plt.ylabel('mean')
    plt.title('Gráfico de barras')
    plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para mayor legibilidad
    plt.show()
        
if __name__ == "__main__":
    directorio_txt = '/home/alumnos/mgonzalez/trt_txt/'
    lista_archivos = listar_archivos(directorio_txt)
    info_archivos = procesar_txt(lista_archivos)
    databases = reparticion(info_archivos)
    info_final = datos_necesarios(databases)
    graficar(info_final)
    #xlsx_funcion(databases)

    # for i in info_archivos:
    #     print(i)
    #     print("\n")

                