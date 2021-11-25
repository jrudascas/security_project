from process import *
import os
import timeit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning) 
import shutup 
shutup.please()

def example1():
    print("Ejemplo para la creación de los poligonos de la red sobre el mapa de Bogotá.")
    os.system("python process.py " + "--log_file 'example_log.log' " + "--summary_file 'example_summary.log' " + "--subprocess 'poligonos_covariados' " + "--fecha_inicial '2021-01-01 00:00:00' " + "--fecha_final '2021-01-31 23:59:59' ")

def example2():
    print("Ejemplo para el procesamiento de los datos: Extracción de los datos de la API, limpieza y preparacion de los datos para el proceso entrenamiento")
    os.system("python process.py " + "--log_file 'example_log.log' " + "--summary_file 'example_summary.log' " + "--subprocess 'clean' " + "--fecha_inicial '2021-01-01 00:00:00' " + "--fecha_final '2021-02-07 23:59:59' ")

def example3():
    print("Ejemplo para el proceso de entrenamiento")
    os.system("python process.py " + "--log_file 'example_log.log' " + "--summary_file 'example_summary.log' " + "--subprocess 'train' " + "--fecha_inicial '2021-01-01 00:00:00' " + "--fecha_final '2021-01-31 23:59:59' ")

def example4():
    print("Ejemplo para el proceso de prediccion")
    os.system("python process.py " + "--log_file 'example_log.log' " + "--summary_file 'example_summary.log' " + "--subprocess 'predict' " + "--fecha_inicial '2021-01-01 00:00:00' " + "--fecha_final '2021-01-03 13:59:59' " + "--fecha_inicial_pr '2021-01-04 00:00:00' " + "--fecha_final_pr '2021-01-04 09:59:00' ")

def example5():
    print("Ejemplo para el proceso de validacion")
    os.system("python process.py " + "--log_file 'example_log.log' " + "--summary_file 'example_summary.log' " + "--subprocess 'validation' " + "--fecha_inicial '2021-01-01 00:00:00' " + "--fecha_final '2021-01-31 23:59:59' "+ "--fecha_inicial_pr '2021-01-21 00:00:00' " + "--fecha_final_pr '2021-01-26 23:59:59' ")

example3()
