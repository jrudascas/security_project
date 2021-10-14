from process import *
import os
import timeit



def example1():
    print("Ejemplo primera ejecución cuando no existe modelo preentrenado y si existen los " +
          "archivos de los modelos basicos de clasificación y cuantificación. Al final " +
          "guarda todos los modelos creados"
          )
    os.system("python process.py " +
              "--log_file '/home/unal/percepcion/Resultados/examples/example_log.log' " +
              "--save_model '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--subprocess 'clean' " +
              "--path_classify_model '/home/unal/percepcion/Resultados/examples/ClassifyModel.pkl' " +
              "--path_quantify_model '/home/unal/percepcion/Resultados/examples/QuantifyModel.pkl' " +
              "--keywords_path '/home/unal/percepcion/security_project/entradas/19032020_Palabras_Filtro.xls' " +
              "--vectors_path '/home/unal/percepcion/extra/words_vectors.vec' " +
              "--cmodel_path '/home/unal/percepcion/security_project/entradas/SVM_class.joblib' " +
              "--qmodel_path '/home/unal/percepcion/security_project/entradas/senticon.es.xml' " +
              "--qmodel_path '/home/unal/percepcion/security_project/entradas/senticon.es.xml' "               
              )


def example2():
    print("Ejemplo para entrenar un modelo ya creado y realizado el proceso de limpieza de datos")
    os.system("python process.py " +
              "--log_file '/home/unal/percepcion/Resultados/examples/example_log.log' " +
              "--exist_model_path '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--save_model '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--subprocess 'train' "          
              )

def example3():
    print("Ejemplo para realizar predicción de un modelo preentrenado")
    os.system("python process.py " +
              "--log_file '/home/unal/percepcion/Resultados/examples/example_log.log' " +
              "--exist_model_path '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--save_model '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--subprocess 'predict' " +
              "--predict_period '2019-03-21 00:00:00,2019-03-23 00:00:00' " +
              "--save_result_predict '/home/unal/percepcion/Resultados/examples/predict.csv' "
              )

start = timeit.default_timer()
example1()
clean = timeit.default_timer()
example2()
train = timeit.default_timer()
example3()
predict = timeit.default_timer()




print('Time clean: ', clean - start)  
print('Time train: ', train - start)  
print('Time predict: ', predict - start)  