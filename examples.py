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
              "--summary_file '/home/unal/percepcion/Resultados/examples/example_summ.log' " +
              "--save_model '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--subprocess 'clean' " +
              "--save_freq_palabras '/home/unal/percepcion/Resultados/examples/example_words.csv' " +
              "--save_real_scores '/home/unal/percepcion/Resultados/examples/example_scores.csv' " +
              "--path_classify_model '/home/unal/percepcion/Resultados/examples/ClassifyModel.pkl' " +
              "--path_quantify_model '/home/unal/percepcion/Resultados/examples/QuantifyModel.pkl' " +
              "--keywords_path '/home/unal/percepcion/security_project/entradas/19032020 Palabras Filtro.xlsx' " +
              "--vectors_path '/home/unal/percepcion/extra/words_vectors.vec' " +
              "--cmodel_path '/home/unal/percepcion/security_project/entradas/SVM_class.joblib' " +
              "--qmodel_path '/home/unal/percepcion/security_project/entradas/senticon.es.xml' "               
              )


def example2():
    print("Ejemplo para entrenar un modelo ya creado y realizado el proceso de limpieza de datos")
    os.system("python process.py " +
              "--log_file '/home/unal/percepcion/Resultados/examples/example_log.log' " +
              "--summary_file '/home/unal/percepcion/Resultados/examples/example_summ_train.log' " +
              "--exist_model_path '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--save_model '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--subprocess 'train' "          
              )

def example3():
    print("Ejemplo para realizar predicción de un modelo preentrenado")
    os.system("python process.py " +
              "--log_file '/home/unal/percepcion/Resultados/examples/example_log.log' " +
              "--summary_file '/home/unal/percepcion/Resultados/examples/example_summ_predict.log' " +
              "--exist_model_path '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--save_model '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--subprocess 'predict' " +
            #   "--predict_period '2019-03-21 00:00:00,2019-03-23 00:00:00' " +
              "--predict_period '2019-05-21 04:00:00,2019-05-30 00:00:00' " +
              "--save_result_predict '/home/unal/percepcion/Resultados/examples/predict.csv' "
              )

def example4():
    print("Ejemplo para realizar validación de un modelo preentrenado ")
    os.system("python process.py " +
              "--log_file '/home/unal/percepcion/Resultados/examples/example_log.log' " +
              "--summary_file '/home/unal/percepcion/Resultados/examples/example_summ_validation.log' " +
              "--exist_model_path '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--save_model '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--subprocess 'validate' " +
              "--valid_period '2019-03-19 00:00:00,2019-03-20 00:00:00' "
              )

def example5():
    print("Ejemplo con modelo preentrenado, descarga de nuevos datos y limpieza "
          )
    os.system("python process.py " +
              "--log_file '/home/unal/percepcion/Resultados/examples/example_log.log' " +
              "--summary_file '/home/unal/percepcion/Resultados/examples/example_download.log' " +
              "--save_model '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--subprocess 'clean' " +
              "--exist_model_path '/home/unal/percepcion/Resultados/examples/modelPoS.pkl' " +
              "--save_freq_palabras '/home/unal/percepcion/Resultados/examples/example_words1.csv' " +
              "--save_real_scores '/home/unal/percepcion/Resultados/examples/example_scores1.csv' " +
              "--f_limite '2019-03-19 00:00:00' "
              )
                  

start = timeit.default_timer()
example1()
clean = timeit.default_timer()
example2()
train = timeit.default_timer()
example3()
predict = timeit.default_timer()
example4()
validate = timeit.default_timer()
example5()
download = timeit.default_timer()


print('Time clean: ', clean - start)  
print('Time train: ', train - clean)  
print('Time predict: ', predict - train)  
print('Time validation: ', validate - predict)  
print('Time dowload: ', download - validate)  