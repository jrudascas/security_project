from process import *
import os
import timeit

scripts_path=os.getcwd()
parent_path=os.path.dirname(scripts_path)
examples_path=os.path.join(parent_path,"Resultados","examples")


def example1():
    print("Ejemplo primera ejecución cuando no existe modelo preentrenado y si existen los " +
          "archivos de los modelos basicos de clasificación y cuantificación. Al final " +
          "guarda todos los modelos creados"
          )
    os.system("python process.py " +
              "--log_file "+os.path.join(examples_path,"examples_log.log") +" "+
              "--summary_file "+os.path.join(examples_path,"example_summ.log") +" "+          
              "--save_model "+os.path.join(examples_path,"modelPoS.pkl") +" "+          
              "--subprocess 'clean' " +
              "--save_freq_palabras "+os.path.join(examples_path,"example_words.csv") +" "+          
              "--save_real_scores "+os.path.join(examples_path,"example_scores.csv") +" "+          
              "--path_classify_model "+os.path.join(examples_path,"ClassifyModel.pkl") +" "+        
              "--path_quantify_model "+os.path.join(examples_path,"QuantifyModel.pkl") +" "+   
              "--keywords_path "+os.path.join(scripts_path,"entradas","'19032020 Palabras Filtro.xlsx'") +" "+ 
              "--vectors_path "+os.path.join(parent_path,"extra","words_vectors.vec") +" "+    
              "--cmodel_path "+os.path.join(scripts_path,"entradas","SVM_class.joblib") +" "+     
              "--qmodel_path "+os.path.join(scripts_path,"entradas","senticon.es.xml") +" "
              )


def example2():
    print("Ejemplo para entrenar un modelo ya creado y realizado el proceso de limpieza de datos")
    os.system("python process.py " +
              "--log_file "+os.path.join(examples_path,"examples_log.log") +" "+
              "--summary_file "+os.path.join(examples_path,"example_summ_train.log") +" "+
              "--exist_model_path "+os.path.join(examples_path,"modelPoS.pkl") +" "+ 
              "--save_model "+os.path.join(examples_path,"modelPoS.pkl") +" "+
              "--subprocess 'train' "          
              )

def example3():
    print("Ejemplo para realizar predicción de un modelo preentrenado")
    os.system("python process.py " +
              "--log_file "+os.path.join(examples_path,"examples_log.log") +" "+
              "--summary_file "+os.path.join(examples_path,"example_summ_predict.log") +" "+
              "--exist_model_path "+os.path.join(examples_path,"modelPoS.pkl") +" "+ 
              "--save_model "+os.path.join(examples_path,"modelPoS.pkl") +" "+
              "--subprocess 'predict' " +
            #   "--predict_period '2019-03-21 00:00:00,2019-03-23 00:00:00' " +
              "--predict_period '2020-04-28 00:00:00,2020-05-05 00:00:00' " +
              "--save_result_predict "+os.path.join(examples_path,"predict1.csv") +" " +
              "--win_size_pred_period 12"
              )

def example4():
    print("Ejemplo para realizar validación de un modelo preentrenado ")
    os.system("python process.py " +
              "--log_file "+os.path.join(examples_path,"examples_log.log") +" "+
              "--summary_file "+os.path.join(examples_path,"example_summ_validation1.log") +" "+
              "--exist_model_path "+os.path.join(examples_path,"modelPoS.pkl") +" "+ 
              "--save_model "+os.path.join(examples_path,"modelPoS.pkl") +" "+
              "--subprocess 'validate' " +
              "--valid_period '2020-03-10 00:00:00,2020-03-20 00:00:00' "
            # "--valid_period '2019-03-19 00:00:00,2019-03-20 00:00:00' "
              )

def example5():
    print("Ejemplo con modelo preentrenado, descarga de nuevos datos y limpieza "
          )
    os.system("python process.py " +
              "--log_file "+os.path.join(examples_path,"examples_log.log") +" "+
              "--summary_file "+os.path.join(examples_path,"example_download.log") +" "+
              "--save_model "+os.path.join(examples_path,"modelPoS.pkl") +" "+
              "--subprocess 'clean' " +
              "--exist_model_path "+os.path.join(examples_path,"modelPoS.pkl") +" "+ 
              "--save_freq_palabras "+os.path.join(examples_path,"example_words1.csv") +" "+          
              "--save_real_scores "+os.path.join(examples_path,"example_scores1.csv") +" "+
              "--f_limite '2019-10-28 00:00:00' "
            #   "--f_limite '2019-03-19 00:00:00' "
              )

def example6():
    print("Ejemplo cambio parametros de entrenamiento "
          )
    os.system("python process.py " +
              "--log_file "+os.path.join(examples_path,"examples_log.log") +" "+
              "--summary_file "+os.path.join(examples_path,"example_change_train.log") +" "+
              "--save_model "+os.path.join(examples_path,"modelPoS.pkl") +" "+
              "--subprocess 'train' " +
              "--exist_model_path "+os.path.join(examples_path,"modelPoS.pkl") +" "+ 
              "--win_size_for_partition_cov 12.5" +" "+ 
              "--followers_rate 4" +" "+ 
              "--win_size_infectious_rate 6" +" "+ 
              "--win_size_train_period 12" +" "
              )                  

# start = timeit.default_timer()
# example1()
# clean = timeit.default_timer()
# example2()
# train = timeit.default_timer()
example3()
# predict = timeit.default_timer()
# example4()
# validate = timeit.default_timer()
# example5()
# download = timeit.default_timer()
# example6()
# changing = timeit.default_timer()

# print('Time clean: ', clean - start)  
# print('Time train: ', train - clean)  
# print('Time predict: ', predict - train)  
# print('Time validation: ', validate - predict)  
# print('Time dowload: ', download - validate)  
# print('Time changing parameters: ', changing - download)  