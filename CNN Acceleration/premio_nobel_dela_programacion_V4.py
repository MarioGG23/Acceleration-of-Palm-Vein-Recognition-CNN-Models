import os
import cv2
import time
import glob
import pathlib
import argparse
import numpy as np
import os.path as osp
from keras.utils import to_categorical
import tensorrt as trt
import common
import tensorflow as tf

TRT_LOGGER = trt.Logger() # Utilizado para registrar mensajes e información durante la ejecución de TensorRT

def lectura_datos(data_dir, bd):
    #Datos del modelo
    if bd == 'CASIA':
        dataset = '850'
    else:
        dataset = bd
    dataset_variant = 'normal'
    nclases = 100
    mode = 'all'
    nsamples = 6
    
    #Tipos de bases de datos
    db = {
        "850": {"percent": 0.8, "ext": "jpg", "IMG_SIZE": 128, "hands": ['l', 'r']},
        "IIT": {"percent": 0.5, "ext": "png", "IMG_SIZE": 128, "hands": ['l', 'r']},
        "POLYU": {"percent": 0.6, "ext": "jpg", "IMG_SIZE": 128, "hands": ['l', 'r']},
        "PUT": {"percent": 0.8, "ext": "jpg", "IMG_SIZE": 128, "hands": ['Left', 'Right']},
        "TONGJI": {"percent": 0.8, "ext": "png", "IMG_SIZE": 64, "hands": ['l', 'r']},
        "VERA": {"percent": 0.8, "ext": "png", "IMG_SIZE": 128, "hands": ['l', 'r']},
        "SYNTHETICSPVD": {"percent": 0.8, "ext": "png", "IMG_SIZE": 64, "hands": ['l', 'r']}}
    
    if dataset in db:       
        percent, ext, IMG_SIZE, hands = db[dataset]["percent"], db[dataset]["ext"], db[dataset]["IMG_SIZE"], db[dataset]["hands"]

    #Creación de clases y etiquetas
    test_data = []
    clases = []
    if bd != 'SYNTHETICSPVD':
        for i in range(1, nclases+1):
            for h in hands:
                c = str(i).zfill(3)
                clases.append(osp.join(c, h)) 
        nclases = len(clases)
    else:
        for i in range(1, nclases+1):
        
            c = str(i).zfill(5)
            clases.append(c) 
        nclases = len(clases) 

    #Almacenamiento fotos
    for c in clases:
        aux = osp.join(data_dir, c)
        num_clase = clases.index(c)
        for s in range(int(nsamples*percent)+1, nsamples+1):
            images = glob.glob(osp.join(aux, '*_{:02d}*rot0_cx0_cy0.'.format(s)+ext))
            for img in images:
                #print(img)
                img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, num_clase])   

    return test_data, nclases, IMG_SIZE

def procesamiento_datos(test_data, nclases, IMG_SIZE):
    mode = 'all'
    test_samples = []
    test_labels = []

    for s, l in test_data:
        test_samples.append(s)
        test_labels.append(l)
    
    #Procesamiento 1
    test_samples = np.array(test_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_labels = np.array(test_labels)
    
    #Procesamiento 2
    meanTest = np.mean(test_samples, axis=0)
    test_samples = test_samples-meanTest
    test_samples = test_samples/255

    #Procesamiento 3
    test_samples = test_samples.reshape(test_samples.shape[0], IMG_SIZE, IMG_SIZE, 1)
    #test_labels = to_categorical(test_labels, nclases)
    
    return test_samples, test_labels 

def get_engine(onnx_file_path, engine_file_path=""):
    if os.path.exists(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
def inferencia(modelos_ruta, bd_ruta):
    lista_modelos_ruta = os.listdir(modelos_ruta)
    probabilidad = []
    tiempo = []
    for modelo in lista_modelos_ruta:
        if modelo != '.DS_Store':
            lista_modelos_tipo = os.listdir(osp.join(modelos_ruta, modelo))
            for modelo_tipo in lista_modelos_tipo:
                #Datos modelo
                modelo_tipo_aux = modelo_tipo
                modelo_tipo = modelo_tipo.split('_')
                nombre_modelo = modelo_tipo[0]
                precision_modelo = modelo_tipo[2]
                print(nombre_modelo, precision_modelo)
                print(osp.join(modelos_ruta, modelo, modelo_tipo_aux))
                for bd in bd_ruta:
                    if modelo.upper().split('-')[0] in bd:
                        # print('Modelo: ', modelo)
                        # print('Bd:', modelo.upper().split('-')[0], bd_ruta[bd])
                        test_data, nclases, IMG_SIZE = lectura_datos(bd_ruta[bd], bd)
                        test_samples, test_labels = procesamiento_datos(test_data, nclases, IMG_SIZE)

                        #Estructura inferencia
                        probabilidad_aux=[]
                        tiempo_aux = []
                        with get_engine("no", osp.join(modelos_ruta, modelo, modelo_tipo_aux, modelo_tipo_aux+".trt")) as engine, engine.create_execution_context() as context:
                            for sample in test_samples:
                                #Normalizacion muestras
                                aux = modelo.upper().split('-')[0] 
                                if aux == 'TONGJI' or aux == 'SYNTHETICSPVD':
                                    #sample = np.reshape(sample, (1, 64, 64, 1))
                                    sample = np.expand_dims(sample, axis=0)
                                    sample = (sample - 64) / 64
                                else:
                                    sample = np.expand_dims(sample, axis=0)
                                    sample = (sample - 128) / 128
                                    #sample = np.reshape(sample, (1, 128, 128, 1))
                                #sample = np.float32(sample)
                                sample = sample.astype(np.int8)
                                sample = tf.convert_to_tensor(sample)
                                inputs_trt, outputs_trt, bindings_trt, stream_trt = common.allocate_buffers(engine, sample)

                                #Inferencia
                                start = time.time()
                                inputs_trt[0].host = sample
                                results = common.do_inference_v2(context, bindings=bindings_trt, inputs=inputs_trt, outputs=outputs_trt, stream=stream_trt)
                                ms = time.time() - start
                                tiempo_aux.append(ms)

                                #Resultados inferencia
                                results = np.squeeze(results)
                                predicted_label = np.argmax(results)
                                probabilidad_aux.append(predicted_label) 
                                score = results[predicted_label]
                            
                            cont = 0
                            for i,j in enumerate(test_labels):
                                if test_labels[i] == probabilidad_aux[i]:
                                    cont+=1
                            probabilidad.append([modelo,precision_modelo, cont/len(probabilidad_aux)])
                            tiempo.append([modelo,precision_modelo, sum(tiempo_aux)/len(tiempo_aux)])    

    for prob in probabilidad:
        print(prob)
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    for t in tiempo:
        print(t)  

if __name__ == "__main__":
    modelos_ruta = '/home/alumnos/mgonzalez/thadli/Modelos/Trt'
    bd_ruta = {'CASIA': '/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/CASIA_MS_Palmprint_v1/augmented_roi/850/clahe/', 
                'IIT': '/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/IIT Indore Hand Vein/augmented_roi/clahe/',
                'POLYU': '/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/PolyU_MS_Palmprint/augmented_roi/clahe/',
                'PUT': '/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/PUT/augmented_roi/clahe/',
                'TONGJI': '/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/Tongji/augmented_roi/clahe/',
                'VERA': '/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/VERA/augmented_roi/clahe/',
                'SYNTHETICSPVD': '/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/LITRP Synthetic Datasets/Synthetic-sPVDB/augmented_roi/clahe/'}
    
    if os.path.exists(modelos_ruta):
        print('Existe')
        inferencia(modelos_ruta, bd_ruta)
    else:
        print('No existe la ruta', modelos_ruta)