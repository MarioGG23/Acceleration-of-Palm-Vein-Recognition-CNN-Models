import os
import os.path as osp
import pathlib
import cv2
import argparse
import numpy as np
import time
from keras.utils import np_utils
import tensorrt as trt
import common
import glob

IMG_SIZE = 128
#IMG_SIZE = 64
TRT_LOGGER = trt.Logger() # Utilizado para registrar mensajes e información durante la ejecución de TensorRT

def lectura_datos(data_dir):
    #Datos del modelo
    dataset = '850'
    dataset_variant = 'normal'
    hands = ['l', 'r']
    nclases = 100
    mode = 'all'
    nsamples = 6
    
    #Tipos de bases de datos
    db = {
        "850": {"percent": 0.8, "ext": "jpg"},
        "CASIA/940": {"percent": 0.8, "ext": "jpg"},
        "FYO": {"percent": 0.0, "ext": "png"},
        "IIT": {"percent": 0.5, "ext": "png"},
        "POLYU": {"percent": 0.6, "ext": "jpg"},
        "PUT": {"percent": 0.8, "ext": "jpg"},
        "TONGJI": {"percent": 0.8, "ext": "png"},
        "VERA": {"percent": 0.8, "ext": "png"},
        "NS-PVDB": {"percent": 0.8, "ext": "png"},
        "Synthetic-sPVDB": {"percent": 0.8, "ext": "png"}
    }
    if dataset in db: 
            percent, ext = db[dataset]["percent"], db[dataset]["ext"]
    
    #Creación de las clases
    clases = []
    if len(hands) == 0:
        for i in range(1, nclases+1):
            c = str(i).zfill(5)
            clases.append(c)
    else:
        for i in range(1, nclases+1):
            for h in hands:
                c = str(i).zfill(3)
                clases.append(osp.join(c, h))
    print(clases)                    
    nclases = len(clases)
    
    #Cración lista con las imagenes a utilizar
    test_data = []
    if mode == "ft+aug":
        pass
    elif mode == "ft":
        pass
    # else:
    #     for class_num in range(nclases): 
    #         for s in range(int(nsamples*percent)+1, nsamples+1):
    #             if (class_num) %2 == 1:
    #                 imagen = str(class_num//2 +1).zfill(3) + '_r_' + dataset + '_0' + str(s) + '_rot0_cx0_cy0' + '.' + ext
    #                 # print(class_num, imagen)
    #             else:
    #                 imagen = str(class_num//2 +1).zfill(3) + '_l_' + dataset + '_0' + str(s) + '_rot0_cx0_cy0' + '.' + ext
    #                 # print(class_num, imagen)
                    
    #             aux = osp.join(data_dir, dataset, imagen)
    #             print(aux)
    #             img_array = cv2.imread(aux, cv2.IMREAD_GRAYSCALE)
    #             new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #             test_data.append([new_array, class_num])
    test_data = []
    info = []
    for c in clases:
        path = osp.join(data_dir, c)
        class_num = clases.index(c)
        if dataset in db:
            percent, ext = db[dataset]["percent"], db[dataset]["ext"]
            # split_data(nsamples, ftsamples, path, percent, ext,
            #            class_num, mode, ft_data, test_data)
            for s in range(int(nsamples*percent)+1, nsamples+1):
            # original data for testing
                images = glob.glob(
                    osp.join(path, '*_{:02d}*rot0_cx0_cy0.'.format(s)+ext))
                info.append(images)
                for img in images:
                    img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                    test_data.append([new_array, class_num])
    return test_data, nclases

def procesamiento_datos(test_data, nclases):
    test_samples = []
    test_labels = []
    test_samples2 = []
    test_samples3 = []
    for s, l in test_data:
        test_samples.append(s)
        test_labels.append(l)
    # for i in test_samples:
    #     i = cv2.resize(i, (IMG_SIZE, IMG_SIZE))
    #     i = np.expand_dims(i, axis=0)
    #     i = i.astype(np.float16)
    #     i = i / 255.0
    #     i = i.astype(np.float16)
    #     test_samples2.append(i)
    

    test_samples = np.array(test_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_labels = np.array(test_labels)
    meanTest = np.mean(test_samples, axis=0)
    test_samples = test_samples-meanTest
    test_samples = test_samples/255
    
    # i = i.reshape(i.shape[0], IMG_SIZE, IMG_SIZE, 1)
    # test_samples3.append(i)
    # i = np.reshape(i, (1, 128, 128, 1))
    # i = np.float32(i) / 255.0
    # print(len(i[0]))


    # #Procesamiento 1
    # test_samples = np.array(test_samples).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # test_labels = np.array(test_labels)
    
    # # #Procesamiento 2
    # # meanTest = np.mean(test_samples, axis=0)
    # # test_samples = test_samples-meanTest
    # # test_samples = test_samples/255

    # #Procesamiento 3
    test_samples = test_samples.reshape(test_samples.shape[0], IMG_SIZE, IMG_SIZE, 1)
    #test_labels = np_utils.to_categorical(test_labels, nclases)

    return test_samples, test_labels    

def get_engine(onnx_file_path, engine_file_path=""):
    if os.path.exists(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
def test(model_path, test_samples, test_labels):
    
    with get_engine("no", model_path) as engine, engine.create_execution_context() as context:
        tiempo = []
        resultados = []
        for imagen in test_samples:
            imagen = np.reshape(imagen, (1, 128, 128, 1))
            imagen = np.float32(imagen)
            inputs_trt, outputs_trt, bindings_trt, stream_trt = common.allocate_buffers(engine,imagen)
            inputs_trt[0].host = imagen
            start = time.time()
            results = common.do_inference_v2(context, bindings=bindings_trt, inputs=inputs_trt, outputs=outputs_trt, stream=stream_trt)
            ms = time.time() - start
            tiempo.append(ms)
            #print(ms)
            results = np.squeeze(results)
            predicted_label = np.argmax(results)
            score = results[predicted_label]
            resultados.append(predicted_label)
            
        cont2 = 0
        print(resultados)
        for i,j in enumerate(test_labels):
            if test_labels[i] == resultados[i]:
                cont2+=1
        print(cont2/len(resultados))

        print(len(resultados))
        print(sum(tiempo))



        
if __name__ == "__main__":
    #Llamada función lectura de datos con el directorio de la bd
    # data_dir1 = osp.join(pathlib.Path().parent.absolute() , 'datasets', 'CASIA')
    # print(data_dir1)
    data_dir= "/media/nas2/LITRP.DBs/Vein_DBs/Palm-vein/CASIA_MS_Palmprint_v1/augmented_roi/850/clahe/"
    #data_dir = "/home/alumnos/mgonzalez/precision/datasets/CASIA/"
    test_data, nclases = lectura_datos(data_dir)

    # #Procesamiento datos
    test_samples, test_labels = procesamiento_datos(test_data, nclases)

    # #Creación estructura 
    parser = argparse.ArgumentParser(description='Test a CNN')
    parser.add_argument('--model_path', type=str, required=False, default="/home/alumnos/mgonzalez/modelos_trt/casia_batch1_int8.trt", help="Path to the trained model")
    args = parser.parse_args()
    model_path = args.model_path

    if os.path.exists(model_path):
        print('Existe')
        test(model_path, test_samples, test_labels)
    else:
        print('No existe la ruta', model_path)