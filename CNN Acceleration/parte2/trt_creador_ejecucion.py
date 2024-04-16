import os
import pathlib
#ruta = pathlib.Path().absolute()
#ruta = str(ruta) + '/trt_sh/' #carpeta en donde se guardaran los .sh
ruta = '/home/alumnos/mgonzalez/trt_sh/' #carpeta en donde se guardaran los .sh
ruta_modelos = '/home/alumnos/mgonzalez/onnx_models/' #ruta de los modelos .onnx
ruta_modelos_trt = '/home/alumnos/mgonzalez/modelos_trt/' # ruta en donde se guardaran los modelos trt

try:
    os.mkdir(ruta)
except FileExistsError:
    print("Carpeta ya creada")
try:
    os.mkdir(ruta_modelos_trt)
except FileExistsError:
    pass

modelos = ['casia', 'polyu', 'vera', 'iit', 'put', 'tongji', 'syntheticspvd']
tipos = ['fp32', 'fp16', 'int8']
batch_size_array = ['8', '16' ,'32', '1']  # ['1']

for modelo in modelos:
    ubicacion1 = ruta_modelos + modelo + '.onnx' #onnx
    ubicacion1_db = ruta_modelos + modelo + '_dynamic_batch.onnx'

    for batch_size in batch_size_array:

        for tipo in tipos:
            trt =  ruta + 'trt_quantization_' + tipo + '_' + modelo + '_' + batch_size + '.txt'
            file = open(trt, "w")

            if batch_size in ['8', '16', '32']:
                ubicacion2_db = ruta_modelos_trt + modelo + '_batch' + batch_size + '_' + tipo + '.trt'
                if tipo in ['fp16', 'int8']:
                    if batch_size == '32':
                        if modelo =='tongji' or modelo =='syntheticspvd':
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db + ' --' + tipo + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x64x64x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                        else:
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db + ' --' + tipo +  ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x128x128x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                    elif batch_size =='16':
                        if modelo =='tongji' or modelo =='syntheticspvd':
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db  + ' --' + tipo + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x64x64x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                        else:
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db  + ' --' + tipo  + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x128x128x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                    else:
                        if modelo =='tongji' or modelo =='syntheticspvd':
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db +  ' --' + tipo + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x64x64x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                        else:
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db + ' --' + tipo + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x128x128x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                else:
                    if batch_size == '32':
                        if modelo =='tongji' or modelo =='syntheticspvd':
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x64x64x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                        else:
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x128x128x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                    elif batch_size =='16':
                        if modelo =='tongji' or modelo =='syntheticspvd':
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db  + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x64x64x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                        else:
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db  + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x128x128x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                    else:
                        if modelo =='tongji' or modelo =='syntheticspvd':
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x64x64x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)
                        else:
                            linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db + ' --shapes=' + '\\' + "'image" + "\\':" + batch_size + 'x128x128x1'
                            linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2_db + ' > ' + modelo + '_batch' + batch_size + '_' + tipo + '.txt'
                            print(linea1)
                            print(linea2)

                file.write(linea1 + os.linesep)
                file.write(linea2)

            else:

                if tipo == 'fp32':
                    ubicacion2 = ruta_modelos_trt + modelo + '_batch1_' + tipo + '.trt'
                    linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1 + ' --saveEngine=' + ubicacion2
                    linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2 + ' > ' + modelo + '_batch1_' + tipo + '.txt'

                elif tipo == 'fp16':
                    ubicacion2 = ruta_modelos_trt + modelo + '_batch1_' + tipo + '.trt'
                    linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1 + ' --saveEngine=' + ubicacion2 + ' --fp16'
                    linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2 + ' > ' + modelo + '_batch1_' + tipo + '.txt'
                else:
                    ubicacion2 = ruta_modelos_trt + modelo + '_batch1_' + tipo + '.trt'
                    linea1 = '/usr/bin/trtexec --onnx=' + ubicacion1 + ' --saveEngine=' + ubicacion2 + ' --int8'
                    linea2 = '/usr/bin/trtexec --loadEngine=' + ubicacion2 + ' > ' + modelo + '_batch1_' + tipo + '.txt'

                file.write(linea1 + os.linesep)
                file.write(linea2)

            file.close()
            trtaux = trt.split('.')
            trt_sh = trtaux[0] + '.sh'
            os.rename(trt, trt_sh)
            comando = 'cd ' + ruta + ' && sh ' + 'trt_quantization_' + tipo + '_' + modelo + '_' + batch_size + '.sh'
            os.system(comando)